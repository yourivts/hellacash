"""Backtesting engine: simulates strategy cycle on historical candles."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

import numpy as np

from bot.exchange.bitvavo_client import CandleData
from bot.risk.fees import compute_trade_fees, get_taker_fee
from bot.indicators.volatility import atr as compute_atr
from bot.risk.position_sizer import kelly_size
from bot.risk.stop_loss import initial_stops, trail_stop, check_stop_triggered
from bot.strategy.base import MarketContext, Signal
from bot.strategy.router import StrategyRouter, detect_regime

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    entry_time: str
    exit_time: str
    size_eur: float
    pnl_eur: float
    pnl_pct: float
    exit_reason: str
    strategy: str


@dataclass
class BacktestResult:
    total_pnl: float = 0.0
    total_return_pct: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    trade_log: List[TradeRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal position tracker
# ---------------------------------------------------------------------------

@dataclass
class _OpenPosition:
    symbol: str
    direction: str
    entry_price: float
    entry_time: str
    size_eur: float
    stop_loss: float
    take_profit: float
    highest_price: float
    strategy: str
    entry_bar: int = 0
    # Range strategy fields
    tp_shifted: bool = False
    range_mid: float = 0.0
    range_upper: float = 0.0
    range_lower: float = 0.0


# ---------------------------------------------------------------------------
# BacktestEngine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """Replay candle history through the strategy router and risk system."""

    # Minimum candles needed before the engine starts generating signals.
    WARMUP = 60
    # How many 5m candles form one simulated 1h candle (12 x 5m = 60m).
    H1_FACTOR = 12

    def __init__(
        self,
        candles: List[CandleData | Dict[str, Any]],
        initial_capital: float = 10_000.0,
        max_open_positions: int = 3,
        slippage_pct: float = 0.001,
        strategy_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._raw_candles = candles
        self.initial_capital = initial_capital
        self.max_open = max_open_positions
        self.slippage_pct = slippage_pct

        self.balance = initial_capital
        self.peak_balance = initial_capital
        self.positions: List[_OpenPosition] = []
        self.closed_trades: List[TradeRecord] = []

        # Extract tunable parameters
        params = strategy_params or {}
        self._atr_multiplier = params.get("atr_multiplier", 2.0)
        self._rr_ratio = params.get("rr_ratio", 2.0)
        self._kelly_fraction = params.get("kelly_fraction", 0.25)
        self._min_confirmations = params.get("min_confirmations", 2)
        self._indicator_weights = params.get("indicator_weights")
        # Trade cooldown: minimum bars between closing a trade and opening a new one
        cooldown_hours = params.get("cooldown_hours", 72)
        self._cooldown_bars = int(cooldown_hours * 12)  # 12 x 5m bars per hour
        # Minimum ADX to allow entries (0 = disabled, let walk-forward optimize)
        self._min_adx = params.get("min_adx", 0)
        # Minimum ATR% to allow entries (0 = disabled)
        self._min_atr_pct = params.get("min_atr_pct", 0.0)
        # Max hold time (walk-forward can optimize)
        max_hold_hours = params.get("max_hold_hours", 48)
        self._max_hold_bars = int(max_hold_hours * 12)
        # Consecutive confirmation: require signal to persist N cycles before entry
        self._consecutive_confirms = params.get("consecutive_confirms", 1)

        # Range strategy: bounce counter per symbol
        self._range_bounces: Dict[str, int] = {}
        # Range strategy: max hold = 24h = 288 5m bars (hardcoded, not walk-forward)
        self._range_max_hold_bars = 288

        self._router = StrategyRouter()
        if strategy_params:
            self._router.update_hybrid_params(
                sentiment_weight=params.get("sentiment_weight", 0.25),
                entry_threshold=params.get("entry_threshold", 0.40),
                indicator_weights=self._indicator_weights,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # How often (in 5m candles) to run signal generation.
    # Stops are still checked every candle for safety.
    SIGNAL_EVERY = 4  # every 4 x 5m = 20 min (more entry opportunities)

    def run(self) -> BacktestResult:
        """Execute the full backtest and return the result summary."""
        df = self._to_dataframe(self._raw_candles)
        if len(df) < self.WARMUP:
            logger.warning("Not enough candles for backtest (%d < %d)", len(df), self.WARMUP)
            return BacktestResult()

        # Pre-compute multi-timeframe candles once
        df_1h = self._resample_1h(df)
        df_4h = self._resample_4h(df)
        df_1d = self._resample_1d(df)
        symbol = df.attrs.get("symbol", "BTC-EUR")

        # Pre-extract numpy arrays for fast per-candle access
        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        timestamps = df.index

        # Pre-compute 1h ATR for trailing stops (5m ATR is too tight)
        atr_1h_series = compute_atr(df_1h["high"], df_1h["low"], df_1h["close"]).values

        # --- Pre-compute ALL indicator signals for the full series ---
        # This replaces ~2000+ per-step pandas computations with a single batch
        precomp_5m = self._precompute_signals(df, symbol)
        precomp_1h = self._precompute_signals(df_1h, symbol)
        precomp_4h = self._precompute_signals(df_4h, symbol)
        precomp_1d = self._precompute_signals(df_1d, symbol)

        # Pre-compute regime for each 1h bar
        from bot.indicators.trend import adx as compute_adx
        regime_adx = compute_adx(df_1h["high"], df_1h["low"], df_1h["close"]).values
        regime_atr = compute_atr(df_1h["high"], df_1h["low"], df_1h["close"]).values
        regime_close = df_1h["close"].values
        h1_timestamps = df_1h.index
        h4_timestamps = df_4h.index
        h1d_timestamps = df_1d.index

        equity_curve: List[float] = []
        last_trade_close_bar = -self._cooldown_bars  # allow first trade immediately
        # Consecutive confirmation state
        _prev_signal_dir: Optional[str] = None
        _signal_streak: int = 0

        for i in range(self.WARMUP, len(df)):
            current_price = closes[i]
            current_time = str(timestamps[i])

            # --- check stops on existing positions (every candle) ---
            if self.positions:
                # Map 5m bar to corresponding 1h bar for ATR lookup
                current_ts = timestamps[i]
                _h1_idx = h1_timestamps.searchsorted(current_ts, side="right") - 1
                atr_val = atr_1h_series[_h1_idx] if 0 <= _h1_idx < len(atr_1h_series) else None
                prev_closed = len(self.closed_trades)
                # Pass pre-computed 5m data for range dynamic TP
                has_range = any(p.strategy == "range" for p in self.positions)
                self._check_exits_fast(
                    current_price, highs[i], lows[i], current_time, atr_val,
                    current_bar=i,
                    precomp_5m=precomp_5m if has_range else None,
                    idx_5m=i if has_range else 0,
                )
                if len(self.closed_trades) > prev_closed:
                    last_trade_close_bar = i

            # Update peak equity every candle for accurate drawdown tracking
            current_equity = self._equity(current_price)
            if current_equity > self.peak_balance:
                self.peak_balance = current_equity

            # --- signal generation (only every SIGNAL_EVERY candles) ---
            if (i - self.WARMUP) % self.SIGNAL_EVERY != 0:
                equity_curve.append(current_equity)
                continue

            # Trade cooldown: wait after a trade closes before opening a new one
            if i - last_trade_close_bar < self._cooldown_bars:
                equity_curve.append(current_equity)
                continue

            # Find corresponding 1h bar index
            current_ts = timestamps[i]
            h1_idx = h1_timestamps.searchsorted(current_ts, side="right") - 1
            if h1_idx < 30:
                equity_curve.append(current_equity)
                continue

            # Fast regime detection from pre-computed arrays
            adx_val = regime_adx[h1_idx]
            atr_val_h = regime_atr[h1_idx]
            price_h = regime_close[h1_idx]
            atr_pct = (atr_val_h / price_h) * 100.0 if price_h > 0 else 0.0
            if atr_pct > 3.0:
                regime = "volatile"
            elif adx_val > 25:
                regime = "trending"
            elif adx_val < 20:
                regime = "ranging"
            else:
                regime = "unknown"

            # Gate: skip entries when ADX is too low (no clear trend)
            if adx_val < self._min_adx:
                equity_curve.append(current_equity)
                continue

            # Gate: skip entries when volatility is too low to cover fees
            if atr_pct < self._min_atr_pct:
                equity_curve.append(current_equity)
                continue

            strategies = self._router.get_strategies(regime)
            size_modifier = self._router.position_size_modifier(regime)

            # Find corresponding 4h and 1d bar indices
            h4_idx = h4_timestamps.searchsorted(current_ts, side="right") - 1
            h1d_idx = h1d_timestamps.searchsorted(current_ts, side="right") - 1

            # Look up pre-computed signal scores instead of recomputing
            best_signal = self._evaluate_precomputed(
                strategies, symbol, current_price, i, h1_idx,
                precomp_5m, precomp_1h, regime,
                h4_idx=h4_idx, h1d_idx=h1d_idx,
                precomp_4h=precomp_4h, precomp_1d=precomp_1d,
            )

            # --- consecutive confirmation tracking ---
            if best_signal is not None and best_signal.direction in ("LONG", "SHORT"):
                if best_signal.direction == _prev_signal_dir:
                    _signal_streak += 1
                else:
                    _prev_signal_dir = best_signal.direction
                    _signal_streak = 1
            else:
                _prev_signal_dir = None
                _signal_streak = 0

            # --- entry logic ---
            if (
                best_signal is not None
                and best_signal.direction in ("LONG", "SHORT")
                and best_signal.strength > 0
                and best_signal.indicator_snapshot.get("confirming_count", 0) >= self._min_confirmations
                and _signal_streak >= self._consecutive_confirms
                and len(self.positions) < self.max_open
            ):
                # Use 1h candles for ATR-based stop calculation (5m ATR is too tight)
                window_1h = df_1h.iloc[max(0, h1_idx - 100): h1_idx + 1]
                self._open_position(
                    best_signal, current_price, current_time,
                    window_1h, size_modifier, bar_index=i,
                )

            equity_curve.append(current_equity)

        # Close any remaining positions at last price
        if len(df) > 0:
            last_price = closes[-1]
            last_time = str(timestamps[-1])
            for pos in list(self.positions):
                self._close_position(pos, last_price, last_time, "end_of_data")

        return self._compile_result(equity_curve)

    # ------------------------------------------------------------------
    # Pre-computation helpers (batch indicator calculation)
    # ------------------------------------------------------------------

    @staticmethod
    def _precompute_signals(df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Compute all indicator series once on the full DataFrame."""
        from bot.indicators.divergence import rsi_divergence, volume_divergence
        from bot.indicators.momentum import rsi, stochastic, cci
        from bot.indicators.trend import ema, macd, adx, supertrend
        from bot.indicators.volatility import bollinger_bands, atr
        from bot.indicators.volume import volume_surge_ratio, cmf

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        result: Dict[str, Any] = {}
        try:
            result["rsi"] = rsi(close).values
        except Exception:
            result["rsi"] = np.zeros(len(df))
        try:
            m = macd(close)
            result["macd_hist"] = m["histogram"].values
        except Exception:
            result["macd_hist"] = np.zeros(len(df))
        try:
            bb = bollinger_bands(close)
            result["bb_pct_b"] = bb["pct_b"].values
            result["bb_upper"] = bb["upper"].values
            result["bb_lower"] = bb["lower"].values
            result["bb_mid"] = bb["mid"].values
            result["bb_bandwidth"] = bb["bandwidth"].values
        except Exception:
            result["bb_pct_b"] = np.full(len(df), 0.5)
            result["bb_upper"] = close.values.copy()
            result["bb_lower"] = close.values.copy()
            result["bb_mid"] = close.values.copy()
            result["bb_bandwidth"] = np.zeros(len(df))
        try:
            result["ema20"] = ema(close, 20).values
            result["ema50"] = ema(close, 50).values
        except Exception:
            result["ema20"] = close.values.copy()
            result["ema50"] = close.values.copy()
        try:
            result["ema200"] = ema(close, 200).values
        except Exception:
            result["ema200"] = close.values.copy()
        try:
            result["supertrend"] = supertrend(high, low, close).values
        except Exception:
            result["supertrend"] = np.zeros(len(df))
        try:
            result["adx"] = adx(high, low, close).values
        except Exception:
            result["adx"] = np.full(len(df), 25.0)
        try:
            result["vsr"] = volume_surge_ratio(volume).values
        except Exception:
            result["vsr"] = np.ones(len(df))
        try:
            result["cci"] = cci(high, low, close).values
        except Exception:
            result["cci"] = np.zeros(len(df))

        # Leading indicators
        try:
            result["rsi_div"] = rsi_divergence(close).values
        except Exception:
            result["rsi_div"] = np.zeros(len(df))
        try:
            result["vol_div"] = volume_divergence(close, volume).values
        except Exception:
            result["vol_div"] = np.zeros(len(df))

        # Volume profile for range trading
        try:
            from bot.indicators.volume import volume_profile_support
            result["vol_profile"] = volume_profile_support(close, volume, high, low).values
        except Exception:
            result["vol_profile"] = np.zeros(len(df))

        result["close"] = close.values
        result["high"] = high.values
        result["low"] = low.values

        return result

    def _score_from_precomputed(self, p: Dict[str, Any], idx: int) -> float:
        """Compute composite technical score from pre-computed indicator arrays."""
        if idx < 0 or idx >= len(p["close"]):
            return 0.0

        scores = {}
        # RSI
        r = p["rsi"][idx]
        if r < 30:
            scores["rsi"] = 1.0
        elif r < 40:
            scores["rsi"] = 0.5
        elif r > 70:
            scores["rsi"] = -1.0
        elif r > 60:
            scores["rsi"] = -0.5
        else:
            scores["rsi"] = 0.0

        # MACD
        hist = p["macd_hist"][idx]
        prev_hist = p["macd_hist"][idx - 1] if idx > 0 else 0.0
        if hist > 0 and hist > prev_hist:
            scores["macd"] = 1.0
        elif hist > 0:
            scores["macd"] = 0.4
        elif hist < 0 and hist < prev_hist:
            scores["macd"] = -1.0
        elif hist < 0:
            scores["macd"] = -0.4
        else:
            scores["macd"] = 0.0

        # Bollinger
        pct_b = p["bb_pct_b"][idx]
        if pct_b < 0.05:
            scores["bollinger"] = 1.0
        elif pct_b < 0.2:
            scores["bollinger"] = 0.5
        elif pct_b > 0.95:
            scores["bollinger"] = -1.0
        elif pct_b > 0.8:
            scores["bollinger"] = -0.5
        else:
            scores["bollinger"] = 0.0

        # EMA trend (4 conditions to match composite.py)
        price = p["close"][idx]
        e20 = p["ema20"][idx]
        e50 = p["ema50"][idx]
        e200 = p["ema200"][idx]
        bullish = sum([price > e20, price > e50, price > e200, e20 > e50])
        scores["ema_trend"] = (bullish - 2) / 2  # maps 0-4 to -1..+1

        # Supertrend
        scores["supertrend"] = float(p["supertrend"][idx])

        # Volume
        vsr = p["vsr"][idx]
        if idx > 0:
            price_change = (p["close"][idx] - p["close"][idx - 1]) / p["close"][idx - 1] if p["close"][idx - 1] > 0 else 0
        else:
            price_change = 0.0
        if vsr > 1.5:
            scores["volume"] = 1.0 if price_change > 0 else -1.0
        elif vsr > 1.2:
            scores["volume"] = 0.5 if price_change > 0 else -0.5
        else:
            scores["volume"] = 0.0

        # CCI
        c = p["cci"][idx]
        if c < -100:
            scores["cci"] = 1.0
        elif c < -50:
            scores["cci"] = 0.4
        elif c > 100:
            scores["cci"] = -1.0
        elif c > 50:
            scores["cci"] = -0.4
        else:
            scores["cci"] = 0.0

        # Leading: RSI divergence
        rsi_div = p["rsi_div"][idx] if "rsi_div" in p else 0.0
        scores["rsi_divergence"] = float(rsi_div)

        # Leading: Volume divergence
        vol_div = p["vol_div"][idx] if "vol_div" in p else 0.0
        scores["volume_divergence"] = float(vol_div)

        # ADX modifier
        a = p["adx"][idx]
        adx_multiplier = min(a / 25.0, 2.0) if a > 20 else 0.5

        # Weighted composite (use params or defaults matching composite.py)
        w = self._indicator_weights or {
            "rsi": 0.15, "macd": 0.15, "bollinger": 0.10, "ema_trend": 0.10,
            "supertrend": 0.10, "volume": 0.05, "cci": 0.05,
            "rsi_divergence": 0.15, "volume_divergence": 0.10,
        }
        raw = sum(scores.get(k, 0) * w[k] for k in w)
        total_w = sum(w.values())
        if total_w > 0:
            raw /= total_w
        raw = max(-1.0, min(1.0, raw * adx_multiplier))
        return raw

    def _evaluate_precomputed(
        self, strategies, symbol, current_price, idx_5m, idx_1h,
        precomp_5m, precomp_1h, regime,
        h4_idx=0, h1d_idx=0, precomp_4h=None, precomp_1d=None,
    ) -> Optional[Signal]:
        """Evaluate strategies using pre-computed indicator values."""
        from bot.strategy.mtf_voter import REGIME_WEIGHTS, BASE_WEIGHTS

        best_signal: Optional[Signal] = None

        # Higher-timeframe trend filters
        h1_ema20 = precomp_1h["ema20"][idx_1h] if idx_1h < len(precomp_1h["ema20"]) else 0
        h1_ema50 = precomp_1h["ema50"][idx_1h] if idx_1h < len(precomp_1h["ema50"]) else 0
        h1_ema200 = precomp_1h["ema200"][idx_1h] if idx_1h < len(precomp_1h["ema200"]) else 0
        h1_trend_bull = h1_ema20 > h1_ema50
        h1_trend_bear = h1_ema20 < h1_ema50
        # EMA200 regime: determines allowed trade direction for directional strategies
        price_above_ema200 = current_price > h1_ema200 if h1_ema200 > 0 else True
        price_below_ema200 = current_price < h1_ema200 if h1_ema200 > 0 else True
        # Strong trend filter: both EMA20/50 alignment + EMA200
        strong_bull = h1_trend_bull and price_above_ema200
        strong_bear = h1_trend_bear and price_below_ema200

        for strat in strategies:
            try:
                if strat.name == "hybrid":
                    # Replicate MTF voter logic with pre-computed scores
                    score_5m = self._score_from_precomputed(precomp_5m, idx_5m)
                    score_1h = self._score_from_precomputed(precomp_1h, idx_1h)

                    score_4h = self._score_from_precomputed(precomp_4h, h4_idx) if precomp_4h and h4_idx >= 0 else score_1h
                    score_1d = self._score_from_precomputed(precomp_1d, h1d_idx) if precomp_1d and h1d_idx >= 0 else score_1h
                    tf_scores = {"15m": score_5m, "1h": score_1h, "4h": score_4h, "1d": score_1d}
                    weights = REGIME_WEIGHTS.get(regime, BASE_WEIGHTS)
                    total_w = sum(weights.values())
                    mtf_score = sum(tf_scores[tf] * weights[tf] for tf in tf_scores) / total_w if total_w > 0 else 0.0
                    mtf_score = max(-1.0, min(1.0, mtf_score))

                    # Agreement check
                    non_neutral = {tf: s for tf, s in tf_scores.items() if abs(s) >= 0.15}
                    if len(non_neutral) >= 2:
                        bullish = sum(1 for s in non_neutral.values() if s > 0)
                        agreement = max(bullish, len(non_neutral) - bullish) / len(non_neutral)
                        if agreement < 0.5:
                            mtf_score *= 0.5

                    # Tech weight dominates in backtest (no sentiment/onchain/orderbook)
                    final_score = mtf_score
                    strength = min(abs(final_score), 1.0)
                    threshold = strat.entry_threshold
                    if final_score > threshold:
                        direction = "LONG"
                    elif final_score < -threshold:
                        direction = "SHORT"
                    else:
                        direction = "NEUTRAL"

                    confirming = sum(1 for s in tf_scores.values() if s * final_score > 0)
                    sig = Signal(
                        symbol=symbol, direction=direction, strength=strength,
                        strategy_name="hybrid", technical_score=final_score,
                        indicator_snapshot={"confirming_count": confirming},
                    )

                elif strat.name == "breakout":
                    # Breakout: price exceeds recent 5m range with volume confirmation
                    lookback = 20
                    if idx_5m < lookback + 5:
                        continue
                    h = precomp_5m["high"]
                    l = precomp_5m["low"]
                    recent_high = np.max(h[idx_5m - lookback:idx_5m])
                    recent_low = np.min(l[idx_5m - lookback:idx_5m])
                    price = current_price
                    vsr = precomp_5m["vsr"][idx_5m]

                    direction = "NEUTRAL"
                    strength = 0.0
                    if price > recent_high and vsr >= 1.3:
                        direction = "LONG"
                        strength = min(vsr / 3.0, 1.0)
                    elif price < recent_low and vsr >= 1.3:
                        direction = "SHORT"
                        strength = min(vsr / 3.0, 1.0)

                    # EMA200 trend filter: block counter-trend breakouts
                    if direction == "LONG" and not price_above_ema200:
                        direction = "NEUTRAL"
                        strength = 0.0
                    elif direction == "SHORT" and not price_below_ema200:
                        direction = "NEUTRAL"
                        strength = 0.0

                    confirming = 0
                    if direction != "NEUTRAL":
                        confirming = 1  # volume confirmed
                        if precomp_5m["adx"][idx_5m] > 20:
                            confirming += 1
                        macd_agrees = (precomp_5m["macd_hist"][idx_5m] > 0) == (direction == "LONG")
                        if macd_agrees:
                            confirming += 1

                    sig = Signal(symbol=symbol, direction=direction, strength=strength,
                                 strategy_name="breakout", technical_score=strength if direction == "LONG" else -strength,
                                 indicator_snapshot={"confirming_count": confirming})

                elif strat.name == "range":
                    # Range trading: BB + RSI + volume profile at 5m, BB bandwidth + ADX at 1h
                    if idx_5m < 200:
                        continue

                    # 1h range confirmation
                    bw_1h = precomp_1h["bb_bandwidth"][idx_1h] if idx_1h < len(precomp_1h.get("bb_bandwidth", [])) else 1.0
                    adx_1h = precomp_1h["adx"][idx_1h] if idx_1h < len(precomp_1h["adx"]) else 50.0

                    # Check bounce counter reset
                    if bw_1h > 0.15 or adx_1h > 25:
                        self._range_bounces[symbol] = 0

                    # Range not confirmed
                    if bw_1h >= 0.10 or bw_1h < 0.02 or adx_1h >= 20:
                        continue

                    # Bounce limit
                    if self._range_bounces.get(symbol, 0) >= 3:
                        continue

                    # 5m entry trigger
                    price = current_price
                    bb_upper = precomp_5m["bb_upper"][idx_5m]
                    bb_lower = precomp_5m["bb_lower"][idx_5m]
                    bb_mid = precomp_5m["bb_mid"][idx_5m]
                    rsi_val = precomp_5m["rsi"][idx_5m]
                    vol_score = precomp_5m["vol_profile"][idx_5m]

                    direction = "NEUTRAL"
                    confirming = 0

                    # LONG: price within 1% of lower BB + RSI < 40 + volume support
                    if bb_lower > 0 and abs(price - bb_lower) / bb_lower <= 0.01 and rsi_val < 40:
                        if vol_score > 0.3:
                            direction = "LONG"
                            confirming = 1
                            if abs(price - bb_lower) / bb_lower <= 0.005:
                                confirming += 1
                            if vol_score > 0.5:
                                confirming += 1
                    # SHORT: price within 1% of upper BB + RSI > 60 + volume resistance
                    elif bb_upper > 0 and abs(price - bb_upper) / bb_upper <= 0.01 and rsi_val > 60:
                        if vol_score < -0.3:
                            direction = "SHORT"
                            confirming = 1
                            if abs(price - bb_upper) / bb_upper <= 0.005:
                                confirming += 1
                            if vol_score < -0.5:
                                confirming += 1

                    strength = 0.0
                    if direction != "NEUTRAL":
                        strength = min(abs(vol_score) + (1.0 - bw_1h / 0.10) * 0.5, 1.0)

                    sig = Signal(
                        symbol=symbol, direction=direction, strength=strength,
                        strategy_name="range",
                        technical_score=strength if direction == "LONG" else -strength,
                        indicator_snapshot={
                            "confirming_count": confirming,
                            "range_mid": float(bb_mid),
                            "range_upper": float(bb_upper),
                            "range_lower": float(bb_lower),
                            "bounce_count": self._range_bounces.get(symbol, 0),
                        },
                    )

                elif strat.name == "trend_following":
                    # Trend following: EMA alignment + ADX strength + MACD
                    if idx_5m < 52:
                        continue
                    e20 = precomp_5m["ema20"]
                    e50 = precomp_5m["ema50"]
                    a = precomp_5m["adx"][idx_5m]
                    trending = a > 20
                    macd_bull = precomp_5m["macd_hist"][idx_5m] > 0
                    macd_bear = precomp_5m["macd_hist"][idx_5m] < 0

                    ema_bull = e20[idx_5m] > e50[idx_5m]
                    ema_bear = e20[idx_5m] < e50[idx_5m]
                    spread_pct = abs(e20[idx_5m] - e50[idx_5m]) / e50[idx_5m] * 100 if e50[idx_5m] > 0 else 0
                    aligned = spread_pct > 0.1

                    direction = "NEUTRAL"
                    strength = 0.0
                    # Require 1h EMA200 agreement: only trade with the macro trend
                    if ema_bull and macd_bull and trending and aligned and price_above_ema200:
                        direction = "LONG"
                        strength = min(a / 50.0, 1.0)
                    elif ema_bear and macd_bear and trending and aligned and price_below_ema200:
                        direction = "SHORT"
                        strength = min(a / 50.0, 1.0)

                    confirming = 0
                    if direction != "NEUTRAL":
                        confirming = 1  # EMA aligned
                        if trending:
                            confirming += 1
                        if (macd_bull and direction == "LONG") or (macd_bear and direction == "SHORT"):
                            confirming += 1

                    sig = Signal(symbol=symbol, direction=direction, strength=strength,
                                 strategy_name="trend_following",
                                 technical_score=strength if direction == "LONG" else -strength,
                                 indicator_snapshot={"confirming_count": confirming})
                else:
                    continue

                if sig.direction in ("LONG", "SHORT") and sig.strength > 0:
                    # Penalize counter-trend signals based on EMA200
                    effective_strength = sig.strength
                    if sig.direction == "LONG" and price_below_ema200:
                        effective_strength *= 0.3  # heavily penalize counter-trend longs
                    elif sig.direction == "SHORT" and price_above_ema200:
                        effective_strength *= 0.3  # heavily penalize counter-trend shorts

                    if best_signal is None or effective_strength > best_signal.strength:
                        sig.strength = effective_strength
                        best_signal = sig
            except Exception:
                continue

        return best_signal

    # ------------------------------------------------------------------
    # Position management helpers
    # ------------------------------------------------------------------

    def _compute_position_size(self, size_modifier: float, current_price: float) -> float:
        """Compute position size in EUR with drawdown scaling.

        Uses equity (cash + unrealized P&L) for drawdown calculation,
        matching live DrawdownGuard.position_size_multiplier behavior.
        """
        win_rate, avg_win, avg_loss = self._trade_stats()
        raw_size = kelly_size(
            win_rate=win_rate,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            portfolio_eur=self.balance,
            kelly_fraction=self._kelly_fraction,
        )
        size_eur = raw_size * size_modifier

        # Drawdown scaling (matches live DrawdownGuard.position_size_multiplier)
        equity = self._equity(current_price)
        drawdown_pct = ((self.peak_balance - equity) / self.peak_balance * 100.0
                        if self.peak_balance > 0 else 0.0)
        if drawdown_pct >= 3.0:
            size_eur *= 0.5

        return size_eur

    def _open_position(
        self,
        signal: Signal,
        price: float,
        time_str: str,
        df_window: pd.DataFrame,
        size_modifier: float,
        bar_index: int = 0,
    ) -> None:
        size_eur = self._compute_position_size(size_modifier, current_price=price)
        if size_eur < 10.0 or size_eur > self.balance:
            return  # skip tiny or over-sized trades

        # Deduct cost (including entry fee)
        symbol = signal.symbol
        taker_fee = get_taker_fee(symbol)
        cost = size_eur * (1 + taker_fee)
        if cost > self.balance:
            return
        self.balance -= cost

        # Stops (direction-aware)
        direction = signal.direction
        sl, tp = initial_stops(price, df_window, direction=direction,
                               atr_multiplier=self._atr_multiplier,
                               rr_ratio=self._rr_ratio)

        # Apply slippage to entry price (worse entry simulates real-world fill)
        if direction == "SHORT":
            slipped_price = price * (1 - self.slippage_pct)  # worse entry for short (sell lower)
        else:
            slipped_price = price * (1 + self.slippage_pct)  # worse entry for long (buy higher)

        self.positions.append(_OpenPosition(
            symbol=symbol,
            direction=direction,
            entry_price=slipped_price,
            entry_time=time_str,
            size_eur=size_eur,
            stop_loss=sl,
            take_profit=tp,
            highest_price=slipped_price,
            strategy=signal.strategy_name,
            entry_bar=bar_index,
        ))

        # Store range levels for range strategy positions
        if signal.strategy_name == "range":
            pos = self.positions[-1]
            snap = signal.indicator_snapshot
            pos.range_mid = snap.get("range_mid", 0.0)
            pos.range_upper = snap.get("range_upper", 0.0)
            pos.range_lower = snap.get("range_lower", 0.0)

            # Override ATR-based stops with range-specific stops
            bandwidth_price = pos.range_upper - pos.range_lower
            if pos.direction == "LONG":
                pos.stop_loss = pos.range_lower - 0.5 * bandwidth_price
                pos.take_profit = pos.range_mid  # Phase 1: TP at mid-band
            else:
                pos.stop_loss = pos.range_upper + 0.5 * bandwidth_price
                pos.take_profit = pos.range_mid

    def _check_exits_fast(
        self,
        price: float,
        candle_high: float,
        candle_low: float,
        time_str: str,
        atr_val: Optional[float],
        current_bar: int = 0,
        precomp_5m: Optional[Dict[str, Any]] = None,
        idx_5m: int = 0,
    ) -> None:
        """Fast exit check using pre-computed values — no pandas overhead."""
        for pos in list(self.positions):
            direction = pos.direction

            # Time-based exit: range uses 24h, others use max_hold_bars
            if pos.strategy == "range":
                max_hold = self._range_max_hold_bars
            else:
                max_hold = self._max_hold_bars
            if current_bar - pos.entry_bar >= max_hold:
                self._close_position(pos, price, time_str, "time_exit")
                continue

            # --- Range dynamic TP: shift TP from mid-band to opposite band ---
            if pos.strategy == "range" and not pos.tp_shifted and precomp_5m is not None:
                crossed_mid = (
                    (direction == "LONG" and price >= pos.range_mid) or
                    (direction == "SHORT" and price <= pos.range_mid)
                )
                if crossed_mid and idx_5m >= 3:
                    current_rsi = precomp_5m["rsi"][idx_5m]
                    prev_rsi = precomp_5m["rsi"][idx_5m - 3]  # 3 bars back (15 min at 5m)
                    # RSI trending favorably?
                    if direction == "LONG" and current_rsi > prev_rsi:
                        pos.tp_shifted = True
                        pos.take_profit = pos.range_upper
                    elif direction == "SHORT" and current_rsi < prev_rsi:
                        pos.tp_shifted = True
                        pos.take_profit = pos.range_lower
                    else:
                        # RSI flat/reversing: close at mid-band
                        self._close_position(pos, pos.range_mid, time_str, "range_mid_exit")
                        continue

            # Update trailing price (highest for LONG, lowest for SHORT)
            if direction == "SHORT":
                if price < pos.highest_price:
                    pos.highest_price = price
            else:
                if price > pos.highest_price:
                    pos.highest_price = price

            # Trailing stop logic
            if pos.strategy == "range" and pos.tp_shifted:
                # Tight 1% trailing stop for range positions after TP shift
                if direction == "LONG":
                    trail_level = pos.highest_price * 0.99
                    if trail_level > pos.stop_loss:
                        pos.stop_loss = trail_level
                elif direction == "SHORT":
                    trail_level = pos.highest_price * 1.01
                    if trail_level < pos.stop_loss:
                        pos.stop_loss = trail_level
            elif atr_val is not None:
                # Standard ATR trailing stop for non-range positions
                trail_dist = atr_val * self._atr_multiplier
                if direction == "SHORT":
                    in_profit = pos.entry_price - price
                else:
                    in_profit = price - pos.entry_price
                risk = abs(pos.entry_price - pos.stop_loss) if abs(pos.entry_price - pos.stop_loss) > 0 else trail_dist

                if in_profit >= risk:
                    pos.stop_loss = trail_stop(
                        price, pos.highest_price, pos.stop_loss,
                        trail_dist,
                        direction=direction,
                    )

            # Direction-aware stop checks
            triggered = None
            if direction == "SHORT":
                if candle_high >= pos.stop_loss:
                    triggered = "stop_loss"
                    price_used = pos.stop_loss
                elif candle_low <= pos.take_profit:
                    triggered = "take_profit"
                    price_used = pos.take_profit
            else:
                if candle_low <= pos.stop_loss:
                    triggered = "stop_loss"
                    price_used = pos.stop_loss
                elif candle_high >= pos.take_profit:
                    triggered = "take_profit"
                    price_used = pos.take_profit

            if triggered:
                self._close_position(pos, price_used, time_str, triggered)

    def _close_position(
        self,
        pos: _OpenPosition,
        exit_price: float,
        time_str: str,
        reason: str,
    ) -> None:
        if pos not in self.positions:
            return
        self.positions.remove(pos)

        # Increment range bounce counter on closed range trades
        if pos.strategy == "range":
            self._range_bounces[pos.symbol] = self._range_bounces.get(pos.symbol, 0) + 1

        quantity = pos.size_eur / pos.entry_price
        direction = pos.direction

        # Apply slippage to exit price (worse exit simulates real-world fill)
        if direction == "SHORT":
            slipped_exit = exit_price * (1 + self.slippage_pct)  # worse exit for short (buy higher)
        else:
            slipped_exit = exit_price * (1 - self.slippage_pct)  # worse exit for long (sell lower)

        # Direction-aware gross P&L
        if direction == "SHORT":
            price_change_pct = (pos.entry_price - slipped_exit) / pos.entry_price
        else:
            price_change_pct = (slipped_exit - pos.entry_price) / pos.entry_price

        gross_pnl = pos.size_eur * price_change_pct

        # Use actual per-market fees (exit fee only — entry fee already deducted)
        exit_fee = slipped_exit * quantity * get_taker_fee(pos.symbol)
        pnl = gross_pnl - exit_fee
        pnl_pct = price_change_pct * 100.0

        # Return capital + P&L
        self.balance += pos.size_eur + pnl

        self.closed_trades.append(TradeRecord(
            symbol=pos.symbol,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            entry_time=pos.entry_time,
            exit_time=time_str,
            size_eur=pos.size_eur,
            pnl_eur=round(pnl, 4),
            pnl_pct=round(pnl_pct, 4),
            exit_reason=reason,
            strategy=pos.strategy,
        ))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _equity(self, current_price: float) -> float:
        """Total equity = cash + mark-to-market value of open positions."""
        pos_value = 0.0
        for p in self.positions:
            if p.direction == "SHORT":
                # Collateral + unrealized P&L
                change = (p.entry_price - current_price) / p.entry_price
                pos_value += p.size_eur * (1 + change)
            else:
                pos_value += p.size_eur * (current_price / p.entry_price)
        return self.balance + pos_value

    MIN_TRADES_FOR_STATS = 10

    def _trade_stats(self) -> tuple[float, float, float]:
        """Return (win_rate, avg_win_pct, avg_loss_pct) from closed trades."""
        if len(self.closed_trades) < self.MIN_TRADES_FOR_STATS:
            return 0.0, 0.0, 0.0
        wins = [t for t in self.closed_trades if t.pnl_eur > 0]
        losses = [t for t in self.closed_trades if t.pnl_eur <= 0]
        win_rate = len(wins) / len(self.closed_trades) if self.closed_trades else 0.0
        avg_win = (sum(t.pnl_pct for t in wins) / len(wins) / 100.0) if wins else 0.02
        avg_loss = (sum(abs(t.pnl_pct) for t in losses) / len(losses) / 100.0) if losses else 0.02
        return win_rate, avg_win, avg_loss

    def _compile_result(self, equity_curve: List[float]) -> BacktestResult:
        trades = self.closed_trades
        total_pnl = sum(t.pnl_eur for t in trades)
        total_return = (total_pnl / self.initial_capital) * 100.0 if self.initial_capital else 0.0

        wins = [t for t in trades if t.pnl_eur > 0]
        losses = [t for t in trades if t.pnl_eur <= 0]

        win_rate = (len(wins) / len(trades) * 100.0) if trades else 0.0
        avg_win = (sum(t.pnl_pct for t in wins) / len(wins)) if wins else 0.0
        avg_loss = (sum(t.pnl_pct for t in losses) / len(losses)) if losses else 0.0

        # Sharpe ratio (annualised, from per-bar equity returns)
        sharpe = 0.0
        if len(equity_curve) > 1:
            returns = []
            for j in range(1, len(equity_curve)):
                if equity_curve[j - 1] > 0:
                    returns.append(equity_curve[j] / equity_curve[j - 1] - 1)
            if returns:
                mean_r = sum(returns) / len(returns)
                std_r = (sum((r - mean_r) ** 2 for r in returns) / len(returns)) ** 0.5
                if std_r > 0:
                    # Annualise assuming 5m bars (~105,120 bars/year)
                    sharpe = (mean_r / std_r) * math.sqrt(105_120)

        # Max drawdown
        max_dd = 0.0
        peak = 0.0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = ((peak - eq) / peak * 100.0) if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd

        return BacktestResult(
            total_pnl=round(total_pnl, 2),
            total_return_pct=round(total_return, 2),
            win_rate=round(win_rate, 2),
            total_trades=len(trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            sharpe_ratio=round(sharpe, 2),
            max_drawdown_pct=round(max_dd, 2),
            avg_win_pct=round(avg_win, 2),
            avg_loss_pct=round(avg_loss, 2),
            trade_log=trades,
        )

    @staticmethod
    def _to_dataframe(candles: List[CandleData | Dict[str, Any]]) -> pd.DataFrame:
        """Convert list of CandleData (or dicts) to an OHLCV DataFrame."""
        rows = []
        symbol = "BTC-EUR"
        for c in candles:
            if isinstance(c, dict):
                rows.append({
                    "timestamp": c.get("timestamp", c.get("time")),
                    "open": float(c["open"]),
                    "high": float(c["high"]),
                    "low": float(c["low"]),
                    "close": float(c["close"]),
                    "volume": float(c.get("volume", 0)),
                })
            else:
                symbol = c.symbol
                rows.append({
                    "timestamp": c.timestamp,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                })
        df = pd.DataFrame(rows)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        df.attrs["symbol"] = symbol
        return df

    @staticmethod
    def _resample_1h(df_5m: pd.DataFrame) -> pd.DataFrame:
        """Resample 5m candles into 1h candles."""
        if df_5m.empty:
            return df_5m
        ohlc = df_5m.resample("1h").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()
        return ohlc

    @staticmethod
    def _resample_4h(df_5m: pd.DataFrame) -> pd.DataFrame:
        """Resample 5m candles into 4h candles."""
        if df_5m.empty:
            return df_5m
        return df_5m.resample("4h").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()

    @staticmethod
    def _resample_1d(df_5m: pd.DataFrame) -> pd.DataFrame:
        """Resample 5m candles into 1d candles."""
        if df_5m.empty:
            return df_5m
        return df_5m.resample("1D").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()


# ---------------------------------------------------------------------------
# Convenience async function
# ---------------------------------------------------------------------------

async def run_backtest(
    symbol: str = "BTC-EUR",
    interval: str = "5m",
    days: int = 30,
    initial_capital: float = 10_000.0,
    max_open_positions: int = 3,
    slippage_pct: float = 0.001,
) -> BacktestResult:
    """
    Fetch historical candles from Bitvavo public API and run the backtest.

    Bitvavo's public candle endpoint has a limit of 1440 per request, so we
    paginate backwards to cover the requested date range.
    """
    import asyncio
    import json as _json
    import urllib.request
    from datetime import timedelta

    candles: List[CandleData] = []
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

    # Interval durations in milliseconds (approximate)
    interval_ms_map = {
        "1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
        "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
    }
    bar_ms = interval_ms_map.get(interval, 300_000)
    limit = 1440

    cursor = end_ms
    while cursor > start_ms:
        url = (
            f"https://api.bitvavo.com/v2/{symbol}/candles"
            f"?interval={interval}&limit={limit}"
            f"&start={start_ms}&end={cursor}"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "hellacash/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = _json.loads(resp.read().decode())
        except Exception as e:
            logger.error("Backtest candle fetch failed: %s", e)
            break

        if not raw:
            break

        batch = []
        for c in raw:
            batch.append(CandleData(
                symbol=symbol,
                interval=interval,
                timestamp=datetime.fromtimestamp(c[0] / 1000, tz=timezone.utc),
                open=float(c[1]),
                high=float(c[2]),
                low=float(c[3]),
                close=float(c[4]),
                volume=float(c[5]),
            ))

        candles.extend(batch)

        # Bitvavo returns newest first; find the oldest timestamp we got
        oldest_ts = min(c[0] for c in raw)
        if oldest_ts <= start_ms:
            break
        cursor = oldest_ts - 1

        # Rate-limit courtesy
        await asyncio.sleep(0.15)

    if not candles:
        logger.warning("No candles fetched for %s", symbol)
        return BacktestResult()

    # Deduplicate by timestamp and sort
    seen = set()
    unique: List[CandleData] = []
    for c in candles:
        key = c.timestamp
        if key not in seen:
            seen.add(key)
            unique.append(c)
    unique.sort(key=lambda c: c.timestamp)

    logger.info(
        "Backtest: %s %s — %d candles from %s to %s",
        symbol, interval, len(unique),
        unique[0].timestamp.isoformat() if unique else "?",
        unique[-1].timestamp.isoformat() if unique else "?",
    )

    engine = BacktestEngine(unique, initial_capital=initial_capital, max_open_positions=max_open_positions, slippage_pct=slippage_pct)
    return engine.run()
