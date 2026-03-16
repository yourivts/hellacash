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
from bot.indicators.volatility import atr as compute_atr
from bot.risk.position_sizer import fixed_fractional_size
from bot.risk.stop_loss import initial_stops, trail_stop
from bot.risk.fee_gate import check_fee_gate
from bot.strategy.base import Signal
from bot.strategy.confluence import check_confluence
from bot.strategy.router import StrategyRouter, detect_regime, Regime

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fee model constants
# ---------------------------------------------------------------------------
MAKER_FEE_PCT = 0.0015   # 0.15% maker fee (limit order)
TAKER_FEE_PCT = 0.0025   # 0.25% taker fee (market order)

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
    regime: str = "neutral"


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
    # New metrics from engine overhaul
    profit_factor: float = 0.0
    total_fees_paid: float = 0.0
    profit_per_fee: float = 0.0
    signals_generated: int = 0
    signals_filled: int = 0
    fill_rate: float = 0.0
    quiet_hours_skipped: int = 0
    regime_pnl: Dict[str, float] = field(default_factory=dict)


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
    entry_fee: float = 0.0
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
        max_open_positions: int = 10,
        slippage_pct: float = 0.001,
        strategy_params: Optional[Dict[str, Any]] = None,
        target_strategy: Optional[str] = None,
    ) -> None:
        self._raw_candles = candles
        self._strategy_params = strategy_params or {}
        self._target_strategy = target_strategy
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
        self._base_risk_pct = params.get("base_risk_pct", 3.0)
        self._atr_pct_history: List[float] = []  # rolling ATR% for median computation
        self._min_confirmations = params.get("min_confirmations", 2)
        self._indicator_weights = params.get("indicator_weights")
        # No trade cooldown — 1 position per coin, reopen immediately on signal
        self._cooldown_bars = 0
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
        # Range strategy: max hold = 72h = 864 5m bars (hardcoded, not walk-forward)
        self._range_max_hold_bars = 864

        # Tracking counters for new metrics
        self._total_fees_paid: float = 0.0
        self._signals_generated: int = 0
        self._signals_filled: int = 0
        self._quiet_hours_skipped: int = 0
        self._regime_pnl: Dict[str, float] = {}
        self._trade_regime: Dict[int, str] = {}  # pos id -> regime at entry

        self._router = StrategyRouter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # How often (in 5m candles) to run signal generation.
    # Stops are still checked every candle for safety.
    SIGNAL_EVERY = 12  # every 12 x 5m = 1h (matches 1h strategy evaluation)

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

        n = len(df)

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

        # Pre-compute 4h regime arrays (regime detection uses 4h data)
        from bot.indicators.trend import adx as compute_adx
        regime_adx_4h = compute_adx(df_4h["high"], df_4h["low"], df_4h["close"]).values
        regime_atr_4h = compute_atr(df_4h["high"], df_4h["low"], df_4h["close"]).values
        regime_close_4h = df_4h["close"].values
        # Also keep 1h arrays for ATR% (used in position sizing)
        regime_atr_1h = compute_atr(df_1h["high"], df_1h["low"], df_1h["close"]).values
        regime_close_1h = df_1h["close"].values

        h1_timestamps = df_1h.index
        h4_timestamps = df_4h.index
        h1d_timestamps = df_1d.index

        equity_curve: List[float] = []
        last_trade_close_bar = -self._cooldown_bars  # allow first trade immediately
        # Consecutive confirmation state
        _prev_signal_dir: Optional[str] = None
        _signal_streak: int = 0

        for i in range(self.WARMUP, n):
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

            # Find corresponding 1h, 4h, 1d bar indices
            current_ts = timestamps[i]
            h1_idx = h1_timestamps.searchsorted(current_ts, side="right") - 1
            if h1_idx < 30:
                equity_curve.append(current_equity)
                continue
            h4_idx = h4_timestamps.searchsorted(current_ts, side="right") - 1
            h1d_idx = h1d_timestamps.searchsorted(current_ts, side="right") - 1

            # --- 4h regime detection (matches detect_regime() priority order) ---
            atr_val_4h = regime_atr_4h[h4_idx] if 0 <= h4_idx < len(regime_atr_4h) else 0
            price_4h = regime_close_4h[h4_idx] if 0 <= h4_idx < len(regime_close_4h) else current_price
            adx_4h = regime_adx_4h[h4_idx] if 0 <= h4_idx < len(regime_adx_4h) else 20
            atr_pct_4h = (atr_val_4h / price_4h * 100.0) if price_4h > 0 else 0.0

            _quiet_thresh = self._strategy_params.get("quiet_atr_threshold", 1.0)
            _regime_adx = self._strategy_params.get("regime_adx_threshold", 24)
            _ranging_adx = self._strategy_params.get("ranging_adx_threshold", 20)
            if atr_pct_4h < _quiet_thresh:
                regime = Regime.QUIET
            elif atr_pct_4h > 4.0:
                regime = Regime.VOLATILE
            elif adx_4h > _regime_adx:
                regime = Regime.TRENDING
            elif adx_4h < _ranging_adx:
                regime = Regime.RANGING
            else:
                regime = Regime.NEUTRAL

            # Skip entries during QUIET regime
            if regime == Regime.QUIET:
                self._quiet_hours_skipped += 1
                equity_curve.append(current_equity)
                continue

            # Compute 1h ATR% for position sizing
            atr_val_1h = regime_atr_1h[h1_idx] if 0 <= h1_idx < len(regime_atr_1h) else 0
            price_1h = regime_close_1h[h1_idx] if 0 <= h1_idx < len(regime_close_1h) else current_price
            atr_pct = (atr_val_1h / price_1h) * 100.0 if price_1h > 0 else 0.0

            # Gate: skip entries when volatility is too low to cover fees
            if atr_pct < self._min_atr_pct:
                equity_curve.append(current_equity)
                continue

            strategies = self._router.get_strategies(regime)
            if self._target_strategy:
                strategies = [s for s in strategies if s.name == self._target_strategy]
                if not strategies:
                    equity_curve.append(current_equity)
                    continue

            # Look up pre-computed signal scores and run strategy evaluate_1h()
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
                and _signal_streak >= self._consecutive_confirms
                and len(self.positions) < self.max_open
            ):
                self._signals_generated += 1
                direction = best_signal.direction

                # --- Fill rate model: check if limit order would fill ---
                limit_price = current_price
                if i + 1 < n:
                    if direction == "LONG" and lows[i + 1] > limit_price:
                        equity_curve.append(current_equity)
                        continue  # limit not filled
                    elif direction == "SHORT" and highs[i + 1] < limit_price:
                        equity_curve.append(current_equity)
                        continue  # limit not filled

                self._signals_filled += 1

                # Use 1h candles for ATR-based stop calculation (5m ATR is too tight)
                window_1h = df_1h.iloc[max(0, h1_idx - 100): h1_idx + 1]

                # Compute stops early for fee gate check
                sl, tp = initial_stops(
                    current_price, window_1h, direction=direction,
                    atr_multiplier=self._atr_multiplier,
                    rr_ratio=self._rr_ratio,
                    total_fee_pct=(MAKER_FEE_PCT + TAKER_FEE_PCT) * 100.0,
                )

                # --- Fee gate: reject trades where fees eat the profit ---
                tp_distance_pct = abs(tp - current_price) / current_price * 100.0 if current_price > 0 else 0.0
                size_eur_est = self._compute_position_size(1.0, current_price, atr_pct=atr_pct)
                fee_result = check_fee_gate(
                    position_size=size_eur_est,
                    tp_distance_pct=tp_distance_pct,
                    is_short=(direction == "SHORT"),
                    min_profit_multiple=self._strategy_params.get("min_profit_multiple", 3.0),
                )
                if not fee_result.approved:
                    equity_curve.append(current_equity)
                    continue

                # Track rolling ATR% for median computation in position sizer
                if atr_pct > 0:
                    self._atr_pct_history.append(atr_pct)
                    if len(self._atr_pct_history) > 200:
                        self._atr_pct_history = self._atr_pct_history[-200:]
                self._open_position(
                    best_signal, current_price, current_time,
                    window_1h, 1.0, bar_index=i,
                    atr_pct=atr_pct,
                    regime=regime,
                )

            equity_curve.append(current_equity)

        # Close any remaining positions at last price
        if n > 0:
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
        try:
            from bot.indicators.divergence import rsi_divergence, volume_divergence
        except ImportError:
            rsi_divergence = None  # type: ignore[assignment]
            volume_divergence = None  # type: ignore[assignment]
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
            result["vol_profile"] = volume_profile_support(close, volume).values
        except Exception:
            result["vol_profile"] = np.zeros(len(df))

        # OBV for orderflow strategy
        try:
            from bot.indicators.volume import obv
            result["obv"] = obv(close, volume).values
        except Exception:
            result["obv"] = np.zeros(len(df))

        # CMF for orderflow strategy
        try:
            result["cmf"] = cmf(high, low, close, volume).values
        except Exception:
            result["cmf"] = np.zeros(len(df))

        result["close"] = close.values
        result["high"] = high.values
        result["low"] = low.values
        result["open"] = df["open"].values if "open" in df.columns else close.values
        result["volume"] = volume.values

        return result

    def _evaluate_precomputed(
        self, strategies, symbol, current_price, idx_5m, idx_1h,
        precomp_5m, precomp_1h, regime,
        h4_idx=0, h1d_idx=0, precomp_4h=None, precomp_1d=None,
    ) -> Optional[Signal]:
        """Evaluate strategies using their evaluate_1h() methods and confluence gate."""
        collected_signals: List[Dict[str, Any]] = []

        for strat in strategies:
            try:
                direction = "NEUTRAL"
                strength = 0.0

                if strat.name == "orderflow":
                    if idx_1h < 2:
                        continue
                    # Compute candle structure from 1h OHLC
                    o = precomp_1h["open"][idx_1h]
                    h = precomp_1h["high"][idx_1h]
                    l = precomp_1h["low"][idx_1h]
                    c = precomp_1h["close"][idx_1h]
                    candle_range = h - l
                    if candle_range <= 0:
                        continue
                    body_ratio = abs(c - o) / candle_range
                    wick_lower_ratio = (min(o, c) - l) / candle_range
                    wick_upper_ratio = (h - max(o, c)) / candle_range
                    volume_surge = precomp_1h["vsr"][idx_1h] if idx_1h < len(precomp_1h["vsr"]) else 1.0
                    rsi_1h = precomp_1h["rsi"][idx_1h] if idx_1h < len(precomp_1h["rsi"]) else 50.0
                    cmf_val = precomp_1h["cmf"][idx_1h] if idx_1h < len(precomp_1h["cmf"]) else 0.0
                    # OBV divergence: compare current vs 12 bars back
                    obv_now = precomp_1h["obv"][idx_1h] if idx_1h < len(precomp_1h["obv"]) else 0.0
                    obv_prev = precomp_1h["obv"][idx_1h - 12] if idx_1h >= 12 and idx_1h < len(precomp_1h["obv"]) else obv_now
                    price_prev = precomp_1h["close"][idx_1h - 12] if idx_1h >= 12 else current_price
                    if current_price < price_prev and obv_now > obv_prev:
                        obv_div = 1.0  # bullish divergence
                    elif current_price > price_prev and obv_now < obv_prev:
                        obv_div = -1.0  # bearish divergence
                    else:
                        obv_div = 0.0

                    direction, strength = strat.evaluate_1h(
                        body_ratio, wick_lower_ratio, wick_upper_ratio,
                        volume_surge, rsi_1h, cmf_val, obv_div,
                    )

                elif strat.name == "funding_contrarian":
                    if idx_1h < 2:
                        continue
                    rsi_1h = precomp_1h["rsi"][idx_1h] if idx_1h < len(precomp_1h["rsi"]) else 50.0
                    rsi_4h = precomp_4h["rsi"][h4_idx] if precomp_4h and 0 <= h4_idx < len(precomp_4h["rsi"]) else 50.0
                    macd_hist = precomp_1h["macd_hist"][idx_1h] if idx_1h < len(precomp_1h["macd_hist"]) else 0.0
                    macd_hist_prev = precomp_1h["macd_hist"][idx_1h - 1] if idx_1h > 0 and idx_1h < len(precomp_1h["macd_hist"]) else 0.0

                    direction, strength = strat.evaluate_1h(
                        rsi_1h, rsi_4h, macd_hist, macd_hist_prev,
                    )

                elif strat.name == "range":
                    if idx_1h < 2:
                        continue
                    price = current_price
                    bb_lower = precomp_1h["bb_lower"][idx_1h] if idx_1h < len(precomp_1h["bb_lower"]) else price
                    bb_upper = precomp_1h["bb_upper"][idx_1h] if idx_1h < len(precomp_1h["bb_upper"]) else price
                    bb_mid = precomp_1h["bb_mid"][idx_1h] if idx_1h < len(precomp_1h["bb_mid"]) else price
                    bb_bandwidth = precomp_1h["bb_bandwidth"][idx_1h] if idx_1h < len(precomp_1h["bb_bandwidth"]) else 0.0
                    adx_4h_val = precomp_4h["adx"][h4_idx] if precomp_4h and 0 <= h4_idx < len(precomp_4h["adx"]) else 25.0
                    rsi_1h = precomp_1h["rsi"][idx_1h] if idx_1h < len(precomp_1h["rsi"]) else 50.0

                    direction, strength = strat.evaluate_1h(
                        price, bb_lower, bb_upper, bb_mid, bb_bandwidth,
                        adx_4h_val, rsi_1h,
                    )

                elif strat.name == "squeeze":
                    if idx_1h < 2:
                        continue
                    bb_bw = precomp_1h["bb_bandwidth"][idx_1h] if idx_1h < len(precomp_1h["bb_bandwidth"]) else 0.0
                    bb_bw_prev = precomp_1h["bb_bandwidth"][idx_1h - 1] if idx_1h > 0 and idx_1h < len(precomp_1h["bb_bandwidth"]) else 0.0
                    price = current_price
                    bb_upper = precomp_1h["bb_upper"][idx_1h] if idx_1h < len(precomp_1h["bb_upper"]) else price
                    bb_lower = precomp_1h["bb_lower"][idx_1h] if idx_1h < len(precomp_1h["bb_lower"]) else price
                    volume_surge = precomp_1h["vsr"][idx_1h] if idx_1h < len(precomp_1h["vsr"]) else 1.0
                    # EMA50 slope from 4h data
                    if precomp_4h and 0 <= h4_idx < len(precomp_4h["ema50"]):
                        ema50_now = precomp_4h["ema50"][h4_idx]
                        ema50_prev = precomp_4h["ema50"][h4_idx - 1] if h4_idx > 0 else ema50_now
                        ema50_slope = ema50_now - ema50_prev
                    else:
                        ema50_slope = 0.0

                    direction, strength = strat.evaluate_1h(
                        bb_bw_prev, bb_bw, price, bb_upper, bb_lower,
                        volume_surge, ema50_slope,
                    )

                else:
                    continue

                if direction in ("LONG", "SHORT") and strength > 0:
                    collected_signals.append({
                        "direction": direction,
                        "strength": strength,
                        "strategy": strat.name,
                    })

            except Exception:
                continue

        if not collected_signals:
            return None

        # --- Confluence gate ---
        if self._target_strategy:
            # Single strategy isolation — skip confluence
            best = max(collected_signals, key=lambda s: s["strength"])
            return Signal(
                symbol=symbol,
                direction=best["direction"],
                strength=best["strength"],
                strategy_name=best["strategy"],
                technical_score=best["strength"] if best["direction"] == "LONG" else -best["strength"],
                indicator_snapshot={"confirming_count": 1},
            )

        confluence = check_confluence(collected_signals)
        if confluence.triggered:
            # Confluence gives a boost — use best strength from agreeing strategies
            return Signal(
                symbol=symbol,
                direction=confluence.direction,
                strength=min(confluence.strength * 1.2, 1.0),  # 20% boost
                strategy_name="confluence:" + "+".join(confluence.agreeing_strategies),
                technical_score=confluence.strength if confluence.direction == "LONG" else -confluence.strength,
                indicator_snapshot={"confirming_count": len(confluence.agreeing_strategies)},
            )

        # No confluence — return the single best signal
        best = max(collected_signals, key=lambda s: s["strength"])
        return Signal(
            symbol=symbol,
            direction=best["direction"],
            strength=best["strength"],
            strategy_name=best["strategy"],
            technical_score=best["strength"] if best["direction"] == "LONG" else -best["strength"],
            indicator_snapshot={"confirming_count": 1},
        )

    # ------------------------------------------------------------------
    # Position management helpers
    # ------------------------------------------------------------------

    def _compute_position_size(
        self, size_modifier: float, current_price: float, atr_pct: float = 1.5
    ) -> float:
        """Compute position size in EUR with volatility scaling and drawdown scaling.

        Uses fixed fractional sizing: position = (equity * risk_pct) / stop_distance_pct,
        scaled by current ATR relative to its median.
        """
        equity = self._equity(current_price)

        # Stop distance: ATR multiplier * current ATR%
        stop_distance_pct = max(0.1, self._atr_multiplier * atr_pct)

        # Median ATR over recent history (fallback to current if no history)
        if len(self._atr_pct_history) >= 5:
            median_atr_pct = float(sorted(self._atr_pct_history)[len(self._atr_pct_history) // 2])
        else:
            median_atr_pct = max(atr_pct, 0.1)

        raw_size = fixed_fractional_size(
            equity=equity,
            base_risk_pct=self._base_risk_pct,
            stop_distance_pct=stop_distance_pct,
            atr_pct=atr_pct,
            median_atr_pct=median_atr_pct,
        )
        size_eur = raw_size * size_modifier

        # Drawdown scaling (matches live DrawdownGuard.position_size_multiplier)
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
        atr_pct: float = 1.5,
        regime: Regime = Regime.NEUTRAL,
    ) -> None:
        size_eur = self._compute_position_size(size_modifier, current_price=price, atr_pct=atr_pct)
        if size_eur < 10.0 or size_eur > self.balance:
            return  # skip tiny or over-sized trades

        # Deduct cost (including entry fee — maker fee for limit orders)
        symbol = signal.symbol
        entry_fee = size_eur * MAKER_FEE_PCT
        cost = size_eur + entry_fee
        if cost > self.balance:
            return
        self.balance -= cost
        self._total_fees_paid += entry_fee

        # Stops (direction-aware, fee-compensated)
        direction = signal.direction
        total_fee_pct = (MAKER_FEE_PCT + TAKER_FEE_PCT) * 100.0  # entry maker + exit taker
        sl, tp = initial_stops(price, df_window, direction=direction,
                               atr_multiplier=self._atr_multiplier,
                               rr_ratio=self._rr_ratio,
                               total_fee_pct=total_fee_pct)

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
            entry_fee=entry_fee,
        ))
        # Track entry regime for regime P&L breakdown
        self._trade_regime[id(self.positions[-1])] = regime.value if isinstance(regime, Regime) else str(regime)

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
                pos.stop_loss = trail_stop(
                    price, pos.highest_price, pos.stop_loss,
                    trail_dist,
                    direction=direction,
                    activation_threshold=1.5,
                    entry_price=pos.entry_price,
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

        # Look up regime at entry for regime P&L tracking
        entry_regime = self._trade_regime.pop(id(pos), "neutral")

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

        # Maker/taker fee differentiation:
        # TP exits use maker fee (limit orders), all other exits use taker fee
        if reason == "take_profit":
            exit_fee_rate = MAKER_FEE_PCT
        else:
            exit_fee_rate = TAKER_FEE_PCT
        exit_fee = slipped_exit * quantity * exit_fee_rate
        self._total_fees_paid += exit_fee
        pnl = gross_pnl - exit_fee - pos.entry_fee
        pnl_pct = (pnl / pos.size_eur) * 100.0 if pos.size_eur > 0 else 0.0

        # Return capital + P&L
        self.balance += pos.size_eur + pnl

        # Track regime P&L
        self._regime_pnl[entry_regime] = self._regime_pnl.get(entry_regime, 0.0) + pnl

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
            regime=entry_regime,
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

        # Profit factor
        gross_wins = sum(t.pnl_eur for t in wins)
        gross_losses = abs(sum(t.pnl_eur for t in losses))
        profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else 999.0

        # Profit per fee
        profit_per_fee = (total_pnl / self._total_fees_paid) if self._total_fees_paid > 0 else 0.0

        # Fill rate
        fill_rate = (self._signals_filled / self._signals_generated * 100.0) if self._signals_generated > 0 else 0.0

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
            profit_factor=round(profit_factor, 2),
            total_fees_paid=round(self._total_fees_paid, 4),
            profit_per_fee=round(profit_per_fee, 2),
            signals_generated=self._signals_generated,
            signals_filled=self._signals_filled,
            fill_rate=round(fill_rate, 2),
            quiet_hours_skipped=self._quiet_hours_skipped,
            regime_pnl={k: round(v, 2) for k, v in self._regime_pnl.items()},
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
    max_open_positions: int = 10,
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
