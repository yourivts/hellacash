"""Backtesting engine: simulates strategy cycle on historical candles."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from bot.exchange.bitvavo_client import CandleData, TAKER_FEE
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
        taker_fee: float = TAKER_FEE,
        max_open_positions: int = 3,
    ) -> None:
        self._raw_candles = candles
        self.initial_capital = initial_capital
        self.taker_fee = taker_fee
        self.max_open = max_open_positions

        self.balance = initial_capital
        self.peak_balance = initial_capital
        self.positions: List[_OpenPosition] = []
        self.closed_trades: List[TradeRecord] = []

        self._router = StrategyRouter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> BacktestResult:
        """Execute the full backtest and return the result summary."""
        df = self._to_dataframe(self._raw_candles)
        if len(df) < self.WARMUP:
            logger.warning("Not enough candles for backtest (%d < %d)", len(df), self.WARMUP)
            return BacktestResult()

        equity_curve: List[float] = []

        for i in range(self.WARMUP, len(df)):
            candle = df.iloc[i]
            current_price = candle["close"]
            current_time = str(candle.name)

            # --- check stops on existing positions ---
            self._check_exits(current_price, candle, current_time, df.iloc[:i + 1])

            # --- build context & generate signals ---
            window_5m = df.iloc[max(0, i - 200): i + 1].copy()
            window_1h = self._resample_1h(df.iloc[max(0, i - 500): i + 1])

            if len(window_1h) < 30:
                equity_curve.append(self._equity(current_price))
                continue

            regime = detect_regime(window_1h)
            strategies = self._router.get_strategies(regime)
            size_modifier = self._router.position_size_modifier(regime)

            ctx = MarketContext(
                symbol=df.attrs.get("symbol", "BTC-EUR"),
                candles_5m=window_5m,
                candles_1h=window_1h,
                current_price=current_price,
                sentiment_score=0.0,  # no sentiment in backtest
                portfolio_equity_eur=self._equity(current_price),
                open_position_count=len(self.positions),
            )

            best_signal: Optional[Signal] = None
            for strat in strategies:
                try:
                    sig = strat.generate_signal(ctx)
                except Exception:
                    continue
                if sig.direction in ("LONG",) and sig.strength > 0:
                    if best_signal is None or sig.strength > best_signal.strength:
                        best_signal = sig

            # --- entry logic ---
            if (
                best_signal is not None
                and best_signal.is_actionable(min_confidence=0.50, min_confirmations=2)
                and len(self.positions) < self.max_open
            ):
                self._open_position(
                    best_signal, current_price, current_time,
                    window_5m, size_modifier,
                )

            equity_curve.append(self._equity(current_price))

        # Close any remaining positions at last price
        if len(df) > 0:
            last_price = df.iloc[-1]["close"]
            last_time = str(df.index[-1])
            for pos in list(self.positions):
                self._close_position(pos, last_price, last_time, "end_of_data")

        return self._compile_result(equity_curve)

    # ------------------------------------------------------------------
    # Position management helpers
    # ------------------------------------------------------------------

    def _open_position(
        self,
        signal: Signal,
        price: float,
        time_str: str,
        df_window: pd.DataFrame,
        size_modifier: float,
    ) -> None:
        # Position sizing (use conservative defaults when no trade history)
        win_rate, avg_win, avg_loss = self._trade_stats()
        raw_size = kelly_size(
            win_rate=win_rate,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            portfolio_eur=self.balance,
        )
        size_eur = raw_size * size_modifier
        if size_eur < 10.0 or size_eur > self.balance:
            return  # skip tiny or over-sized trades

        # Deduct cost (including fee)
        cost = size_eur * (1 + self.taker_fee)
        if cost > self.balance:
            return
        self.balance -= cost

        # Stops
        sl, tp = initial_stops(price, df_window)

        self.positions.append(_OpenPosition(
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=price,
            entry_time=time_str,
            size_eur=size_eur,
            stop_loss=sl,
            take_profit=tp,
            highest_price=price,
            strategy=signal.strategy_name,
        ))

    def _check_exits(
        self,
        price: float,
        candle: pd.Series,
        time_str: str,
        df_up_to_now: pd.DataFrame,
    ) -> None:
        for pos in list(self.positions):
            # Update trailing high
            if price > pos.highest_price:
                pos.highest_price = price

            # Trailing stop update
            if len(df_up_to_now) >= 15:
                atr_val = compute_atr(
                    df_up_to_now["high"], df_up_to_now["low"], df_up_to_now["close"]
                ).iloc[-1]
                pos.stop_loss = trail_stop(
                    price, pos.highest_price, pos.stop_loss, atr_val,
                )

            # Use candle low for stop-loss check (more realistic)
            check_price_low = candle["low"]
            check_price_high = candle["high"]

            triggered = None
            if check_price_low <= pos.stop_loss:
                triggered = "stop_loss"
                price_used = pos.stop_loss  # assume fill at stop
            elif check_price_high >= pos.take_profit:
                triggered = "take_profit"
                price_used = pos.take_profit
            else:
                continue

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

        # Proceeds after fee
        gross = pos.size_eur * (exit_price / pos.entry_price)
        fee = gross * self.taker_fee
        proceeds = gross - fee
        self.balance += proceeds

        pnl = proceeds - pos.size_eur
        pnl_pct = ((exit_price / pos.entry_price) - 1) * 100.0

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
        pos_value = sum(
            p.size_eur * (current_price / p.entry_price)
            for p in self.positions
        )
        return self.balance + pos_value

    def _trade_stats(self) -> tuple[float, float, float]:
        """Return (win_rate, avg_win_pct, avg_loss_pct) from closed trades."""
        if not self.closed_trades:
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


# ---------------------------------------------------------------------------
# Convenience async function
# ---------------------------------------------------------------------------

async def run_backtest(
    symbol: str = "BTC-EUR",
    interval: str = "5m",
    days: int = 30,
    initial_capital: float = 10_000.0,
    max_open_positions: int = 3,
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

    engine = BacktestEngine(unique, initial_capital=initial_capital, max_open_positions=max_open_positions)
    return engine.run()
