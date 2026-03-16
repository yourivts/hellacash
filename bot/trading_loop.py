"""TradingLoop — WebSocket event handlers and strategy execution cycle."""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

import pandas as pd

from bot.config import get_settings
from bot.data_loader import CandleCache
from bot.events.bus import (
    TOPIC_PORTFOLIO_UPDATE,
    TOPIC_SIGNAL,
    TOPIC_TICKER,
    TOPIC_TRADE_CLOSED,
    TOPIC_TRADE_OPENED,
    get_bus,
)
from bot.indicators.volatility import atr as compute_atr
from bot.risk.fees import get_trading_fees
from bot.risk.position_sizer import fixed_fractional_size
from bot.risk.stop_loss import check_stop_triggered, initial_stops, trail_stop
from bot.strategy.base import MarketContext

logger = logging.getLogger(__name__)


class TradingLoop:
    """Handles WebSocket events and executes the core strategy cycle."""

    def __init__(
        self,
        candle_cache: CandleCache,
        portfolio: Any,
        risk_engine: Any,
        order_mgr: Any,
        sentiment: Any,
        router: Any,
        drawdown: Any,
        trade_analyzer: Any,
        signal_eval: Any,
        settings: Any,
        onchain: Any = None,
        orderbook: Any = None,
    ) -> None:
        self.candle_cache = candle_cache
        self.portfolio = portfolio
        self.risk_engine = risk_engine
        self.order_mgr = order_mgr
        self.sentiment = sentiment
        self.router = router
        self.drawdown = drawdown
        self.trade_analyzer = trade_analyzer
        self.signal_eval = signal_eval
        self.settings = settings
        self.onchain = onchain
        self.orderbook = orderbook

        self._pending_orders: set[str] = set()

        # Per-symbol cooldown: tracks last trade close time (monotonic)
        self._last_trade_closed: Dict[str, float] = {}
        self._cooldown_seconds: float = settings.trade_cooldown_hours * 3600 if hasattr(settings, "trade_cooldown_hours") else 96 * 3600

        # Trade stats cache (instance-level)
        self._trade_stats_cache: Dict[str, Any] = {}
        self._trade_stats_ts: float = 0
        self._last_halt_log: float = 0  # throttle halt log messages
        self._last_portfolio_publish: float = 0  # throttle portfolio update events

    def is_symbol_pending(self, symbol: str) -> bool:
        """Check if a symbol has a pending order (used by manual close endpoint too)."""
        return symbol in self._pending_orders

    @staticmethod
    def _resample(df_5m: pd.DataFrame, freq: str) -> pd.DataFrame:
        if df_5m.empty:
            return df_5m
        return df_5m.resample(freq).agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()

    # ── WebSocket event handlers ──────────────────────────────────────────────

    async def on_candle(self, data: Dict[str, Any]) -> None:
        self.candle_cache.cache_candle(data["symbol"], data["interval"], data)
        self.portfolio.update_price(data["symbol"], data["close"])

        # Check stop-loss / take-profit on every candle
        if data["interval"] == "1m":
            await self._check_stops(data["symbol"], data["close"])

    async def on_ticker(self, data: Dict[str, Any]) -> None:
        symbol = data.get("symbol", "")
        price = data.get("price", 0.0)
        if symbol and price:
            self.portfolio.update_price(symbol, price)
            await get_bus().publish(TOPIC_TICKER, data)

            # Publish portfolio update (throttled to every 3s)
            now = time.monotonic()
            if now - self._last_portfolio_publish >= 3.0:
                self._last_portfolio_publish = now
                await get_bus().publish(TOPIC_PORTFOLIO_UPDATE, {
                    "equity_eur": self.portfolio.get_equity_eur(),
                    "positions_value_eur": self.portfolio.get_positions_value_eur(),
                })

    async def on_fill(self, data: Dict[str, Any]) -> None:
        await self.order_mgr.on_fill_notification(data)

    # ── Stop-loss monitoring ──────────────────────────────────────────────────

    async def _check_stops(self, symbol: str, current_price: float) -> None:
        pos = self.portfolio.get_position(symbol)
        if pos is None:
            return

        direction = pos.get("direction", "LONG")

        # Update trailing stop
        df_5m = self.candle_cache.get_df(symbol, "5m")
        if len(df_5m) >= 14:
            atr_val = compute_atr(df_5m["high"], df_5m["low"], df_5m["close"]).iloc[-1]
            default_stop = current_price * (1.05 if direction == "SHORT" else 0.95)
            new_stop = trail_stop(
                current_price,
                pos["highest_price"],
                pos["stop_loss_price"] or default_stop,
                atr_val,
                direction=direction,
            )
            if direction == "SHORT":
                if new_stop < (pos["stop_loss_price"] or float("inf")):
                    pos["stop_loss_price"] = new_stop
            else:
                if new_stop > (pos["stop_loss_price"] or 0):
                    pos["stop_loss_price"] = new_stop

        # Range 24h max hold time
        if pos.get("strategy_name") == "range":
            entry_time = pos.get("entry_time")
            if entry_time is not None:
                from datetime import datetime, timezone
                try:
                    if isinstance(entry_time, str):
                        entry_dt = datetime.fromisoformat(entry_time)
                    else:
                        entry_dt = entry_time
                    now = datetime.now(timezone.utc)
                    if (now - entry_dt).total_seconds() >= 86400:  # 24 hours
                        await self._close_position(symbol, current_price, "range_time_exit")
                        return
                except (ValueError, TypeError):
                    pass

        # Range dynamic TP: shift from mid to opposite band
        if pos.get("strategy_name") == "range" and not pos.get("tp_shifted", False):
            crossed_mid = (
                (direction == "LONG" and current_price >= pos.get("range_mid", 0)) or
                (direction == "SHORT" and current_price <= pos.get("range_mid", float("inf")))
            )
            if crossed_mid and pos.get("range_mid", 0) > 0:
                from bot.indicators.momentum import rsi as compute_rsi
                if len(df_5m) >= 20:
                    rsi_series = compute_rsi(df_5m["close"])
                    current_rsi = rsi_series.iloc[-1]
                    prev_rsi = rsi_series.iloc[-4] if len(rsi_series) >= 4 else current_rsi

                    if direction == "LONG" and current_rsi > prev_rsi:
                        pos["tp_shifted"] = True
                        pos["take_profit_price"] = pos["range_upper"]
                        logger.info("Range TP shifted to upper band %.2f for %s", pos["range_upper"], symbol)
                    elif direction == "SHORT" and current_rsi < prev_rsi:
                        pos["tp_shifted"] = True
                        pos["take_profit_price"] = pos["range_lower"]
                        logger.info("Range TP shifted to lower band %.2f for %s", pos["range_lower"], symbol)
                    else:
                        await self._close_position(symbol, pos["range_mid"], "range_mid_exit")
                        return

        # Tight trailing stop for range positions with shifted TP
        if pos.get("strategy_name") == "range" and pos.get("tp_shifted", False):
            if direction == "LONG":
                trail_level = pos["highest_price"] * 0.99
                if trail_level > (pos["stop_loss_price"] or 0):
                    pos["stop_loss_price"] = trail_level
            elif direction == "SHORT":
                trail_level = pos["highest_price"] * 1.01
                if trail_level < (pos["stop_loss_price"] or float("inf")):
                    pos["stop_loss_price"] = trail_level

        default_tp = 0.0 if direction == "SHORT" else float("inf")
        exit_reason = check_stop_triggered(
            current_price,
            pos["stop_loss_price"] or (float("inf") if direction == "SHORT" else 0),
            pos["take_profit_price"] or default_tp,
            direction=direction,
        )

        if exit_reason:
            await self._close_position(symbol, current_price, exit_reason)

    async def _close_position(self, symbol: str, price: float, reason: str) -> None:
        portfolio = self.portfolio
        order_mgr = self.order_mgr
        sentiment = self.sentiment

        pos = portfolio.get_position(symbol)
        if pos is None:
            return

        if symbol in self._pending_orders:
            logger.warning("Skipping close for %s — order already pending", symbol)
            return

        self._pending_orders.add(symbol)
        try:
            quantity = pos["quantity"]
            direction = pos.get("direction", "LONG")

            # LONG: sell to close. SHORT: buy to close.
            if direction == "SHORT":
                order_id = await order_mgr.submit_buy(symbol, quantity, pos["strategy_name"])
            else:
                order_id = await order_mgr.submit_sell(symbol, quantity, pos["strategy_name"])

            sentiment_score = sentiment.get_score(symbol)
            result = await portfolio.close_position(symbol, price, order_id, reason, sentiment_score)

            if result:
                # Record cooldown start for this symbol
                self._last_trade_closed[symbol] = time.monotonic()

                # Increment range bounce counter
                if pos.get("strategy_name") == "range":
                    if hasattr(self.router, '_range'):
                        self.router._range.increment_bounce(symbol)

                # Trigger learning
                analyzer = self.trade_analyzer
                evaluator = self.signal_eval
                await analyzer.process_closed_trade(
                    trade_id=result["trade_id"],
                    symbol=symbol,
                    strategy_name=pos["strategy_name"],
                    net_pnl=result["net_pnl"],
                    entry_features=pos.get("entry_features"),
                )

                if pos.get("entry_features"):
                    breakdown = pos["entry_features"].get("breakdown", {})
                    evaluator.record_trade(breakdown, direction, result["net_pnl"] > 0)

                if result["net_pnl"] < 0:
                    self.drawdown.record_realized_loss(abs(result["net_pnl"]))

                await get_bus().publish(TOPIC_TRADE_CLOSED, {
                    **result,
                    "price": price,
                    "reason": reason,
                })
        finally:
            self._pending_orders.discard(symbol)

    # ── Trade stats for position sizing ───────────────────────────────────────

    async def _get_trade_stats(self, strategy_name: str) -> tuple:
        """Return (win_rate, avg_win_pct, avg_loss_pct) from trade history."""
        import time as _time
        now = _time.time()
        # Refresh cache every 5 minutes
        if now - self._trade_stats_ts > 300:
            self._trade_stats_cache.clear()
            self._trade_stats_ts = now
            try:
                from bot.data.database import get_session as _get_sess
                from bot.data.repositories import get_trades
                async with _get_sess() as session:
                    trades = await get_trades(session, limit=500)
                for t in trades:
                    name = t.strategy_name or "unknown"
                    self._trade_stats_cache.setdefault(name, {"wins": 0, "losses": 0, "win_pcts": [], "loss_pcts": []})
                    if t.net_pnl > 0:
                        self._trade_stats_cache[name]["wins"] += 1
                        if t.roi_pct:
                            self._trade_stats_cache[name]["win_pcts"].append(abs(t.roi_pct) / 100.0)
                    else:
                        self._trade_stats_cache[name]["losses"] += 1
                        if t.roi_pct:
                            self._trade_stats_cache[name]["loss_pcts"].append(abs(t.roi_pct) / 100.0)
            except Exception as e:
                logger.debug("Trade stats fetch failed: %s", e)

        stats = self._trade_stats_cache.get(strategy_name)
        if not stats or (stats["wins"] + stats["losses"]) < 10:
            # Not enough history — use conservative defaults
            return 0.5, 0.02, 0.01

        total = stats["wins"] + stats["losses"]
        win_rate = stats["wins"] / total
        avg_win = sum(stats["win_pcts"]) / len(stats["win_pcts"]) if stats["win_pcts"] else 0.02
        avg_loss = sum(stats["loss_pcts"]) / len(stats["loss_pcts"]) if stats["loss_pcts"] else 0.01
        return win_rate, avg_win, avg_loss

    # ── Main strategy cycle ───────────────────────────────────────────────────

    async def run_cycle(self, tradeable_symbols: List[str], is_running: bool) -> None:
        settings = self.settings
        portfolio = self.portfolio
        risk_engine = self.risk_engine
        order_mgr = self.order_mgr
        sentiment = self.sentiment
        router = self.router
        drawdown = self.drawdown

        if not is_running:
            return

        equity = portfolio.get_equity_eur()
        drawdown.update(equity)

        for symbol in list(tradeable_symbols):
            try:
                df_5m = self.candle_cache.get_df(symbol, "5m")
                df_1h = self.candle_cache.get_df(symbol, "1h")

                if len(df_5m) < 30:
                    continue

                # Skip if position already open
                if portfolio.has_position(symbol):
                    continue

                # Skip if order is pending (prevents race condition)
                if symbol in self._pending_orders:
                    continue

                # Per-symbol cooldown — prevent overtrading after a close
                last_closed = self._last_trade_closed.get(symbol)
                if last_closed is not None:
                    elapsed = time.monotonic() - last_closed
                    if elapsed < self._cooldown_seconds:
                        remaining_h = (self._cooldown_seconds - elapsed) / 3600
                        logger.debug(
                            "Cooldown active for %s — %.1fh remaining",
                            symbol, remaining_h,
                        )
                        continue

                # Check if trading is allowed
                allowed, halt_reason = drawdown.is_trading_allowed(equity)
                if not allowed:
                    import time as _t
                    now = _t.time()
                    if now - self._last_halt_log > 300:  # log once per 5 min
                        logger.warning("Trading halted: %s", halt_reason)
                        self._last_halt_log = now
                    return

                # Detect regime and select strategies
                regime = router._hybrid.__class__.__name__  # placeholder
                from bot.strategy.router import detect_regime
                regime = detect_regime(df_1h)
                strategies = router.get_strategies(regime)

                # Get sentiment
                sentiment_score = sentiment.get_score(symbol)

                # Build market context
                # Resample for MTF
                df_15m = self._resample(df_5m, "15min") if len(df_5m) >= 10 else df_5m
                df_4h = self._resample(df_5m, "4h") if len(df_5m) >= 200 else df_1h
                df_1d = self._resample(df_5m, "1D") if len(df_5m) >= 500 else df_1h

                # Get provider scores
                onchain_score = self.onchain.score(symbol) if self.onchain and self.onchain.is_available() else 0.0
                orderbook_imb = self.orderbook.score(symbol) if self.orderbook and self.orderbook.is_available() else 0.0

                ctx = MarketContext(
                    symbol=symbol,
                    candles_5m=df_5m,
                    candles_1h=df_1h,
                    candles_15m=df_15m,
                    candles_4h=df_4h,
                    candles_1d=df_1d,
                    current_price=df_5m["close"].iloc[-1],
                    sentiment_score=sentiment_score,
                    portfolio_equity_eur=equity,
                    open_position_count=portfolio.open_position_count(),
                    onchain_score=onchain_score,
                    orderbook_imbalance=orderbook_imb,
                    market_regime=regime,
                )

                # Run strategies and pick strongest signal
                best_signal = None
                for strategy in strategies:
                    sig = strategy.generate_signal(ctx)
                    if sig.direction == "NEUTRAL":
                        continue
                    if best_signal is None or sig.strength > best_signal.strength:
                        best_signal = sig

                if best_signal is None:
                    continue

                # Publish every signal to the live feed
                await get_bus().publish(TOPIC_SIGNAL, {
                    "symbol": symbol,
                    "direction": best_signal.direction,
                    "strength": best_signal.strength,
                    "strategy_name": best_signal.strategy_name,
                    "technical_score": best_signal.technical_score,
                    "sentiment_score": best_signal.sentiment_score,
                    "actionable": best_signal.is_actionable(settings.min_signal_confidence),
                })

                if not best_signal.is_actionable(settings.min_signal_confidence):
                    logger.info(
                        "Signal SKIPPED %s %s %s (strength=%.2f < min=%.2f)",
                        best_signal.direction, symbol, best_signal.strategy_name,
                        best_signal.strength, settings.min_signal_confidence,
                    )
                    continue

                direction = best_signal.direction
                if direction not in ("LONG", "SHORT"):
                    continue

                entry_price = ctx.current_price
                if entry_price <= 0:
                    continue

                # Compute stops first — stop distance drives position sizing
                stop_loss, take_profit = initial_stops(entry_price, df_1h, direction=direction)
                stop_distance_pct = max(0.1, abs(entry_price - stop_loss) / entry_price * 100.0)
                risk = stop_distance_pct
                expected_roi = risk * 2.0  # 2:1 reward/risk

                # Compute current ATR% and median ATR% for volatility scaling
                atr_series = compute_atr(df_1h["high"], df_1h["low"], df_1h["close"])
                current_atr = float(atr_series.iloc[-1]) if len(atr_series) > 0 else 0.0
                atr_pct = (current_atr / entry_price * 100.0) if entry_price > 0 else 1.5
                atr_pct = max(atr_pct, 0.1)
                # Median over last 50 bars as normalisation reference
                atr_50 = atr_series.iloc[-50:] if len(atr_series) >= 50 else atr_series
                median_atr = float(atr_50.median()) if len(atr_50) > 0 else current_atr
                median_atr_pct = max((median_atr / entry_price * 100.0) if entry_price > 0 else atr_pct, 0.1)

                # Fixed fractional position size
                size_eur = fixed_fractional_size(
                    equity=equity,
                    base_risk_pct=settings.base_risk_pct,
                    stop_distance_pct=stop_distance_pct,
                    atr_pct=atr_pct,
                    median_atr_pct=median_atr_pct,
                )
                # Apply regime modifier and drawdown scaling
                size_eur *= router.position_size_modifier(regime)
                size_eur *= drawdown.position_size_multiplier(equity)

                amount_base = size_eur / entry_price

                # Risk gate (use actual per-market taker fee)
                _, taker_pct = get_trading_fees(symbol)
                side = "buy" if direction == "LONG" else "sell"
                decision = risk_engine.approve(
                    symbol=symbol,
                    side=side,
                    position_size_eur=size_eur,
                    signal_confidence=best_signal.strength,
                    expected_roi_pct=expected_roi,
                    portfolio_equity_eur=equity,
                    open_position_count=portfolio.open_position_count(),
                    daily_loss_eur=drawdown.daily_realized_loss_eur(),
                    current_drawdown_pct=drawdown.current_drawdown_pct(equity),
                    taker_fee_pct=taker_pct,
                )

                if not decision.approved:
                    # RiskEngine already logs rejections
                    continue

                logger.info(
                    "Signal ACCEPTED %s %s %s — size €%.2f, confidence=%.2f",
                    direction, symbol, best_signal.strategy_name,
                    size_eur, best_signal.strength,
                )

                # Save signal
                from bot.data.database import get_session as _gs
                from bot.data.repositories import save_signal
                async with _gs() as session:
                    sig_record = await save_signal(
                        session,
                        symbol=symbol,
                        strategy_name=best_signal.strategy_name,
                        direction=direction,
                        strength=best_signal.strength,
                        technical_score=best_signal.technical_score,
                        sentiment_score=best_signal.sentiment_score,
                        indicator_snapshot=best_signal.indicator_snapshot,
                        acted_on=True,
                    )
                    signal_id = sig_record.id

                self._pending_orders.add(symbol)
                try:
                    journal_data = {
                        "signal": best_signal,
                        "market_regime": regime,
                        "candle_data": df_5m.tail(100).to_dict("records") if len(df_5m) > 0 else [],
                    }
                    # LONG: buy to open. SHORT: sell to open.
                    if direction == "SHORT":
                        order_id = await order_mgr.submit_sell(
                            symbol, amount_base, best_signal.strategy_name, signal_id,
                            extra_data=journal_data,
                        )
                    else:
                        order_id = await order_mgr.submit_buy(
                            symbol, amount_base, best_signal.strategy_name, signal_id,
                            extra_data=journal_data,
                        )

                    if order_id:
                        await portfolio.add_position(
                            symbol=symbol,
                            strategy_name=best_signal.strategy_name,
                            entry_price=entry_price,
                            quantity=amount_base,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            entry_order_id=order_id,
                            paper_trade=settings.paper_trading,
                            direction=direction,
                        )
                        # Store features on position for learning
                        pos = portfolio.get_position(symbol)
                        if pos:
                            pos["entry_features"] = best_signal.indicator_snapshot

                            # Store range levels for range strategy
                            if best_signal.strategy_name == "range":
                                snap = best_signal.indicator_snapshot
                                pos["range_mid"] = snap.get("range_mid", 0.0)
                                pos["range_upper"] = snap.get("range_upper", 0.0)
                                pos["range_lower"] = snap.get("range_lower", 0.0)
                                pos["tp_shifted"] = False

                                # Override ATR stops with range-specific stops
                                bandwidth_price = pos["range_upper"] - pos["range_lower"]
                                if direction == "LONG":
                                    pos["stop_loss_price"] = pos["range_lower"] - 0.5 * bandwidth_price
                                    pos["take_profit_price"] = pos["range_mid"]
                                else:
                                    pos["stop_loss_price"] = pos["range_upper"] + 0.5 * bandwidth_price
                                    pos["take_profit_price"] = pos["range_mid"]
                finally:
                    self._pending_orders.discard(symbol)

            except Exception as e:
                logger.error("Strategy cycle error for %s: %s", symbol, e, exc_info=True)
