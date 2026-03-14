"""TradingLoop — WebSocket event handlers and strategy execution cycle."""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

import pandas as pd

from bot.config import get_settings
from bot.data_loader import CandleCache
from bot.events.bus import (
    TOPIC_SIGNAL,
    TOPIC_TICKER,
    TOPIC_TRADE_CLOSED,
    TOPIC_TRADE_OPENED,
    get_bus,
)
from bot.indicators.volatility import atr as compute_atr
from bot.risk.fees import get_trading_fees
from bot.risk.position_sizer import kelly_size
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

        # Trade stats cache (instance-level)
        self._trade_stats_cache: Dict[str, Any] = {}
        self._trade_stats_ts: float = 0
        self._last_halt_log: float = 0  # throttle halt log messages

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

                # Get win/loss stats from trade history for position sizing
                win_rate, avg_win, avg_loss = await self._get_trade_stats(best_signal.strategy_name)

                size_eur = kelly_size(
                    win_rate=win_rate,
                    avg_win_pct=avg_win,
                    avg_loss_pct=avg_loss,
                    portfolio_eur=equity,
                    current_drawdown_pct=drawdown.current_drawdown_pct(equity),
                    max_position_pct=settings.max_position_size_pct,
                    kelly_fraction=settings.kelly_fraction,
                )
                # Apply regime modifier
                size_eur *= router.position_size_modifier(regime)
                size_eur *= drawdown.position_size_multiplier(equity)

                entry_price = ctx.current_price
                if entry_price <= 0:
                    continue
                amount_base = size_eur / entry_price

                # Compute expected ROI from stop/TP levels
                stop_loss, take_profit = initial_stops(entry_price, df_5m, direction=direction)
                risk = abs(entry_price - stop_loss) / entry_price * 100
                expected_roi = risk * 3.0  # 3:1 reward/risk

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
                finally:
                    self._pending_orders.discard(symbol)

            except Exception as e:
                logger.error("Strategy cycle error for %s: %s", symbol, e, exc_info=True)
