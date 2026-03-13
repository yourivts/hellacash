"""
HellaCash — Main orchestrator.
Starts all async loops and the FastAPI server in a single process.
"""
from __future__ import annotations

import sys as _sys
# Ensure this module is always accessible as 'bot.main' even when run as __main__,
# so API routers that `import bot.main` see the same module-level state.
if __name__ == "__main__" and "bot.main" not in _sys.modules:
    _sys.modules["bot.main"] = _sys.modules[__name__]

import asyncio
import logging
import signal
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import uvicorn

from bot.config import get_settings
from bot.data.database import close_db, init_db
from bot.data_loader import CandleCache
from bot.events.bus import (
    TOPIC_BOT_STARTED,
    TOPIC_SIGNAL,
    TOPIC_TICKER,
    TOPIC_TRADE_CLOSED,
    TOPIC_TRADE_OPENED,
    get_bus,
)
from bot.exchange.bitvavo_client import BitvavoClient
from bot.exchange.bitvavo_ws import BitvavoWebSocket
from bot.exchange.order_manager import OrderManager
from bot.indicators.composite import DEFAULT_WEIGHTS
from bot.indicators.volatility import atr as compute_atr
from bot.learning.param_optimizer import ParamOptimizer
from bot.learning.regime_classifier import RegimeClassifier
from bot.learning.signal_evaluator import SignalEvaluator
from bot.learning.trade_analyzer import TradeAnalyzer
from bot.portfolio.tracker import PortfolioTracker
from bot.risk.drawdown_guard import DrawdownGuard
from bot.risk.engine import RiskEngine
from bot.risk.position_sizer import kelly_size
from bot.risk.stop_loss import check_stop_triggered, initial_stops, trail_stop
from bot.sentiment.aggregator import SentimentAggregator
from bot.strategy.base import MarketContext
from bot.strategy.router import StrategyRouter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Candle cache ──────────────────────────────────────────────────────────────

_candle_cache_obj = CandleCache()


def get_candle_cache() -> CandleCache:
    return _candle_cache_obj


# ── Bot state ─────────────────────────────────────────────────────────────────

_running = False
_active_symbols: List[str] = []
_tradeable_symbols: List[str] = []  # subset with enough volume for trading


async def _init_symbols() -> None:
    """Load market symbols without starting trading."""
    global _active_symbols, _tradeable_symbols
    if _active_symbols:
        return
    settings = get_settings()
    client = _get_client()
    markets = client.get_markets()
    _active_symbols = [m.symbol for m in markets[: settings.max_pairs]]
    _tradeable_symbols = [
        m.symbol for m in markets[: settings.max_pairs]
        if m.volume_24h >= settings.min_volume_eur
    ]
    logger.info("Active symbols: %d total, %d tradeable (vol >= €%.0f)",
                len(_active_symbols), len(_tradeable_symbols), settings.min_volume_eur)


async def start_bot() -> None:
    global _running
    if _running:
        return
    await _init_symbols()
    _running = True
    logger.info("Bot ACTIVATED — trading enabled")

    bus = get_bus()
    await bus.publish(TOPIC_BOT_STARTED, {"symbols": _active_symbols})


async def stop_bot() -> None:
    global _running
    _running = False
    logger.info("Bot stopped")


def is_running() -> bool:
    return _running


def get_active_symbols() -> List[str]:
    return list(_active_symbols)


# ── Global singletons (lazy init) ─────────────────────────────────────────────

_client: Optional[BitvavoClient] = None
_portfolio: Optional[PortfolioTracker] = None
_order_mgr: Optional[OrderManager] = None
_risk_engine: Optional[RiskEngine] = None
_drawdown: Optional[DrawdownGuard] = None
_sentiment: Optional[SentimentAggregator] = None
_router: Optional[StrategyRouter] = None
_trade_analyzer: Optional[TradeAnalyzer] = None
_signal_eval: Optional[SignalEvaluator] = None
_param_optimizer: Optional[ParamOptimizer] = None


def _get_client() -> BitvavoClient:
    global _client
    if _client is None:
        s = get_settings()
        _client = BitvavoClient(s.bitvavo_api_key, s.bitvavo_api_secret, s.paper_trading)
    return _client


def _get_portfolio() -> PortfolioTracker:
    global _portfolio
    if _portfolio is None:
        _portfolio = PortfolioTracker(_get_client())
    return _portfolio


def _get_order_mgr() -> OrderManager:
    global _order_mgr
    if _order_mgr is None:
        _order_mgr = OrderManager(_get_client())
    return _order_mgr


def _get_risk_engine() -> RiskEngine:
    global _risk_engine
    if _risk_engine is None:
        _risk_engine = RiskEngine(get_settings())
    return _risk_engine


def _get_drawdown() -> DrawdownGuard:
    global _drawdown
    if _drawdown is None:
        s = get_settings()
        _drawdown = DrawdownGuard(s.max_drawdown_pct, daily_loss_limit_eur=s.max_daily_loss_eur)
    return _drawdown


def _get_sentiment() -> SentimentAggregator:
    global _sentiment
    if _sentiment is None:
        _sentiment = SentimentAggregator(get_settings())
    return _sentiment


def _get_router() -> StrategyRouter:
    global _router
    if _router is None:
        _router = StrategyRouter()
    return _router


def _get_trade_analyzer() -> TradeAnalyzer:
    global _trade_analyzer
    if _trade_analyzer is None:
        _trade_analyzer = TradeAnalyzer()
    return _trade_analyzer


def _get_signal_eval() -> SignalEvaluator:
    global _signal_eval
    if _signal_eval is None:
        _signal_eval = SignalEvaluator()
    return _signal_eval


def _get_param_optimizer() -> ParamOptimizer:
    global _param_optimizer
    if _param_optimizer is None:
        _param_optimizer = ParamOptimizer(get_settings().model_dir)
    return _param_optimizer


# ── Candle handler (from WebSocket) ───────────────────────────────────────────


async def _on_candle(data: Dict[str, Any]) -> None:
    _candle_cache_obj.cache_candle(data["symbol"], data["interval"], data)
    _get_portfolio().update_price(data["symbol"], data["close"])

    # Check stop-loss / take-profit on every candle
    if data["interval"] == "1m":
        await _check_stops(data["symbol"], data["close"])


async def _on_ticker(data: Dict[str, Any]) -> None:
    symbol = data.get("symbol", "")
    price = data.get("price", 0.0)
    if symbol and price:
        _get_portfolio().update_price(symbol, price)
        await get_bus().publish(TOPIC_TICKER, data)


async def _on_fill(data: Dict[str, Any]) -> None:
    await _get_order_mgr().on_fill_notification(data)


# ── Stop-loss monitoring ──────────────────────────────────────────────────────


async def _check_stops(symbol: str, current_price: float) -> None:
    pos = _get_portfolio().get_position(symbol)
    if pos is None:
        return

    # Update trailing stop
    df_5m = _candle_cache_obj.get_df(symbol, "5m")
    if len(df_5m) >= 14:
        atr_val = compute_atr(df_5m["high"], df_5m["low"], df_5m["close"]).iloc[-1]
        new_stop = trail_stop(
            current_price,
            pos["highest_price"],
            pos["stop_loss_price"] or (current_price * 0.95),
            atr_val,
        )
        if new_stop > (pos["stop_loss_price"] or 0):
            pos["stop_loss_price"] = new_stop

    exit_reason = check_stop_triggered(
        current_price,
        pos["stop_loss_price"] or 0,
        pos["take_profit_price"] or float("inf"),
    )

    if exit_reason:
        await _close_position(symbol, current_price, exit_reason)


async def _close_position(symbol: str, price: float, reason: str) -> None:
    portfolio = _get_portfolio()
    order_mgr = _get_order_mgr()
    sentiment = _get_sentiment()

    pos = portfolio.get_position(symbol)
    if pos is None:
        return

    quantity = pos["quantity"]
    order_id = await order_mgr.submit_sell(symbol, quantity, pos["strategy_name"])

    sentiment_score = sentiment.get_score(symbol)
    result = await portfolio.close_position(symbol, price, order_id, reason, sentiment_score)

    if result:
        # Trigger learning
        analyzer = _get_trade_analyzer()
        evaluator = _get_signal_eval()
        await analyzer.process_closed_trade(
            trade_id=result["trade_id"],
            symbol=symbol,
            strategy_name=pos["strategy_name"],
            net_pnl=result["net_pnl"],
            entry_features=pos.get("entry_features"),
        )

        if pos.get("entry_features"):
            breakdown = pos["entry_features"].get("breakdown", {})
            direction = "LONG"  # we only go long currently
            evaluator.record_trade(breakdown, direction, result["net_pnl"] > 0)

        if result["net_pnl"] < 0:
            _get_drawdown().record_realized_loss(abs(result["net_pnl"]))

        await get_bus().publish(TOPIC_TRADE_CLOSED, {
            **result,
            "price": price,
            "reason": reason,
        })


# ── Trade stats for position sizing ───────────────────────────────────────────

_trade_stats_cache: Dict[str, Any] = {}
_trade_stats_ts: float = 0


async def _get_trade_stats(strategy_name: str) -> tuple:
    """Return (win_rate, avg_win_pct, avg_loss_pct) from trade history."""
    global _trade_stats_ts
    import time as _time
    now = _time.time()
    # Refresh cache every 5 minutes
    if now - _trade_stats_ts > 300:
        _trade_stats_cache.clear()
        _trade_stats_ts = now
        try:
            from bot.data.database import get_session as _get_sess
            from bot.data.repositories import get_trades
            async with _get_sess() as session:
                trades = await get_trades(session, limit=500)
            for t in trades:
                name = t.strategy_name or "unknown"
                _trade_stats_cache.setdefault(name, {"wins": 0, "losses": 0, "win_pcts": [], "loss_pcts": []})
                if t.net_pnl > 0:
                    _trade_stats_cache[name]["wins"] += 1
                    if t.roi_pct:
                        _trade_stats_cache[name]["win_pcts"].append(abs(t.roi_pct) / 100.0)
                else:
                    _trade_stats_cache[name]["losses"] += 1
                    if t.roi_pct:
                        _trade_stats_cache[name]["loss_pcts"].append(abs(t.roi_pct) / 100.0)
        except Exception as e:
            logger.debug("Trade stats fetch failed: %s", e)

    stats = _trade_stats_cache.get(strategy_name)
    if not stats or (stats["wins"] + stats["losses"]) < 10:
        # Not enough history — use conservative defaults
        return 0.5, 0.02, 0.01

    total = stats["wins"] + stats["losses"]
    win_rate = stats["wins"] / total
    avg_win = sum(stats["win_pcts"]) / len(stats["win_pcts"]) if stats["win_pcts"] else 0.02
    avg_loss = sum(stats["loss_pcts"]) / len(stats["loss_pcts"]) if stats["loss_pcts"] else 0.01
    return win_rate, avg_win, avg_loss


# ── Main strategy cycle ───────────────────────────────────────────────────────


async def _strategy_cycle() -> None:
    settings = get_settings()
    portfolio = _get_portfolio()
    risk_engine = _get_risk_engine()
    order_mgr = _get_order_mgr()
    sentiment = _get_sentiment()
    router = _get_router()
    drawdown = _get_drawdown()

    if not _running:
        return

    equity = portfolio.get_equity_eur()
    drawdown.update(equity)

    for symbol in list(_tradeable_symbols):
        try:
            df_5m = _candle_cache_obj.get_df(symbol, "5m")
            df_1h = _candle_cache_obj.get_df(symbol, "1h")

            if len(df_5m) < 30:
                continue

            # Skip if position already open
            if portfolio.has_position(symbol):
                continue

            # Check if trading is allowed
            allowed, halt_reason = drawdown.is_trading_allowed(equity)
            if not allowed:
                logger.warning("Trading halted: %s", halt_reason)
                return

            # Detect regime and select strategies
            regime = router._hybrid.__class__.__name__  # placeholder
            from bot.strategy.router import detect_regime
            regime = detect_regime(df_1h)
            strategies = router.get_strategies(regime)

            # Get sentiment
            sentiment_score = sentiment.get_score(symbol)

            # Build market context
            ctx = MarketContext(
                symbol=symbol,
                candles_5m=df_5m,
                candles_1h=df_1h,
                current_price=df_5m["close"].iloc[-1],
                sentiment_score=sentiment_score,
                portfolio_equity_eur=equity,
                open_position_count=portfolio.open_position_count(),
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
                continue

            if best_signal.direction != "LONG":
                continue  # only long positions (no shorting spot market)

            # Multi-timeframe confirmation: 1h trend must agree
            if len(df_1h) >= 20:
                ema20_1h = df_1h["close"].ewm(span=20).mean().iloc[-1]
                if best_signal.direction == "LONG" and df_1h["close"].iloc[-1] < ema20_1h:
                    continue  # 1h trend is bearish, skip long

            # Get win/loss stats from trade history for position sizing
            win_rate, avg_win, avg_loss = await _get_trade_stats(best_signal.strategy_name)

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
            stop_loss, take_profit = initial_stops(entry_price, df_5m)
            risk = (entry_price - stop_loss) / entry_price * 100
            expected_roi = risk * 3.0  # 3:1 reward/risk

            # Risk gate
            decision = risk_engine.approve(
                symbol=symbol,
                side="buy",
                position_size_eur=size_eur,
                signal_confidence=best_signal.strength,
                expected_roi_pct=expected_roi,
                portfolio_equity_eur=equity,
                open_position_count=portfolio.open_position_count(),
                daily_loss_eur=drawdown.daily_realized_loss_eur(),
                current_drawdown_pct=drawdown.current_drawdown_pct(equity),
            )

            if not decision.approved:
                continue

            # Execute buy
            from bot.data.database import get_session as _gs
            from bot.data.repositories import save_signal
            async with _gs() as session:
                sig_record = await save_signal(
                    session,
                    symbol=symbol,
                    strategy_name=best_signal.strategy_name,
                    direction=best_signal.direction,
                    strength=best_signal.strength,
                    technical_score=best_signal.technical_score,
                    sentiment_score=best_signal.sentiment_score,
                    indicator_snapshot=best_signal.indicator_snapshot,
                    acted_on=True,
                )
                signal_id = sig_record.id

            order_id = await order_mgr.submit_buy(
                symbol, amount_base, best_signal.strategy_name, signal_id
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
                )
                # Store features on position for learning
                pos = portfolio.get_position(symbol)
                if pos:
                    pos["entry_features"] = best_signal.indicator_snapshot

        except Exception as e:
            logger.error("Strategy cycle error for %s: %s", symbol, e, exc_info=True)


# ── Scheduled jobs ────────────────────────────────────────────────────────────


async def _sentiment_loop(interval_secs: int) -> None:
    sentiment = _get_sentiment()
    while True:
        if _active_symbols:  # always collect sentiment, even before trading activated
            try:
                await sentiment.run_cycle(_active_symbols)
            except Exception as e:
                logger.error("Sentiment loop error: %s", e)
        await asyncio.sleep(interval_secs)


async def _strategy_loop(interval_secs: int) -> None:
    while True:
        if _running:
            await _strategy_cycle()
        await asyncio.sleep(interval_secs)


async def _portfolio_snapshot_loop(interval_secs: int = 300) -> None:
    while True:
        if _running:
            try:
                await _get_portfolio().snapshot()
            except Exception as e:
                logger.error("Snapshot error: %s", e)
        await asyncio.sleep(interval_secs)


async def _optimizer_loop(interval_hours: int = 24) -> None:
    while True:
        await asyncio.sleep(interval_hours * 3600)
        if _running:
            try:
                await _get_param_optimizer().optimize_hybrid(
                    min_trades=get_settings().optimizer_min_trades
                )
            except Exception as e:
                logger.error("Optimizer error: %s", e)


# ── Main entry point ──────────────────────────────────────────────────────────


async def _main() -> None:
    settings = get_settings()

    logger.info(
        "Starting HellaCash (paper_trading=%s, max_pairs=%d)",
        settings.paper_trading, settings.max_pairs,
    )

    # Init DB
    await init_db()

    # Pre-load symbols (but don't activate trading)
    await _init_symbols()

    logger.info("Bot ready — click Start on dashboard to begin trading")

    # Initialize portfolio
    portfolio = _get_portfolio()
    await portfolio.refresh()

    # Start WebSocket
    ws = BitvavoWebSocket(
        api_key=settings.bitvavo_api_key,
        api_secret=settings.bitvavo_api_secret,
        on_candle=_on_candle,
        on_ticker=_on_ticker,
        on_fill=_on_fill,
    )
    ws.set_markets(_active_symbols)

    # Import the FastAPI app
    from api.app import create_app
    api_app = create_app()

    # Build uvicorn server config
    uvicorn_config = uvicorn.Config(
        app=api_app,
        host=settings.api_host,
        port=settings.api_port,
        log_level="warning",
    )
    server = uvicorn.Server(uvicorn_config)

    # Run everything concurrently
    logger.info("Dashboard: http://%s:%d", settings.api_host, settings.api_port)

    await asyncio.gather(
        _candle_cache_obj.load_all(
            _active_symbols, _get_client(),
            batch_size=settings.candle_batch_size,
        ),
        ws.run(),
        _sentiment_loop(settings.sentiment_poll_interval_secs),
        _strategy_loop(settings.strategy_cycle_secs),
        _portfolio_snapshot_loop(300),
        _optimizer_loop(24),
        server.serve(),
    )


def main() -> None:
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        logger.info("Shutdown requested")


if __name__ == "__main__":
    main()
