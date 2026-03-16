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
import os
import signal
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional

import uvicorn

from bot.config import get_settings
from bot.data.database import close_db, init_db
from bot.data_loader import CandleCache
from bot.events.bus import (
    TOPIC_BOT_STARTED,
    get_bus,
)
from bot.exchange.bitvavo_client import BitvavoClient
from bot.exchange.bitvavo_ws import BitvavoWebSocket
from bot.exchange.order_manager import OrderManager
from bot.indicators.onchain import OnchainProvider
from bot.indicators.orderbook import OrderBookProvider
from bot.learning.param_optimizer import ParamOptimizer
from bot.learning.regime_classifier import RegimeClassifier
from bot.learning.signal_evaluator import SignalEvaluator
from bot.learning.trade_analyzer import TradeAnalyzer
from bot.learning.walk_forward import WalkForwardOptimizer, TRAIN_DAYS, TEST_DAYS, MIN_WINDOWS
from bot.execution.limit_order_manager import LimitOrderManager
from bot.portfolio.tracker import PortfolioTracker
from bot.risk.drawdown_guard import DrawdownGuard
from bot.risk.engine import RiskEngine
from bot.scheduler import BotScheduler
from bot.sentiment.aggregator import SentimentAggregator
from bot.strategy.router import StrategyRouter
from bot.notifications.discord import DiscordNotifier
from bot.trading_loop import TradingLoop

_LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
_LOG_FILE = os.path.join(_LOG_DIR, "bot.log")
os.makedirs(_LOG_DIR, exist_ok=True)

_log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=_log_format,
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler(
            _LOG_FILE, mode="a", encoding="utf-8",
            maxBytes=10_000_000, backupCount=5,
        ),
    ],
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
_ws_instance: Optional[BitvavoWebSocket] = None


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


def get_tradeable_symbols() -> List[str]:
    return list(_tradeable_symbols)


def get_ws() -> Optional[BitvavoWebSocket]:
    return _ws_instance


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
_onchain_provider: Optional[OnchainProvider] = None
_orderbook_provider: Optional[OrderBookProvider] = None
_walk_forward: Optional[WalkForwardOptimizer] = None
_trading_loop: Optional[TradingLoop] = None
_scheduler: Optional[BotScheduler] = None
_limit_mgr: Optional[LimitOrderManager] = None


def _get_limit_mgr() -> LimitOrderManager:
    global _limit_mgr
    if _limit_mgr is None:
        _limit_mgr = LimitOrderManager()
    return _limit_mgr


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
        _drawdown = DrawdownGuard(s.max_drawdown_pct, daily_loss_limit_eur=s.max_daily_loss_eur, paper_mode=s.paper_trading)
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


def _get_onchain() -> OnchainProvider:
    global _onchain_provider
    if _onchain_provider is None:
        _onchain_provider = OnchainProvider()
    return _onchain_provider


def _get_orderbook() -> OrderBookProvider:
    global _orderbook_provider
    if _orderbook_provider is None:
        settings = get_settings()
        _orderbook_provider = OrderBookProvider(depth_levels=settings.orderbook_depth_levels)
    return _orderbook_provider


def _get_walk_forward() -> WalkForwardOptimizer:
    global _walk_forward
    if _walk_forward is None:
        _walk_forward = WalkForwardOptimizer()
    return _walk_forward


def _get_trading_loop() -> TradingLoop:
    global _trading_loop
    if _trading_loop is None:
        _trading_loop = TradingLoop(
            candle_cache=_candle_cache_obj,
            portfolio=_get_portfolio(),
            risk_engine=_get_risk_engine(),
            order_mgr=_get_order_mgr(),
            sentiment=_get_sentiment(),
            router=_get_router(),
            drawdown=_get_drawdown(),
            trade_analyzer=_get_trade_analyzer(),
            signal_eval=_get_signal_eval(),
            settings=get_settings(),
            onchain=_get_onchain() if get_settings().onchain_enabled else None,
            orderbook=_get_orderbook() if get_settings().orderbook_enabled else None,
        )
        _trading_loop._limit_mgr = _get_limit_mgr()
    return _trading_loop


# ── Walk-forward post-processing ──────────────────────────────────────────────


async def _process_walk_forward_results(
    results, get_router_fn, trading_loop, discord, settings,
):
    """Process walk-forward results: adopt params, log summary, auto-start paper trading."""
    adopted_count = 0
    for symbol, result in results.items():
        if result.adopted and result.recommended_params:
            adopted_count += 1
            logger.info(
                "Walk-forward %s ADOPTED params (avg_sharpe=%.2f, avg_pnl=€%.2f): %s",
                symbol, result.avg_oos_sharpe, result.avg_oos_pnl, result.recommended_params,
            )
        else:
            logger.info(
                "Walk-forward %s — params not adopted (avg_sharpe=%.2f, avg_pnl=€%.2f)",
                symbol, result.avg_oos_sharpe, result.avg_oos_pnl,
            )
    # Apply best adopted result to router (use the one with highest avg P&L)
    best_adopted = None
    best_pnl = float("-inf")
    for symbol, result in results.items():
        if result.adopted and result.recommended_params and result.avg_oos_pnl > best_pnl:
            best_pnl = result.avg_oos_pnl
            best_adopted = result
    if best_adopted and best_adopted.recommended_params:
        router = get_router_fn()
        params = best_adopted.recommended_params
        router.update_params(**params)
        # Apply cooldown to live trading loop if present
        if "cooldown_hours" in params and trading_loop is not None:
            trading_loop._cooldown_seconds = params["cooldown_hours"] * 3600
            logger.info("Walk-forward: updated live cooldown to %dh", params["cooldown_hours"])
        logger.info("Walk-forward: applied best adopted params to router")
    # ── Final analysis summary ──
    wf_symbols = list(results.keys())
    logger.info("=" * 70)
    logger.info("WALK-FORWARD ANALYSIS COMPLETE — %d symbols evaluated", len(wf_symbols))
    logger.info("=" * 70)
    total_avg_pnl = 0.0
    total_avg_sharpe = 0.0
    best_symbol = None
    best_symbol_pnl = float("-inf")
    worst_symbol = None
    worst_symbol_pnl = float("inf")
    for symbol, result in results.items():
        total_avg_pnl += result.avg_oos_pnl
        total_avg_sharpe += result.avg_oos_sharpe
        status = "ADOPTED" if result.adopted else "NOT ADOPTED"
        n_windows = len(result.windows)
        profitable = sum(1 for w in result.windows if w.pnl > 0)
        logger.info(
            "  %-10s | %s | avg_pnl=€%+.2f | avg_sharpe=%+.2f | windows=%d/%d profitable",
            symbol, status, result.avg_oos_pnl, result.avg_oos_sharpe, profitable, n_windows,
        )
        if result.avg_oos_pnl > best_symbol_pnl:
            best_symbol_pnl = result.avg_oos_pnl
            best_symbol = symbol
        if result.avg_oos_pnl < worst_symbol_pnl:
            worst_symbol_pnl = result.avg_oos_pnl
            worst_symbol = symbol
    n = len(results) or 1
    logger.info("-" * 70)
    logger.info("  Portfolio avg P&L:    €%+.2f across %d symbols", total_avg_pnl, len(results))
    logger.info("  Portfolio avg Sharpe: %+.2f", total_avg_sharpe / n)
    logger.info("  Best symbol:          %s (€%+.2f avg OOS P&L)", best_symbol, best_symbol_pnl)
    logger.info("  Worst symbol:         %s (€%+.2f avg OOS P&L)", worst_symbol, worst_symbol_pnl)
    logger.info("  Adopted:              %d/%d symbols", adopted_count, len(wf_symbols))
    logger.info("=" * 70)
    # Send report to Discord
    try:
        await discord.send_walk_forward_report(results)
    except Exception as e:
        logger.warning("Failed to send walk-forward report to Discord: %s", e)

    # Auto-enable trading in paper mode after walk-forward completes
    if settings.paper_trading and not is_running():
        logger.info("Walk-forward complete — auto-enabling paper trading")
        await start_bot()


# ── Main entry point ──────────────────────────────────────────────────────────


async def _shutdown_cleanup(discord) -> None:
    """Clean up resources during shutdown."""
    await discord.shutdown()
    await close_db()


async def _main() -> None:
    global _ws_instance, _scheduler
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

    # Create TradingLoop
    trading_loop = _get_trading_loop()

    # Start WebSocket
    async def _on_book(msg):
        symbol = msg.get("market")
        bids = msg.get("bids", [])
        asks = msg.get("asks", [])
        ob = _get_orderbook()
        if "nonce" in msg and not ob._books.get(symbol):
            ob.set_snapshot(symbol, bids, asks)
        else:
            for price, qty in bids:
                ob.update_level(symbol, "bid", float(price), float(qty))
            for price, qty in asks:
                ob.update_level(symbol, "ask", float(price), float(qty))

    ws = BitvavoWebSocket(
        api_key=settings.bitvavo_api_key,
        api_secret=settings.bitvavo_api_secret,
        on_candle=trading_loop.on_candle,
        on_ticker=trading_loop.on_ticker,
        on_fill=trading_loop.on_fill,
        on_book=_on_book if settings.orderbook_enabled else None,
    )
    ws.set_markets(_active_symbols)
    _ws_instance = ws

    # Create Discord notifier
    discord = DiscordNotifier(settings)

    import glob as _glob
    import time as _time

    async def _cleanup_old_charts():
        charts_dir = "data/charts"
        if not os.path.exists(charts_dir):
            return
        cutoff = _time.time() - (90 * 86400)  # 90 days
        for f in _glob.glob(os.path.join(charts_dir, "*.png")):
            if os.path.getmtime(f) < cutoff:
                os.remove(f)
                logger.info("Removed old chart: %s", f)

    # Create BotScheduler
    sentiment = _get_sentiment()
    param_optimizer = _get_param_optimizer()
    walk_forward = _get_walk_forward()

    # Walk-forward candle fetcher factory: returns a fetcher for a given symbol
    def _wf_candle_fetcher_factory(symbol: str):
        async def _fetcher(start_day: int, end_day: int):
            from datetime import datetime, timedelta, timezone
            now = datetime.now(timezone.utc)
            total_days = TRAIN_DAYS + TEST_DAYS * MIN_WINDOWS
            start_dt = now - timedelta(days=total_days - start_day)
            end_dt = now - timedelta(days=total_days - end_day)
            logger.info("Walk-forward: fetching %s candles day %d-%d (%s to %s)",
                         symbol, start_day, end_day, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
            loop = asyncio.get_running_loop()
            client = _get_client()
            candles = await loop.run_in_executor(
                None, lambda: client.get_candles_range(symbol, "5m", start_dt, end_dt)
            )
            logger.info("Walk-forward: got %d candles for %s day %d-%d", len(candles), symbol, start_day, end_day)
            return candles
        return _fetcher

    async def _run_walk_forward():
        symbols = get_tradeable_symbols() or ["BTC-EUR"]
        wf_symbols = symbols
        logger.info("Walk-forward: running for %d symbols: %s", len(wf_symbols), wf_symbols)
        results = await walk_forward.run_multi(_wf_candle_fetcher_factory, wf_symbols)
        await _process_walk_forward_results(
            results, _get_router, trading_loop, discord, settings,
        )

    # Market data snapshot callback: saves funding rate + orderbook data for future backtesting
    async def _save_market_data_snapshots():
        onchain_prov = _get_onchain()
        orderbook_prov = _get_orderbook()
        if onchain_prov and hasattr(onchain_prov, 'save_snapshots'):
            await onchain_prov.save_snapshots()
        if orderbook_prov and hasattr(orderbook_prov, 'save_snapshots'):
            await orderbook_prov.save_snapshots()

    scheduler = BotScheduler(
        run_cycle=lambda: trading_loop.run_cycle(_tradeable_symbols, _running),
        sentiment_cycle=sentiment.run_cycle,
        portfolio_snapshot=portfolio.snapshot,
        optimizer_run=lambda: param_optimizer.optimize_hybrid(
            min_trades=settings.optimizer_min_trades
        ),
        is_running=is_running,
        active_symbols=get_active_symbols,
        settings=settings,
        discord_run=discord.run,
        walk_forward_run=_run_walk_forward,
        market_data_snapshot=_save_market_data_snapshots,
    )
    _scheduler = scheduler

    # Import the FastAPI app
    from api.app import create_app
    api_app = create_app()

    from api.routers.analytics import router as analytics_router
    from api.routers.journal import router as journal_router
    from api.routers.orderbook import router as orderbook_router, set_provider as set_ob_provider

    api_app.include_router(analytics_router)
    api_app.include_router(journal_router)
    api_app.include_router(orderbook_router)

    if settings.orderbook_enabled:
        set_ob_provider(_get_orderbook())

    # Journal event subscriber
    from bot.analytics.journal import generate_entry_reasoning
    from bot.charts.snapshot import generate_entry_chart
    from bot.events.bus import TOPIC_TRADE_OPENED as _TOPIC_TRADE_OPENED

    async def _journal_on_trade_opened(queue):
        while True:
            event = await queue.get()
            try:
                data = event.payload
                sig = data.get("signal")
                if not sig:
                    continue
                snap = sig.indicator_snapshot or {}
                from bot.data.database import get_session as _get_session
                from bot.data.repositories import save_journal_entry
                async with _get_session() as session:
                    reasoning = generate_entry_reasoning(
                        symbol=sig.symbol, direction=sig.direction,
                        strategy=sig.strategy_name,
                        regime=data.get("market_regime", "unknown"),
                        final_score=snap.get("final_score", 0),
                        threshold=0.35,
                        mtf_scores=snap.get("mtf_scores", {}),
                        mtf_agreement=snap.get("mtf_agreement", 0),
                        technical_score=sig.technical_score,
                        sentiment_score=sig.sentiment_score,
                        onchain_score=snap.get("onchain_score", 0),
                        orderbook_imbalance=snap.get("orderbook_imbalance", 0),
                        indicator_values=snap,
                    )
                    await save_journal_entry(
                        session,
                        entry_order_id=data.get("order_id"),
                        symbol=data.get("symbol"),
                        direction=sig.direction,
                        strategy_name=sig.strategy_name,
                        market_regime=data.get("market_regime", "unknown"),
                        entry_technical_scores=snap.get("mtf_scores"),
                        entry_mtf_scores=snap.get("mtf_scores"),
                        entry_sentiment_score=sig.sentiment_score,
                        entry_onchain_score=snap.get("onchain_score"),
                        entry_orderbook_imbalance=snap.get("orderbook_imbalance"),
                        entry_composite_score=snap.get("final_score", 0),
                        entry_reasoning=reasoning,
                    )

                # Chart in background thread
                candle_data = data.get("candle_data", [])
                if candle_data:
                    import pandas as pd
                    df = pd.DataFrame(candle_data)
                    if not df.empty and "timestamp" in df.columns:
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                        df = df.set_index("timestamp").sort_index()
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(
                            None, generate_entry_chart,
                            df, data.get("fill_price", 0), data.get("symbol", ""),
                            data.get("order_id", 0),
                        )
            except Exception as e:
                logger.error("Journal subscriber error: %s", e, exc_info=True)

    _journal_queue = await get_bus().subscribe(_TOPIC_TRADE_OPENED)
    asyncio.create_task(_journal_on_trade_opened(_journal_queue))

    # Expose bot singletons on app.state for API route access
    api_app.state.trading_loop = trading_loop
    api_app.state.portfolio = portfolio
    api_app.state.order_mgr = _get_order_mgr()
    api_app.state.sentiment = sentiment
    api_app.state.client = _get_client()

    # Build uvicorn server config
    uvicorn_config = uvicorn.Config(
        app=api_app,
        host=settings.api_host,
        port=settings.api_port,
        log_level="warning",
    )
    server = uvicorn.Server(uvicorn_config)

    # Graceful shutdown signal handler
    def _handle_shutdown(sig: int, frame: Any) -> None:
        logger.info("Received signal %s — shutting down gracefully", sig)
        if _scheduler:
            _scheduler.shutdown()
        if _ws_instance:
            asyncio.ensure_future(_ws_instance.stop())

    try:
        signal.signal(signal.SIGTERM, _handle_shutdown)
        signal.signal(signal.SIGINT, _handle_shutdown)
    except (OSError, ValueError):
        # Windows may not support SIGTERM in all contexts; ignore gracefully
        pass

    # Run everything concurrently
    logger.info("Dashboard: http://%s:%d", settings.api_host, settings.api_port)

    try:
        await asyncio.gather(
            _candle_cache_obj.load_all(
                _active_symbols, _get_client(),
                batch_size=settings.candle_batch_size,
            ),
            ws.run(),
            scheduler.run_all(),
            server.serve(),
        )
    except asyncio.CancelledError:
        logger.info("Tasks cancelled — shutting down")

    # Graceful shutdown with timeout
    try:
        await asyncio.wait_for(_shutdown_cleanup(discord), timeout=30)
    except asyncio.TimeoutError:
        logger.warning("Shutdown timed out after 30s — forcing exit")


def main() -> None:
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        logger.info("Shutdown requested")


if __name__ == "__main__":
    main()
