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
from bot.learning.param_optimizer import ParamOptimizer
from bot.learning.regime_classifier import RegimeClassifier
from bot.learning.signal_evaluator import SignalEvaluator
from bot.learning.trade_analyzer import TradeAnalyzer
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
_trading_loop: Optional[TradingLoop] = None
_scheduler: Optional[BotScheduler] = None


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
        )
    return _trading_loop


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
    ws = BitvavoWebSocket(
        api_key=settings.bitvavo_api_key,
        api_secret=settings.bitvavo_api_secret,
        on_candle=trading_loop.on_candle,
        on_ticker=trading_loop.on_ticker,
        on_fill=trading_loop.on_fill,
    )
    ws.set_markets(_active_symbols)
    _ws_instance = ws

    # Create Discord notifier
    discord = DiscordNotifier(settings)

    # Create BotScheduler
    sentiment = _get_sentiment()
    param_optimizer = _get_param_optimizer()

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
    )
    _scheduler = scheduler

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
