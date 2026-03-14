"""Bot control endpoints."""
from __future__ import annotations

from fastapi import APIRouter

import bot.main as bot_main
from bot.exchange.bitvavo_client import get_rate_status

router = APIRouter(prefix="/api/bot", tags=["control"])


@router.get("/status")
async def bot_status():
    portfolio = bot_main._get_portfolio()
    return {
        "running": bot_main.is_running(),
        "paper_trading": bot_main.get_settings().paper_trading,
        "active_symbols": bot_main.get_active_symbols(),
        "open_positions": portfolio.open_position_count(),
        "equity_eur": portfolio.get_equity_eur(),
        "drawdown_pct": portfolio.current_drawdown_pct(),
        "candles_ready": bot_main.get_candle_cache().is_ready,
        "candles_progress": bot_main.get_candle_cache().progress,
        "rate_limit": get_rate_status(),
    }


@router.post("/start")
async def start_bot():
    await bot_main.start_bot()
    return {"status": "started"}


@router.post("/stop")
async def stop_bot():
    await bot_main.stop_bot()
    return {"status": "stopped"}


@router.post("/halt/reset")
async def reset_halt():
    bot_main._get_drawdown().reset_hard_halt()
    return {"status": "halt_reset"}


@router.post("/paper/reset")
async def reset_paper():
    """Reset paper trading balance, drawdown guard, and all trade history."""
    import logging
    logger = logging.getLogger(__name__)

    # Clear DB trade data
    try:
        from bot.data.database import get_session
        from bot.data.repositories import clear_all_trade_data
        async with get_session() as session:
            await clear_all_trade_data(session)
        logger.info("Paper reset: cleared all trade data from DB")
    except Exception as e:
        logger.error("Paper reset: failed to clear DB: %s", e, exc_info=True)

    # Reset in-memory state (always runs even if DB clear fails)
    client = bot_main._get_client()
    client._paper_balance = {"EUR": 10000.0}
    client._save_paper_state()
    drawdown = bot_main._get_drawdown()
    drawdown.reset_hard_halt()
    drawdown._peak_equity = 0.0
    drawdown._daily_loss = 0.0
    portfolio = bot_main._get_portfolio()
    portfolio._positions = {}
    portfolio._peak_equity = 0.0
    portfolio._daily_realized_loss = 0.0
    logger.info("Paper reset: balance restored to €10,000")
    return {"status": "paper_reset", "balance": 10000.0}
