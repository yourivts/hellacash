"""Bot control endpoints."""
from __future__ import annotations

from fastapi import APIRouter

import bot.main as bot_main

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
