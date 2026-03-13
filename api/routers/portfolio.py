"""Portfolio endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Query

from bot.data.database import get_session
from bot.data.repositories import get_snapshots
from bot.main import _get_portfolio

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])


@router.get("")
async def get_portfolio():
    portfolio = _get_portfolio()
    return {
        "equity_eur": portfolio.get_equity_eur(),
        "cash_eur": portfolio.get_cash_eur(),
        "positions_value_eur": portfolio.get_positions_value_eur(),
        "open_positions": portfolio.get_all_positions(),
        "drawdown_pct": portfolio.current_drawdown_pct(),
        "daily_realized_loss_eur": portfolio.daily_realized_loss_eur(),
    }


@router.get("/history")
async def get_history(days: int = Query(7, ge=1, le=90)):
    async with get_session() as session:
        snaps = await get_snapshots(session, days=days)
    return [
        {
            "total_equity_eur": s.total_equity_eur,
            "daily_pnl": s.daily_pnl,
            "max_drawdown_pct": s.max_drawdown_pct,
            "win_rate": s.win_rate,
            "snapshot_at": s.snapshot_at.isoformat() if s.snapshot_at else None,
        }
        for s in snaps
    ]
