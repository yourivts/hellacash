"""Analytics endpoints: P&L, win rate, Sharpe ratio."""
from __future__ import annotations

import statistics
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Query

from bot.data.database import get_session
from bot.data.repositories import get_trades_since

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("")
async def get_analytics(days: int = Query(30, ge=1, le=365)):
    since = datetime.now(timezone.utc) - timedelta(days=days)
    async with get_session() as session:
        trades = await get_trades_since(session, since)

    if not trades:
        return {
            "total_trades": 0,
            "win_rate": None,
            "total_pnl": 0.0,
            "avg_roi_pct": None,
            "sharpe_ratio": None,
            "max_drawdown_pct": None,
            "days": days,
        }

    pnls = [t.net_pnl for t in trades]
    roi_pcts = [t.roi_pct for t in trades]
    wins = [p for p in pnls if p > 0]
    total_pnl = sum(pnls)
    win_rate = len(wins) / len(pnls)
    avg_roi = statistics.mean(roi_pcts) if roi_pcts else 0.0

    # Simplified Sharpe: mean(roi) / std(roi)
    sharpe = None
    if len(roi_pcts) > 1:
        std_roi = statistics.stdev(roi_pcts)
        if std_roi > 0:
            sharpe = (statistics.mean(roi_pcts) / std_roi) * (252 ** 0.5)  # annualised

    return {
        "total_trades": len(trades),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_roi_pct": avg_roi,
        "sharpe_ratio": sharpe,
        "best_trade": max(pnls) if pnls else None,
        "worst_trade": min(pnls) if pnls else None,
        "days": days,
    }
