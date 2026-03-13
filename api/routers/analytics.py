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
    since = datetime.utcnow() - timedelta(days=days)
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


@router.get("/strategies")
async def get_strategy_performance(days: int = Query(30, ge=1, le=365)):
    """Per-strategy breakdown: trades, win rate, total P&L, avg P&L."""
    since = datetime.utcnow() - timedelta(days=days)
    async with get_session() as session:
        trades = await get_trades_since(session, since)

    if not trades:
        return []

    # Group by strategy_name
    groups: dict[str, list] = {}
    for t in trades:
        name = t.strategy_name or "unknown"
        groups.setdefault(name, []).append(t)

    results = []
    for name, strades in groups.items():
        pnls = [t.net_pnl for t in strades]
        wins = [p for p in pnls if p > 0]
        total_pnl = sum(pnls)
        results.append({
            "name": name,
            "total_trades": len(strades),
            "win_rate": len(wins) / len(strades) if strades else 0,
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / len(strades) if strades else 0,
        })

    # Sort by total P&L descending
    results.sort(key=lambda x: x["total_pnl"], reverse=True)
    return results
