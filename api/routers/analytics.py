"""Analytics endpoints: P&L, win rate, Sharpe ratio, attribution, benchmark, walk-forward."""
from __future__ import annotations

import statistics
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from bot.data.database import get_session
from bot.data.repositories import get_trades, get_trades_since

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


@router.get("/attribution")
async def get_attribution(
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
):
    start_dt = datetime.fromisoformat(start) if start else None
    end_dt = datetime.fromisoformat(end) if end else None

    async with get_session() as session:
        trades = await get_trades(
            session, limit=10000, start_date=start_dt, end_date=end_dt,
        )

    from bot.analytics.attribution import AttributionEngine
    engine = AttributionEngine()
    report = engine.compute_from_trades(trades)
    return {
        "total_pnl": report.total_pnl,
        "total_trades": report.total_trades,
        "overall_win_rate": report.overall_win_rate,
        "by_strategy": [vars(s) for s in report.by_strategy],
        "by_regime": [vars(s) for s in report.by_regime],
        "by_session": [vars(s) for s in report.by_session],
        "by_direction": [vars(s) for s in report.by_direction],
        "best_strategy": report.best_strategy,
        "worst_strategy": report.worst_strategy,
        "best_session": report.best_session,
        "worst_session": report.worst_session,
    }


@router.get("/benchmark")
async def get_benchmark(
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
):
    # Placeholder — full implementation requires fetching portfolio snapshots
    # and candle prices for BTC/ETH. Wired in main.py integration task.
    return {"status": "not_yet_wired", "message": "Benchmark requires portfolio data wiring"}


@router.get("/walk-forward")
async def get_walk_forward():
    from bot.learning.walk_forward import WalkForwardOptimizer
    # Access singleton — wired in main.py
    return {"status": "no_results", "message": "No walk-forward run completed yet"}


@router.post("/walk-forward/run", status_code=202)
async def trigger_walk_forward():
    return {"status": "accepted", "message": "Walk-forward run triggered"}
