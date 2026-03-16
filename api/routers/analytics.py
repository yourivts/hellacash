"""Analytics endpoints: P&L, win rate, Sharpe ratio, attribution, benchmark, walk-forward."""
from __future__ import annotations

import statistics
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from bot.data.database import get_session
from bot.data.repositories import get_snapshots_since, get_trades, get_trades_since

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


def _get_equity_pnl() -> dict:
    """Compute P&L from equity delta (captures fees, unrealized, everything)."""
    try:
        from bot.main import _get_portfolio
        portfolio = _get_portfolio()
        equity = portfolio.get_equity_eur()
        # Paper trading starts at 10000; live uses first snapshot as baseline
        from bot.config import get_settings
        initial = 10000.0 if get_settings().paper_trading else equity
        if not get_settings().paper_trading:
            # For live, try to get initial equity from first-ever snapshot
            pass  # equity delta not reliable for live without baseline
        return {
            "equity": equity,
            "total_pnl": equity - initial,
        }
    except Exception:
        return {"equity": 0.0, "total_pnl": 0.0}


async def _get_daily_pnl_from_snapshots() -> float:
    """Compute daily P&L from first snapshot of today vs current equity."""
    try:
        from bot.main import _get_portfolio
        portfolio = _get_portfolio()
        equity = portfolio.get_equity_eur()
        async with get_session() as session:
            snaps = await get_snapshots_since(session, datetime.utcnow().replace(hour=0, minute=0, second=0))
        if snaps:
            return equity - snaps[0].total_equity_eur
        # No snapshot today yet — fall back to equity-based total P&L
        from bot.config import get_settings
        if get_settings().paper_trading:
            return equity - 10000.0
        return 0.0
    except Exception:
        return 0.0


@router.get("")
async def get_analytics(days: int = Query(30, ge=1, le=365)):
    since = datetime.utcnow() - timedelta(days=days)
    async with get_session() as session:
        trades = await get_trades_since(session, since)

    # Equity-based P&L (captures everything: fees, unrealized, etc.)
    equity_info = _get_equity_pnl()
    total_pnl = equity_info["total_pnl"]
    daily_pnl = await _get_daily_pnl_from_snapshots()

    if not trades:
        return {
            "total_trades": 0,
            "win_rate": None,
            "total_pnl": total_pnl,
            "daily_pnl": daily_pnl,
            "avg_roi_pct": None,
            "sharpe_ratio": None,
            "max_drawdown_pct": None,
            "days": days,
        }

    pnls = [t.net_pnl for t in trades]
    roi_pcts = [t.roi_pct for t in trades]
    wins = [p for p in pnls if p > 0]
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
        "daily_pnl": daily_pnl,
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
    from bot.main import _get_walk_forward
    wf = _get_walk_forward()
    result = wf.latest_result
    if result is None:
        return {"status": "no_results", "message": "No walk-forward run completed yet"}
    return {
        "status": "completed",
        "adopted": result.adopted,
        "avg_oos_sharpe": result.avg_oos_sharpe,
        "avg_oos_pnl": result.avg_oos_pnl,
        "sharpe_stability": result.sharpe_stability,
        "recommended_params": result.recommended_params,
        "windows": len(result.windows),
    }


@router.post("/walk-forward/run", status_code=202)
async def trigger_walk_forward():
    import asyncio
    from bot.main import _get_walk_forward, _get_universe, get_tradeable_symbols, _get_client
    from bot.learning.walk_forward import TRAIN_DAYS, TEST_DAYS, MIN_WINDOWS

    wf = _get_walk_forward()
    if wf.is_running:
        return {"status": "already_running", "message": "Walk-forward is already running"}

    async def _run():
        import logging
        logger = logging.getLogger(__name__)

        def _candle_fetcher_factory(symbol):
            async def _fetcher(start_day, end_day):
                from datetime import datetime, timedelta, timezone
                now = datetime.now(timezone.utc)
                total_days = TRAIN_DAYS + TEST_DAYS * MIN_WINDOWS
                start_dt = now - timedelta(days=total_days - start_day)
                end_dt = now - timedelta(days=total_days - end_day)
                loop = asyncio.get_running_loop()
                client = _get_client()
                return await loop.run_in_executor(
                    None, lambda: client.get_candles_range(symbol, "5m", start_dt, end_dt)
                )
            return _fetcher

        symbols = get_tradeable_symbols() or ["BTC-EUR"]
        results = await wf.run_multi_per_strategy(
            _candle_fetcher_factory, symbols,
        )
        universe = _get_universe()
        universe.update(results)
        logger.info("Walk-forward completed via API trigger — %d symbols adopted", universe.count())

    asyncio.create_task(_run())
    return {"status": "accepted", "message": "Walk-forward run started in background"}


@router.get("/universe")
async def get_universe():
    """Return the current adopted universe."""
    from bot.main import _get_universe
    universe = _get_universe()
    symbols = {}
    total_combos = 0
    for sym in universe.adopted_symbols():
        strategies = {}
        for strat in universe.get_enabled_strategies(sym):
            sp = universe.get_strategy_params(sym, strat)
            sc = universe._adopted[sym].strategies[strat]
            strategies[strat] = {
                "params": sp,
                "avg_sharpe": sc.avg_sharpe,
                "avg_pnl": sc.avg_pnl,
            }
            total_combos += 1
        symbols[sym] = {
            "regime_params": universe.get_regime_params(sym),
            "strategies": strategies,
        }
    return {
        "adopted_count": universe.count(),
        "strategy_combo_count": total_combos,
        "symbols": symbols,
    }
