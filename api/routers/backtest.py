"""Backtest endpoint: run historical strategy simulation."""
from __future__ import annotations

import asyncio
from typing import List

from fastapi import APIRouter, Query
from pydantic import BaseModel

from bot.backtest.engine import run_backtest

router = APIRouter(prefix="/api/backtest", tags=["backtest"])


class CompareRequest(BaseModel):
    symbols: List[str]
    days: int = 30
    interval: str = "5m"
    initial_capital: float = 10_000.0
    max_open_positions: int = 3


@router.post("/compare")
async def backtest_compare(req: CompareRequest):
    """Run backtests for multiple symbols with rate-limited concurrency."""
    sem = asyncio.Semaphore(2)

    async def run_one(symbol: str) -> dict:
        if "-" not in symbol:
            symbol = symbol[:3] + "-" + symbol[3:]
        symbol = symbol.upper()
        async with sem:
            try:
                result = await run_backtest(
                    symbol=symbol, interval=req.interval, days=req.days,
                    initial_capital=req.initial_capital,
                    max_open_positions=req.max_open_positions,
                )
                return {
                    "symbol": symbol,
                    "total_pnl": result.total_pnl,
                    "total_return_pct": result.total_return_pct,
                    "win_rate": result.win_rate,
                    "total_trades": result.total_trades,
                    "sharpe_ratio": result.sharpe_ratio,
                    "max_drawdown_pct": result.max_drawdown_pct,
                }
            except Exception as e:
                return {"symbol": symbol, "error": str(e)}

    tasks = [run_one(s) for s in req.symbols]
    results = await asyncio.gather(*tasks)

    return {
        "results": results,
        "params": {
            "days": req.days,
            "interval": req.interval,
            "initial_capital": req.initial_capital,
        },
    }


@router.get("/{symbol}")
async def backtest_symbol(
    symbol: str,
    interval: str = Query("5m", description="Candle interval (1m, 5m, 15m, 1h, etc.)"),
    days: int = Query(30, ge=1, le=365, description="Number of days of history"),
    initial_capital: float = Query(10_000.0, ge=100, description="Starting capital in EUR"),
    max_open_positions: int = Query(3, ge=1, le=10, description="Max concurrent positions"),
):
    """Run a backtest for the given symbol and return performance metrics."""
    # Normalise symbol: accept BTC-EUR or BTCEUR
    if "-" not in symbol:
        symbol = symbol[:3] + "-" + symbol[3:]
    symbol = symbol.upper()

    result = await run_backtest(
        symbol=symbol, interval=interval, days=days,
        initial_capital=initial_capital, max_open_positions=max_open_positions,
    )

    return {
        "symbol": symbol,
        "interval": interval,
        "days": days,
        "total_pnl": result.total_pnl,
        "total_return_pct": result.total_return_pct,
        "win_rate": result.win_rate,
        "total_trades": result.total_trades,
        "winning_trades": result.winning_trades,
        "losing_trades": result.losing_trades,
        "sharpe_ratio": result.sharpe_ratio,
        "max_drawdown_pct": result.max_drawdown_pct,
        "avg_win_pct": result.avg_win_pct,
        "avg_loss_pct": result.avg_loss_pct,
        "trade_log": [
            {
                "symbol": t.symbol,
                "direction": t.direction,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "size_eur": t.size_eur,
                "pnl_eur": t.pnl_eur,
                "pnl_pct": t.pnl_pct,
                "exit_reason": t.exit_reason,
                "strategy": t.strategy,
            }
            for t in result.trade_log
        ],
    }
