"""Candle data endpoint for charts."""
from __future__ import annotations

from fastapi import APIRouter, Path, Query

import bot.main as bot_main

router = APIRouter(prefix="/api/candles", tags=["charts"])


@router.get("/{symbol}")
async def get_candles(
    symbol: str = Path(...),
    interval: str = Query("5m"),
    limit: int = Query(200, le=500),
):
    df = bot_main._get_df(symbol, interval)
    if df.empty:
        return []
    rows = df.tail(limit).reset_index()
    return [
        {
            "timestamp": str(r["timestamp"]),
            "open": r["open"],
            "high": r["high"],
            "low": r["low"],
            "close": r["close"],
            "volume": r["volume"],
        }
        for _, r in rows.iterrows()
    ]
