# api/routers/journal.py
"""API endpoints for trade journal and chart serving."""
from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from bot.data.database import get_session

router = APIRouter(prefix="/api", tags=["journal"])

CHARTS_DIR = "data/charts"


@router.get("/journal")
async def get_journal(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    from bot.data.repositories import get_journal_entries
    async with get_session() as session:
        entries = await get_journal_entries(session, limit=limit, offset=offset)
    return [_journal_to_dict(e) for e in entries]


@router.get("/journal/{trade_id}")
async def get_journal_entry(trade_id: int):
    from bot.data.repositories import get_journal_by_trade_id
    async with get_session() as session:
        entry = await get_journal_by_trade_id(session, trade_id)
    if not entry:
        raise HTTPException(404, f"No journal entry for trade {trade_id}")
    return _journal_to_dict(entry)


@router.get("/charts/{filename}")
async def serve_chart(filename: str):
    path = os.path.join(CHARTS_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(404, "Chart not found")
    return FileResponse(path, media_type="image/png")


def _journal_to_dict(entry) -> dict:
    return {
        "id": entry.id,
        "trade_id": entry.trade_id,
        "symbol": entry.symbol,
        "direction": entry.direction,
        "strategy_name": entry.strategy_name,
        "market_regime": entry.market_regime,
        "entry_composite_score": entry.entry_composite_score,
        "entry_reasoning": entry.entry_reasoning,
        "exit_reasoning": entry.exit_reasoning,
        "entry_chart_path": entry.entry_chart_path,
        "exit_chart_path": entry.exit_chart_path,
        "created_at": entry.created_at.isoformat() if entry.created_at else None,
    }
