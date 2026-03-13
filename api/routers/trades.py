"""Trade history endpoints."""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Query

from bot.data.database import get_session
from bot.data.repositories import get_trades

router = APIRouter(prefix="/api/trades", tags=["trades"])


@router.get("")
async def list_trades(
    symbol: Optional[str] = Query(None),
    limit: int = Query(100, le=500),
    offset: int = Query(0, ge=0),
):
    async with get_session() as session:
        trades = await get_trades(session, symbol=symbol, limit=limit, offset=offset)
    return [
        {
            "id": t.id,
            "symbol": t.symbol,
            "strategy_name": t.strategy_name,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "quantity": t.quantity,
            "gross_pnl": t.gross_pnl,
            "net_pnl": t.net_pnl,
            "roi_pct": t.roi_pct,
            "hold_seconds": t.hold_seconds,
            "exit_reason": t.exit_reason,
            "paper_trade": t.paper_trade,
            "created_at": t.created_at.isoformat() if t.created_at else None,
        }
        for t in trades
    ]
