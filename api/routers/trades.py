"""Trade history endpoints."""
from __future__ import annotations

import csv
import io
from datetime import date
from typing import List, Literal, Optional

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from bot.data.database import get_session
from bot.data.repositories import get_trades

router = APIRouter(prefix="/api/trades", tags=["trades"])


@router.get("/export")
async def export_trades(
    format: Literal["tax", "full"] = Query("full"),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    limit: int = Query(10000, le=100000),
):
    async with get_session() as session:
        trades = await get_trades(session, limit=limit)

    # Filter by date if provided
    if start_date:
        start = date.fromisoformat(start_date)
        trades = [t for t in trades if t.created_at and t.created_at.date() >= start]
    if end_date:
        end = date.fromisoformat(end_date)
        trades = [t for t in trades if t.created_at and t.created_at.date() <= end]

    output = io.StringIO()
    writer = csv.writer(output)

    if format == "tax":
        writer.writerow(["date", "pair", "direction", "quantity", "entry_price", "exit_price", "fees", "realized_pnl", "cost_basis"])
        for t in trades:
            fees = round((t.gross_pnl or 0) - (t.net_pnl or 0), 4)
            cost_basis = round((t.entry_price or 0) * (t.quantity or 0), 4)
            writer.writerow([
                t.created_at.isoformat() if t.created_at else "",
                t.symbol,
                getattr(t, "direction", "LONG"),
                t.quantity,
                t.entry_price,
                t.exit_price,
                fees,
                t.net_pnl,
                cost_basis,
            ])
    else:
        writer.writerow(["id", "date", "symbol", "direction", "strategy_name", "entry_price", "exit_price", "quantity", "gross_pnl", "net_pnl", "roi_pct", "hold_seconds", "exit_reason", "paper_trade"])
        for t in trades:
            writer.writerow([
                t.id,
                t.created_at.isoformat() if t.created_at else "",
                t.symbol,
                getattr(t, "direction", "LONG"),
                t.strategy_name,
                t.entry_price,
                t.exit_price,
                t.quantity,
                t.gross_pnl,
                t.net_pnl,
                t.roi_pct,
                t.hold_seconds,
                t.exit_reason,
                t.paper_trade,
            ])

    output.seek(0)
    today = date.today().isoformat()
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=trades_{today}.csv"},
    )


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
