"""Manual position management endpoints."""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from bot.data.database import get_session
from bot.data.repositories import save_trade, update_position
from bot.risk.fees import compute_trade_fees

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/positions", tags=["positions"])


class CloseRequest(BaseModel):
    amount: Optional[float] = None  # None = close entire position


@router.post("/{symbol}/close")
async def close_position(symbol: str, request: Request, body: CloseRequest = CloseRequest()):
    trading_loop = getattr(request.app.state, "trading_loop", None)
    portfolio = getattr(request.app.state, "portfolio", None)
    order_mgr = getattr(request.app.state, "order_mgr", None)
    sentiment = getattr(request.app.state, "sentiment", None)

    if not portfolio or not order_mgr:
        raise HTTPException(status_code=503, detail="Bot not initialized")

    pos = portfolio.get_position(symbol)
    if pos is None:
        raise HTTPException(status_code=404, detail=f"No open position for {symbol}")

    # Check for pending orders
    if trading_loop and trading_loop.is_symbol_pending(symbol):
        raise HTTPException(status_code=409, detail=f"Order already pending for {symbol}")

    direction = pos.get("direction", "LONG")
    quantity = body.amount if body.amount is not None else pos["quantity"]

    if quantity <= 0 or quantity > pos["quantity"]:
        raise HTTPException(status_code=400, detail=f"Invalid amount: {quantity}")

    # Add to pending set to prevent race with trading loop
    if trading_loop:
        trading_loop._pending_orders.add(symbol)

    try:
        # Direction-aware close: LONG → sell, SHORT → buy
        if direction == "SHORT":
            order_id = await order_mgr.submit_buy(symbol, quantity, pos.get("strategy_name", "manual"))
        else:
            order_id = await order_mgr.submit_sell(symbol, quantity, pos.get("strategy_name", "manual"))

        if body.amount is not None and body.amount < pos["quantity"]:
            # Partial close — record trade for closed portion and update position
            remaining = pos["quantity"] - quantity
            exit_price = pos.get("current_price", pos["entry_price"])
            entry_price = pos["entry_price"]

            # Compute P&L for the closed portion
            if direction == "SHORT":
                gross_pnl = (entry_price - exit_price) * quantity
            else:
                gross_pnl = (exit_price - entry_price) * quantity

            fee = compute_trade_fees(
                symbol=symbol, entry_price=entry_price, exit_price=exit_price,
                quantity=quantity, direction=direction, hold_hours=0,
            )
            net_pnl = gross_pnl - fee
            cost_basis = entry_price * quantity
            roi_pct = net_pnl / cost_basis * 100.0 if cost_basis > 0 else 0.0

            # Record the partial close as a trade in DB
            async with get_session() as session:
                await save_trade(
                    session,
                    symbol=symbol,
                    direction=direction,
                    strategy_name=pos.get("strategy_name", "manual"),
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=quantity,
                    gross_pnl=gross_pnl,
                    net_pnl=net_pnl,
                    roi_pct=roi_pct,
                    hold_seconds=0,
                    exit_reason="manual_partial_close",
                    paper_trade=pos.get("paper_trade", True),
                )
                # Update position quantity in DB
                if pos.get("id"):
                    await update_position(session, pos["id"], quantity=remaining)

            # Update in-memory position
            pos["quantity"] = remaining
            logger.info("Partial close %s: sold %.4f, remaining %.4f, P&L €%.2f", symbol, quantity, remaining, net_pnl)
            return {
                "symbol": symbol,
                "closed_amount": quantity,
                "remaining_amount": remaining,
                "order_id": order_id,
                "net_pnl": round(net_pnl, 4),
                "partial": True,
            }
        else:
            # Full close
            exit_price = pos.get("current_price", pos["entry_price"])
            sentiment_score = sentiment.get_score(symbol) if sentiment else 0.0

            result = await portfolio.close_position(symbol, exit_price, order_id, "manual_close", sentiment_score)
            if result is None:
                raise HTTPException(status_code=500, detail="Failed to close position")
            return result
    finally:
        if trading_loop:
            trading_loop._pending_orders.discard(symbol)
