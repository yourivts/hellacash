# api/routers/orderbook.py
"""API endpoint for order book depth analysis."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from dataclasses import asdict

router = APIRouter(prefix="/api", tags=["orderbook"])

# Singleton reference — set during main.py wiring
_orderbook_provider = None


def set_provider(provider):
    global _orderbook_provider
    _orderbook_provider = provider


@router.get("/orderbook/{symbol}")
async def get_orderbook(symbol: str):
    if _orderbook_provider is None:
        raise HTTPException(503, "Order book provider not initialized")
    analysis = _orderbook_provider.analyze(symbol)
    return {
        "imbalance": analysis.imbalance,
        "spread_pct": analysis.spread_pct,
        "bid_walls": [{"price": w.price, "quantity": w.quantity, "eur_value": w.eur_value} for w in analysis.bid_walls],
        "ask_walls": [{"price": w.price, "quantity": w.quantity, "eur_value": w.eur_value} for w in analysis.ask_walls],
        "support_levels": analysis.support_levels,
        "resistance_levels": analysis.resistance_levels,
        "depth_buckets": [{"price_low": b.price_low, "price_high": b.price_high, "bid_volume": b.bid_volume, "ask_volume": b.ask_volume} for b in analysis.depth_buckets],
    }
