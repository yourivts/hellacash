"""Sentiment endpoints."""
from __future__ import annotations

import asyncio
import json
import logging
import urllib.request

from fastapi import APIRouter, Path, Query

from bot.data.database import get_session
from bot.data.repositories import get_latest_sentiment, get_sentiment_history
from bot.main import _get_sentiment

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/sentiment", tags=["sentiment"])

# In-memory fallback cache for when aggregator hasn't run yet
_fallback_cache: dict[str, float] = {}


def _fetch_fear_greed() -> float:
    """Quick sync fetch of Fear & Greed Index score."""
    try:
        req = urllib.request.Request(
            "https://api.alternative.me/fng/?limit=1",
            headers={"User-Agent": "hellacash/1.0"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        value = int(data["data"][0]["value"])
        return (value / 50.0) - 1.0  # 0..100 → -1..+1
    except Exception:
        return 0.0


@router.get("/all/scores")
async def get_all_sentiment():
    """Return all cached sentiment scores in one call, sorted by absolute score descending."""
    aggregator = _get_sentiment()
    scores = [
        {"symbol": f"{base}-EUR", "base": base, "score": score}
        for base, score in aggregator._cache.items()
    ]
    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores


@router.get("/{symbol}")
async def get_sentiment(symbol: str = Path(...), hours: int = Query(24, ge=1, le=168)):
    # Latest score from aggregator cache
    aggregator = _get_sentiment()
    current_score = aggregator.get_score(symbol)
    base = symbol.split("-")[0].upper()

    # If aggregator cache is empty, try DB, then live Fear & Greed
    if current_score == 0.0 and not aggregator._cache.get(base):
        # Try DB first
        async with get_session() as session:
            latest = await get_latest_sentiment(session, symbol)
            if latest:
                current_score = latest.score
            else:
                # No DB data either — fetch Fear & Greed live as fallback
                if base not in _fallback_cache:
                    loop = asyncio.get_event_loop()
                    _fallback_cache[base] = await loop.run_in_executor(None, _fetch_fear_greed)
                current_score = _fallback_cache.get(base, 0.0)

    # Historical from DB
    async with get_session() as session:
        history = await get_sentiment_history(session, symbol, hours=hours)

    return {
        "symbol": symbol,
        "current_score": current_score,
        "history": [
            {
                "score": s.score,
                "source": s.source,
                "computed_at": s.computed_at.isoformat() if s.computed_at else None,
            }
            for s in history
        ],
    }
