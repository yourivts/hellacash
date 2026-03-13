"""Sentiment endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Path, Query

from bot.data.database import get_session
from bot.data.repositories import get_latest_sentiment, get_sentiment_history
from bot.main import _get_sentiment

router = APIRouter(prefix="/api/sentiment", tags=["sentiment"])


@router.get("/{symbol}")
async def get_sentiment(symbol: str = Path(...), hours: int = Query(24, ge=1, le=168)):
    # Latest score from aggregator cache
    aggregator = _get_sentiment()
    current_score = aggregator.get_score(symbol)

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
