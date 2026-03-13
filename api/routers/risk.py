"""Risk decisions endpoint."""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, Query

from bot.main import _get_risk_engine

router = APIRouter(prefix="/api/risk", tags=["risk"])


@router.get("/decisions")
def get_decisions(
    limit: int = Query(20, ge=1, le=100),
) -> List[dict]:
    """Return the most recent risk gate decisions, newest first."""
    engine = _get_risk_engine()
    return engine.get_recent_decisions(limit=limit)
