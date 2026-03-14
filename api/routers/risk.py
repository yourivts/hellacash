"""Risk decisions endpoint."""
from __future__ import annotations

import json
from typing import List

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from bot.main import _get_risk_engine

router = APIRouter(prefix="/api/risk", tags=["risk"])


def _sanitize(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    return obj


@router.get("/decisions")
def get_decisions(
    limit: int = Query(20, ge=1, le=100),
):
    """Return the most recent risk gate decisions, newest first."""
    engine = _get_risk_engine()
    decisions = engine.get_recent_decisions(limit=limit)
    return JSONResponse(content=_sanitize(decisions))
