"""Health check endpoint."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List

import aiohttp
from fastapi import APIRouter
from sqlalchemy import text

from bot.data.database import get_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/health", tags=["health"])

_start_time = time.time()


async def _check_database() -> Dict[str, Any]:
    """Run SELECT 1 against the database and measure latency."""
    t0 = time.monotonic()
    try:
        async with get_session() as session:
            await session.execute(text("SELECT 1"))
        latency_ms = round((time.monotonic() - t0) * 1000)
        return {"status": "ok", "latency_ms": latency_ms}
    except Exception as exc:
        logger.warning("Health check: database unreachable: %s", exc)
        return {"status": "error", "error": str(exc)}


async def _check_exchange() -> Dict[str, Any]:
    """Hit Bitvavo GET /v2/time with a 5s timeout."""
    t0 = time.monotonic()
    try:
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get("https://api.bitvavo.com/v2/time") as resp:
                resp.raise_for_status()
        latency_ms = round((time.monotonic() - t0) * 1000)
        return {"status": "ok", "latency_ms": latency_ms}
    except Exception as exc:
        logger.warning("Health check: exchange API unreachable: %s", exc)
        return {"status": "error", "error": str(exc)}


def _check_websocket() -> Dict[str, str]:
    """Check if the Bitvavo WebSocket connection is active."""
    try:
        from bot.main import get_ws
        ws = get_ws()
        if ws is not None and ws.is_connected:
            return {"status": "connected"}
    except Exception as exc:
        logger.warning("Health check: websocket check failed: %s", exc)
    return {"status": "disconnected"}


def _check_sentiment() -> Dict[str, Any]:
    """Check sentiment aggregator for degraded sources."""
    degraded_sources: List[str] = []
    try:
        from bot.main import _get_sentiment
        from bot.sentiment.aggregator import SOURCE_WEIGHTS
        sentiment = _get_sentiment()
        for source in SOURCE_WEIGHTS:
            if sentiment._is_source_disabled(source):
                degraded_sources.append(source)
    except Exception as exc:
        logger.warning("Health check: sentiment check failed: %s", exc)
        return {"status": "error", "degraded_sources": list(SOURCE_WEIGHTS.keys())}

    status = "degraded" if degraded_sources else "ok"
    return {"status": status, "degraded_sources": degraded_sources}


def _compute_overall_status(checks: Dict[str, Dict[str, Any]]) -> str:
    """Derive top-level status from individual checks.

    - "unhealthy" if database or exchange are unreachable
    - "degraded"  if websocket is disconnected or sentiment sources are down
    - "healthy"   otherwise
    """
    db_ok = checks.get("database", {}).get("status") == "ok"
    exchange_ok = checks.get("exchange_api", {}).get("status") == "ok"

    if not db_ok or not exchange_ok:
        return "unhealthy"

    ws_connected = checks.get("websocket", {}).get("status") == "connected"
    sentiment_ok = checks.get("sentiment", {}).get("status") == "ok"

    if not ws_connected or not sentiment_ok:
        return "degraded"

    return "healthy"


@router.get("")
async def health() -> Dict[str, Any]:
    # Run DB and exchange checks concurrently (they are async/IO-bound).
    # Wrap each in a 10s timeout so the endpoint never hangs.
    db_task = asyncio.create_task(_check_database())
    exchange_task = asyncio.create_task(_check_exchange())

    try:
        db_result = await asyncio.wait_for(db_task, timeout=10)
    except asyncio.TimeoutError:
        db_result = {"status": "error", "error": "timeout"}

    try:
        exchange_result = await asyncio.wait_for(exchange_task, timeout=10)
    except asyncio.TimeoutError:
        exchange_result = {"status": "error", "error": "timeout"}

    # Synchronous checks
    ws_result = _check_websocket()
    sentiment_result = _check_sentiment()

    checks = {
        "database": db_result,
        "exchange_api": exchange_result,
        "websocket": ws_result,
        "sentiment": sentiment_result,
    }

    return {
        "status": _compute_overall_status(checks),
        "checks": checks,
        "uptime_seconds": round(time.time() - _start_time),
        "version": "1.0.0",
    }
