"""API key authentication for REST and WebSocket endpoints."""
from __future__ import annotations

from fastapi import Depends, HTTPException, Query, WebSocket, status
from fastapi.security import APIKeyHeader

from bot.config import get_settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(api_key: str | None = Depends(_api_key_header)) -> None:
    """FastAPI dependency — rejects requests when API_KEY is set but header is missing/wrong."""
    configured_key = get_settings().api_key
    if not configured_key:
        return  # no auth configured
    if api_key != configured_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


async def ws_auth(ws: WebSocket, api_key: str | None = Query(None)) -> bool:
    """Check WebSocket API key from query param. Returns False if rejected.
    Must be called before hub.connect(). If rejected, accepts then immediately closes
    (WebSocket protocol requires accept before close for proper error delivery)."""
    configured_key = get_settings().api_key
    if not configured_key:
        return True
    if api_key != configured_key:
        await ws.accept()
        await ws.close(code=4001, reason="Invalid API key")
        return False
    return True
