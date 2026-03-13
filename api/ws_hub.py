"""WebSocket connection manager for live dashboard feed."""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Set

from fastapi import WebSocket, WebSocketDisconnect

from bot.events.bus import (
    TOPIC_PORTFOLIO_UPDATE,
    TOPIC_RISK_HALT,
    TOPIC_SENTIMENT_UPDATED,
    TOPIC_TICKER,
    TOPIC_TRADE_CLOSED,
    TOPIC_TRADE_OPENED,
    get_bus,
)

logger = logging.getLogger(__name__)


class WebSocketHub:
    def __init__(self) -> None:
        self._connections: Set[WebSocket] = set()
        self._bus = get_bus()
        self._started = False

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.add(ws)
        logger.debug("WS client connected (total=%d)", len(self._connections))

    async def disconnect(self, ws: WebSocket) -> None:
        self._connections.discard(ws)

    async def broadcast(self, event_type: str, data: dict) -> None:
        if not self._connections:
            return
        payload = json.dumps({"type": event_type, "data": data, "ts": datetime.utcnow().isoformat()})
        dead = set()
        for ws in list(self._connections):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.add(ws)
        self._connections -= dead

    async def start_relay(self) -> None:
        """Subscribe to EventBus topics and relay to WebSocket clients."""
        if self._started:
            return
        self._started = True

        topics = [
            TOPIC_TICKER,
            TOPIC_TRADE_OPENED,
            TOPIC_TRADE_CLOSED,
            TOPIC_SENTIMENT_UPDATED,
            TOPIC_RISK_HALT,
            TOPIC_PORTFOLIO_UPDATE,
        ]
        queues = []
        for topic in topics:
            q = await self._bus.subscribe(topic)
            queues.append((topic, q))

        asyncio.create_task(self._relay_loop(queues))

    async def _relay_loop(self, queues) -> None:
        while True:
            for topic, q in queues:
                try:
                    event = q.get_nowait()
                    await self.broadcast(topic, event.payload)
                except asyncio.QueueEmpty:
                    pass
            await asyncio.sleep(0.05)


# Singleton
_hub: WebSocketHub | None = None


def get_hub() -> WebSocketHub:
    global _hub
    if _hub is None:
        _hub = WebSocketHub()
    return _hub
