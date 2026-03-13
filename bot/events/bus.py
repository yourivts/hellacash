"""Async in-process event bus using asyncio.Queue per topic."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Event topics ──────────────────────────────────────────────────────────────
TOPIC_CANDLE = "market.candle"
TOPIC_TICKER = "market.ticker"
TOPIC_FILL = "account.fill"
TOPIC_ORDER_UPDATE = "account.order_update"
TOPIC_SIGNAL = "strategy.signal"
TOPIC_TRADE_OPENED = "trade.opened"
TOPIC_TRADE_CLOSED = "trade.closed"
TOPIC_SENTIMENT_UPDATED = "sentiment.updated"
TOPIC_RISK_HALT = "risk.halt"
TOPIC_PORTFOLIO_UPDATE = "portfolio.update"


# ── Event envelope ────────────────────────────────────────────────────────────


@dataclass
class Event:
    topic: str
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ── Bus ───────────────────────────────────────────────────────────────────────


class EventBus:
    """Simple asyncio-based pub/sub bus. All components share one instance."""

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, topic: str, maxsize: int = 256) -> asyncio.Queue:
        async with self._lock:
            q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
            self._subscribers.setdefault(topic, []).append(q)
            return q

    async def publish(self, topic: str, payload: Dict[str, Any]) -> None:
        event = Event(topic=topic, payload=payload)
        queues = self._subscribers.get(topic, [])
        for q in queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("EventBus: queue full for topic %s, dropping event", topic)

    async def publish_event(self, event: Event) -> None:
        await self.publish(event.topic, event.payload)


# ── Singleton ─────────────────────────────────────────────────────────────────

_bus: Optional[EventBus] = None


def get_bus() -> EventBus:
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus
