"""Discord webhook notifier — fire-and-forget alerts."""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from bot.config import Settings
from bot.events.bus import (
    TOPIC_BOT_STARTED,
    TOPIC_RISK_HALT,
    TOPIC_TRADE_CLOSED,
    TOPIC_TRADE_OPENED,
    get_bus,
)

logger = logging.getLogger(__name__)

COLOR_CRITICAL = 0xEF4444
COLOR_WARNING = 0xF59E0B
COLOR_INFO = 0x3B82F6
COLOR_SUCCESS = 0x10B981


class DiscordNotifier:
    """Consumes event bus messages and sends Discord webhook embeds."""

    def __init__(self, settings: Settings) -> None:
        self._webhook_url = settings.discord_webhook_url
        self._notify_trades = settings.discord_notify_trades
        self._session = None
        self._last_error_sent: float = 0.0
        self._error_cooldown = 300

    @property
    def enabled(self) -> bool:
        return bool(self._webhook_url)

    async def run(self) -> None:
        if not self.enabled:
            return
        try:
            import aiohttp
            self._session = aiohttp.ClientSession()
        except ImportError:
            logger.warning("aiohttp not installed — Discord notifications disabled")
            return

        bus = get_bus()
        queues = {}
        topics = [TOPIC_RISK_HALT, TOPIC_BOT_STARTED]
        if self._notify_trades:
            topics.extend([TOPIC_TRADE_OPENED, TOPIC_TRADE_CLOSED])

        for topic in topics:
            queues[topic] = await bus.subscribe(topic)

        logger.info("Discord notifier started (topics: %s)", [t for t in topics])

        while True:
            for topic, queue in queues.items():
                try:
                    event = queue.get_nowait()
                    await self._handle_event(topic, event.payload)
                except asyncio.QueueEmpty:
                    pass
            await asyncio.sleep(1)

    async def _handle_event(self, topic: str, data: Dict[str, Any]) -> None:
        if topic == TOPIC_RISK_HALT:
            await self._send_embed(
                title="CIRCUIT BREAKER TRIGGERED",
                description=data.get("reason", "Trading halted"),
                color=COLOR_CRITICAL,
            )
        elif topic == TOPIC_BOT_STARTED:
            count = len(data.get("symbols", []))
            await self._send_embed(
                title="Bot Started",
                description=f"Trading activated with {count} symbols",
                color=COLOR_INFO,
            )
        elif topic == TOPIC_TRADE_OPENED:
            await self._send_embed(
                title=f"Trade Opened: {data.get('symbol', '?')}",
                description=(
                    f"Strategy: {data.get('strategy_name', '?')}\n"
                    f"Price: EUR{data.get('fill_price', 0):.2f}\n"
                    f"Amount: {data.get('filled_amount', 0):.4f}"
                ),
                color=COLOR_SUCCESS,
            )
        elif topic == TOPIC_TRADE_CLOSED:
            pnl = data.get("net_pnl", 0)
            color = COLOR_SUCCESS if pnl >= 0 else COLOR_CRITICAL
            await self._send_embed(
                title=f"Trade Closed: {data.get('symbol', '?')}",
                description=(
                    f"P&L: EUR{pnl:.2f}\n"
                    f"Reason: {data.get('reason', '?')}"
                ),
                color=color,
            )

    async def send_error(self, message: str) -> None:
        now = time.time()
        if now - self._last_error_sent < self._error_cooldown:
            return
        self._last_error_sent = now
        await self._send_embed(title="Exchange Error", description=message, color=COLOR_WARNING)

    async def _send_embed(self, title: str, description: str, color: int) -> None:
        if not self._session:
            return
        payload = {
            "embeds": [{
                "title": title,
                "description": description,
                "color": color,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "footer": {"text": "HellaCash Bot"},
            }]
        }
        try:
            async with self._session.post(self._webhook_url, json=payload, timeout=5) as resp:
                if resp.status >= 400:
                    logger.warning("Discord webhook returned %d", resp.status)
        except Exception as e:
            logger.debug("Discord send failed: %s", e)

    async def shutdown(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None
