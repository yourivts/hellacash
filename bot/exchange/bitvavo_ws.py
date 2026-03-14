"""Bitvavo WebSocket client with auto-reconnect."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Callable, List, Optional, Set

logger = logging.getLogger(__name__)

try:
    import websockets  # type: ignore
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    logger.warning("websockets not installed; WebSocket feed disabled")


class BitvavoWebSocket:
    """
    Maintains a persistent WebSocket connection to Bitvavo.
    Dispatches received messages to registered handler callbacks.
    Auto-reconnects with exponential backoff on failure.
    """

    WS_URL = "wss://ws.bitvavo.com/v2/"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        on_candle: Optional[Callable] = None,
        on_ticker: Optional[Callable] = None,
        on_fill: Optional[Callable] = None,
        on_book: Optional[Callable] = None,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self._on_candle = on_candle
        self._on_ticker = on_ticker
        self._on_fill = on_fill
        self._on_book = on_book
        self._subscribed_markets: Set[str] = set()
        self._running = False
        self._ws = None

    @property
    def is_connected(self) -> bool:
        """Whether the WebSocket connection is active."""
        return self._running and self._ws is not None

    def set_markets(self, markets: List[str]) -> None:
        self._subscribed_markets = set(markets)

    async def run(self) -> None:
        """Main loop: connect → subscribe → receive → reconnect on error."""
        if not HAS_WEBSOCKETS:
            logger.info("WebSocket disabled (websockets package missing)")
            return

        self._running = True
        backoff = 1.0
        while self._running:
            try:
                await self._connect_and_listen()
                backoff = 1.0  # reset on clean exit
            except Exception as e:
                if not self._running:
                    break
                logger.error("WebSocket error: %s — reconnecting in %.1fs", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    async def _connect_and_listen(self) -> None:
        async with websockets.connect(self.WS_URL, ping_interval=20, ping_timeout=30) as ws:
            self._ws = ws
            logger.info("WebSocket connected")
            await self._subscribe(ws)
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                    await self._dispatch(msg)
                except json.JSONDecodeError:
                    pass

    async def _subscribe(self, ws) -> None:
        """Subscribe to ticker, candles (5m), and account channels."""
        markets = list(self._subscribed_markets)
        if not markets:
            return

        # Ticker subscription
        await ws.send(json.dumps({
            "action": "subscribe",
            "channels": [{"name": "ticker", "markets": markets}],
        }))

        # Candle subscription (1m for fast data, 5m for strategy)
        await ws.send(json.dumps({
            "action": "subscribe",
            "channels": [{"name": "candles", "markets": markets, "interval": ["1m", "5m"]}],
        }))

        # Order book subscription
        if self._on_book and self._subscribed_markets:
            await ws.send(json.dumps({
                "action": "subscribe",
                "channels": [{"name": "book", "markets": markets}],
            }))

        # Account subscription (requires auth)
        if self.api_key and self.api_secret:
            ts = int(time.time() * 1000)
            import hmac, hashlib
            sig = hmac.new(
                self.api_secret.encode(), f"{ts}GET/v2/websocket".encode(), hashlib.sha256
            ).hexdigest()
            await ws.send(json.dumps({
                "action": "authenticate",
                "key": self.api_key,
                "signature": sig,
                "timestamp": ts,
            }))
            await ws.send(json.dumps({
                "action": "subscribe",
                "channels": [{"name": "account", "markets": markets}],
            }))

        logger.info("WebSocket subscribed to %d markets", len(markets))

    async def _dispatch(self, msg: dict) -> None:
        event = msg.get("event", "")

        if event == "candle" and self._on_candle:
            for c in msg.get("candle", []):
                await self._on_candle({
                    "symbol": msg["market"],
                    "interval": msg["interval"],
                    "timestamp": datetime.fromtimestamp(c[0] / 1000, tz=timezone.utc),
                    "open": float(c[1]),
                    "high": float(c[2]),
                    "low": float(c[3]),
                    "close": float(c[4]),
                    "volume": float(c[5]),
                })

        elif event == "ticker" and self._on_ticker:
            await self._on_ticker({
                "symbol": msg.get("market"),
                "price": float(msg.get("lastPrice", 0) or 0),
                "bid": float(msg.get("bestBid", 0) or 0),
                "ask": float(msg.get("bestAsk", 0) or 0),
                "timestamp": datetime.now(timezone.utc),
            })

        elif event == "fill" and self._on_fill:
            await self._on_fill({
                "order_id": msg.get("orderId"),
                "symbol": msg.get("market"),
                "side": msg.get("side"),
                "price": float(msg.get("price", 0) or 0),
                "amount": float(msg.get("amount", 0) or 0),
                "fee": float(msg.get("fee", 0) or 0),
                "fee_currency": msg.get("feeCurrency"),
                "timestamp": datetime.now(timezone.utc),
            })

        elif event == "book" and self._on_book:
            await self._on_book(msg)

    async def stop(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()
