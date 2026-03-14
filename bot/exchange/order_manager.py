"""Order placement, tracking, and fill handling."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from bot.data.database import get_session
from bot.data.models import OrderStatus
from bot.data.repositories import save_order, update_order_status
from bot.events.bus import (
    TOPIC_FILL,
    TOPIC_ORDER_UPDATE,
    TOPIC_TRADE_OPENED,
    get_bus,
)
from bot.exchange.bitvavo_client import BitvavoClient

logger = logging.getLogger(__name__)


class OrderManager:
    def __init__(self, client: BitvavoClient) -> None:
        self.client = client
        self._bus = get_bus()

    async def submit_buy(
        self,
        symbol: str,
        amount_base: float,
        strategy_name: str,
        signal_id: Optional[int] = None,
        price: Optional[float] = None,
        extra_data: Optional[Dict] = None,
    ) -> Optional[int]:
        """Submit a buy order. Returns internal order DB id."""
        try:
            resp = self.client.place_order(symbol, "buy", "market", amount_base, price)
            return await self._record_order(resp, strategy_name, signal_id, extra_data=extra_data)
        except Exception as e:
            logger.error("submit_buy(%s) failed: %s", symbol, e)
            return None

    async def submit_sell(
        self,
        symbol: str,
        amount_base: float,
        strategy_name: str,
        signal_id: Optional[int] = None,
        price: Optional[float] = None,
        extra_data: Optional[Dict] = None,
    ) -> Optional[int]:
        """Submit a sell order. Returns internal order DB id."""
        try:
            resp = self.client.place_order(symbol, "sell", "market", amount_base, price)
            return await self._record_order(resp, strategy_name, signal_id, extra_data=extra_data)
        except Exception as e:
            logger.error("submit_sell(%s) failed: %s", symbol, e)
            return None

    async def _record_order(
        self, resp: Dict[str, Any], strategy_name: str, signal_id: Optional[int],
        extra_data: Optional[Dict] = None,
    ) -> int:
        fill_price = float(resp.get("price", 0) or resp.get("filledAmountQuote", 0))
        filled = float(resp.get("filledAmount", resp.get("amount", 0)) or 0)
        fee = float(resp.get("feePaid", 0) or 0)
        paper = bool(resp.get("paper", False))

        async with get_session() as session:
            order = await save_order(
                session,
                bitvavo_order_id=resp.get("orderId"),
                symbol=resp.get("market"),
                side=resp.get("side"),
                order_type=resp.get("orderType", "market"),
                status=resp.get("status", "filled"),
                requested_price=float(resp.get("price", 0) or 0) or None,
                fill_price=fill_price or None,
                requested_amount=float(resp.get("amount", 0) or 0),
                filled_amount=filled,
                fee=fee,
                fee_currency=resp.get("feeCurrency"),
                paper_trade=paper or self.client.paper_trading,
                strategy_name=strategy_name,
                signal_id=signal_id,
            )
            order_id = order.id

        payload = {
            "order_id": order_id,
            "bitvavo_order_id": resp.get("orderId"),
            "symbol": resp.get("market"),
            "side": resp.get("side"),
            "fill_price": fill_price,
            "filled_amount": filled,
            "fee": fee,
            "strategy_name": strategy_name,
            "paper_trade": paper or self.client.paper_trading,
        }
        if extra_data:
            payload.update(extra_data)
        await self._bus.publish(TOPIC_TRADE_OPENED, payload)
        return order_id

    async def on_fill_notification(self, fill: Dict[str, Any]) -> None:
        """Called by WebSocket handler when a fill arrives."""
        await self._bus.publish(TOPIC_FILL, fill)

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        return self.client.cancel_order(symbol, order_id)
