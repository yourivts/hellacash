"""Limit order lifecycle management."""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PendingOrder:
    order_id: str
    symbol: str
    direction: str
    limit_price: float
    size_eur: float
    stop_loss: float
    take_profit: float
    strategy: str
    status: str = "pending"
    created_at: float = field(default_factory=time.time)
    exchange_order_id: Optional[str] = None
    fill_price: Optional[float] = None
    filled_eur: float = 0.0


class LimitOrderManager:
    def __init__(self, client, timeout_seconds=600, max_price_deviation_pct=0.3):
        self._client = client
        self._timeout = timeout_seconds
        self._max_deviation = max_price_deviation_pct
        self.pending_orders: Dict[str, PendingOrder] = {}
        self.filled_orders: List[PendingOrder] = []

    def create_pending(self, symbol, direction, limit_price, size_eur,
                       stop_loss, take_profit, strategy) -> PendingOrder:
        order = PendingOrder(
            order_id=str(uuid.uuid4())[:8], symbol=symbol, direction=direction,
            limit_price=limit_price, size_eur=size_eur, stop_loss=stop_loss,
            take_profit=take_profit, strategy=strategy)
        self.pending_orders[order.order_id] = order
        logger.info("Pending order %s: %s %s @ %.2f", order.order_id, direction, symbol, limit_price)
        return order

    def check_timeouts(self) -> List[PendingOrder]:
        now = time.time()
        timed_out = []
        for oid in list(self.pending_orders):
            order = self.pending_orders[oid]
            if now - order.created_at >= self._timeout:
                order.status = "timed_out"
                del self.pending_orders[oid]
                timed_out.append(order)
        return timed_out

    def check_price_deviation(self, symbol, current_price) -> List[PendingOrder]:
        cancelled = []
        for oid in list(self.pending_orders):
            order = self.pending_orders[oid]
            if order.symbol != symbol:
                continue
            deviation_pct = abs(current_price - order.limit_price) / order.limit_price * 100
            if deviation_pct > self._max_deviation:
                order.status = "cancelled"
                del self.pending_orders[oid]
                cancelled.append(order)
        return cancelled

    def mark_filled(self, order_id, fill_price) -> Optional[PendingOrder]:
        if order_id not in self.pending_orders:
            return None
        order = self.pending_orders.pop(order_id)
        order.status = "filled"
        order.fill_price = fill_price
        self.filled_orders.append(order)
        return order

    def mark_partial_fill(self, order_id, filled_eur, fill_price) -> Optional[PendingOrder]:
        if order_id not in self.pending_orders:
            return None
        order = self.pending_orders[order_id]
        if filled_eur >= order.size_eur:
            return self.mark_filled(order_id, fill_price)

        partial = PendingOrder(
            order_id=f"{order_id}-p", symbol=order.symbol, direction=order.direction,
            limit_price=order.limit_price, size_eur=order.size_eur,
            stop_loss=order.stop_loss, take_profit=order.take_profit,
            strategy=order.strategy, status="filled", fill_price=fill_price)
        partial.filled_eur = filled_eur
        self.filled_orders.append(partial)
        order.size_eur -= filled_eur
        return partial

    def cancel_all(self, symbol=None) -> List[PendingOrder]:
        cancelled = []
        for oid in list(self.pending_orders):
            order = self.pending_orders[oid]
            if symbol and order.symbol != symbol:
                continue
            order.status = "cancelled"
            del self.pending_orders[oid]
            cancelled.append(order)
        return cancelled

    def get_pending_for_symbol(self, symbol) -> List[PendingOrder]:
        return [o for o in self.pending_orders.values() if o.symbol == symbol]
