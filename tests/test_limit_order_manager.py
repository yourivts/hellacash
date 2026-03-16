import time
from unittest.mock import MagicMock
from bot.execution.limit_order_manager import LimitOrderManager, PendingOrder

class TestLimitOrderManager:
    def test_create_pending_order(self):
        mgr = LimitOrderManager(client=MagicMock())
        order = mgr.create_pending(
            symbol="BTC-EUR", direction="LONG", limit_price=50000.0,
            size_eur=100.0, stop_loss=48000.0, take_profit=53000.0, strategy="orderflow")
        assert order.symbol == "BTC-EUR"
        assert order.direction == "LONG"
        assert order.status == "pending"
        assert len(mgr.pending_orders) == 1

    def test_check_timeout_cancels(self):
        mgr = LimitOrderManager(client=MagicMock(), timeout_seconds=0)
        order = mgr.create_pending(
            symbol="BTC-EUR", direction="LONG", limit_price=50000.0,
            size_eur=100.0, stop_loss=48000.0, take_profit=53000.0, strategy="orderflow")
        order.created_at = time.time() - 1
        timed_out = mgr.check_timeouts()
        assert len(timed_out) == 1
        assert timed_out[0].status == "timed_out"
        assert len(mgr.pending_orders) == 0

    def test_price_moved_away_cancels(self):
        mgr = LimitOrderManager(client=MagicMock(), max_price_deviation_pct=0.3)
        mgr.create_pending(
            symbol="BTC-EUR", direction="LONG", limit_price=50000.0,
            size_eur=100.0, stop_loss=48000.0, take_profit=53000.0, strategy="orderflow")
        cancelled = mgr.check_price_deviation(symbol="BTC-EUR", current_price=50250.0)
        assert len(cancelled) == 1

    def test_price_within_range_keeps_order(self):
        mgr = LimitOrderManager(client=MagicMock(), max_price_deviation_pct=0.3)
        mgr.create_pending(
            symbol="BTC-EUR", direction="LONG", limit_price=50000.0,
            size_eur=100.0, stop_loss=48000.0, take_profit=53000.0, strategy="orderflow")
        cancelled = mgr.check_price_deviation(symbol="BTC-EUR", current_price=50100.0)
        assert len(cancelled) == 0
        assert len(mgr.pending_orders) == 1

    def test_mark_filled(self):
        mgr = LimitOrderManager(client=MagicMock())
        order = mgr.create_pending(
            symbol="BTC-EUR", direction="LONG", limit_price=50000.0,
            size_eur=100.0, stop_loss=48000.0, take_profit=53000.0, strategy="orderflow")
        mgr.mark_filled(order.order_id, fill_price=50001.0)
        assert order.status == "filled"
        assert len(mgr.pending_orders) == 0
        assert len(mgr.filled_orders) == 1

    def test_partial_fill(self):
        mgr = LimitOrderManager(client=MagicMock())
        order = mgr.create_pending(
            symbol="BTC-EUR", direction="LONG", limit_price=50000.0,
            size_eur=100.0, stop_loss=48000.0, take_profit=53000.0, strategy="orderflow")
        result = mgr.mark_partial_fill(order.order_id, filled_eur=60.0, fill_price=50001.0)
        assert result is not None
        assert result.filled_eur == 60.0
        assert order.size_eur == 40.0
        assert order.status == "pending"
        assert len(mgr.filled_orders) == 1
