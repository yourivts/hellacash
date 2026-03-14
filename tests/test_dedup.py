# tests/test_dedup.py
"""Tests for position deduplication via _pending_orders set."""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from bot.trading_loop import TradingLoop


def _make_trading_loop():
    """Create a TradingLoop with mocked dependencies."""
    settings = MagicMock()
    settings.min_signal_confidence = 0.60
    settings.max_position_size_pct = 20.0
    settings.kelly_fraction = 0.25
    settings.paper_trading = True

    portfolio = MagicMock()
    portfolio.has_position.return_value = False
    portfolio.get_position.return_value = None
    portfolio.get_equity_eur.return_value = 10000.0
    portfolio.open_position_count.return_value = 0
    portfolio.add_position = AsyncMock()

    order_mgr = MagicMock()
    order_mgr.submit_buy = AsyncMock(return_value="order-123")

    drawdown = MagicMock()
    drawdown.is_trading_allowed.return_value = (True, None)
    drawdown.current_drawdown_pct.return_value = 0.0
    drawdown.position_size_multiplier.return_value = 1.0
    drawdown.daily_realized_loss_eur.return_value = 0.0

    loop = TradingLoop(
        candle_cache=MagicMock(),
        portfolio=portfolio,
        risk_engine=MagicMock(),
        order_mgr=order_mgr,
        sentiment=MagicMock(),
        router=MagicMock(),
        drawdown=drawdown,
        trade_analyzer=MagicMock(),
        signal_eval=MagicMock(),
        settings=settings,
    )
    return loop


class TestPendingOrders:
    def test_pending_orders_initialized(self):
        loop = _make_trading_loop()
        assert hasattr(loop, "_pending_orders")
        assert isinstance(loop._pending_orders, set)
        assert len(loop._pending_orders) == 0

    def test_is_symbol_pending(self):
        loop = _make_trading_loop()
        assert not loop.is_symbol_pending("BTC-EUR")
        loop._pending_orders.add("BTC-EUR")
        assert loop.is_symbol_pending("BTC-EUR")
