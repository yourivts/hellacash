"""Tests for bot.exchange.order_manager.OrderManager."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.exchange.order_manager import OrderManager


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.paper_trading = True
    return client


@pytest.fixture
def mgr(mock_client):
    return OrderManager(mock_client)


class TestSubmitBuy:
    @pytest.mark.asyncio
    async def test_submit_buy_success(self, mgr, mock_client):
        mock_client.place_order.return_value = {
            "orderId": "abc123", "market": "BTC-EUR", "side": "buy",
            "orderType": "market", "status": "filled", "price": "50000",
            "amount": "0.1", "filledAmount": "0.1", "feePaid": "1.25",
            "feeCurrency": "EUR", "paper": True,
        }
        with patch("bot.exchange.order_manager.get_session") as mock_gs, \
             patch("bot.exchange.order_manager.save_order") as mock_save:
            mock_session = AsyncMock()
            mock_gs.return_value = AsyncMock()
            mock_gs.return_value.__aenter__.return_value = mock_session
            mock_gs.return_value.__aexit__.return_value = False
            mock_order = MagicMock()
            mock_order.id = 42
            mock_save.return_value = mock_order
            result = await mgr.submit_buy("BTC-EUR", 0.1, "hybrid")
            assert result == 42

    @pytest.mark.asyncio
    async def test_submit_buy_failure_returns_none(self, mgr, mock_client):
        mock_client.place_order.side_effect = Exception("API error")
        result = await mgr.submit_buy("BTC-EUR", 0.1, "hybrid")
        assert result is None


class TestSubmitSell:
    @pytest.mark.asyncio
    async def test_submit_sell_success(self, mgr, mock_client):
        mock_client.place_order.return_value = {
            "orderId": "def456", "market": "BTC-EUR", "side": "sell",
            "orderType": "market", "status": "filled", "price": "51000",
            "amount": "0.1", "filledAmount": "0.1", "feePaid": "1.28",
            "feeCurrency": "EUR",
        }
        with patch("bot.exchange.order_manager.get_session") as mock_gs, \
             patch("bot.exchange.order_manager.save_order") as mock_save:
            mock_session = AsyncMock()
            mock_gs.return_value = AsyncMock()
            mock_gs.return_value.__aenter__.return_value = mock_session
            mock_gs.return_value.__aexit__.return_value = False
            mock_order = MagicMock()
            mock_order.id = 43
            mock_save.return_value = mock_order
            result = await mgr.submit_sell("BTC-EUR", 0.1, "hybrid")
            assert result == 43

    @pytest.mark.asyncio
    async def test_submit_sell_failure_returns_none(self, mgr, mock_client):
        mock_client.place_order.side_effect = Exception("API error")
        result = await mgr.submit_sell("BTC-EUR", 0.1, "hybrid")
        assert result is None
