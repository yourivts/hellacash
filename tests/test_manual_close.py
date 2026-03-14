# tests/test_manual_close.py
"""Tests for manual position close endpoint."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


def _make_app_and_settings():
    """Create app with a mocked open position, return (app, settings)."""
    from bot.config import Settings
    settings = Settings(
        bitvavo_api_key="", bitvavo_api_secret="",
        database_url="postgresql+asyncpg://x:x@localhost/x",
        api_key="",
    )

    mock_portfolio = MagicMock()
    mock_portfolio.get_position.return_value = {
        "symbol": "BTC-EUR",
        "direction": "LONG",
        "quantity": 0.5,
        "entry_price": 50000.0,
        "current_price": 51000.0,
        "strategy_name": "hybrid",
    }
    mock_portfolio.close_position = AsyncMock(return_value={
        "trade_id": 1,
        "symbol": "BTC-EUR",
        "direction": "LONG",
        "net_pnl": 100.0,
        "roi_pct": 2.0,
        "exit_reason": "manual_close",
    })

    mock_order_mgr = MagicMock()
    mock_order_mgr.submit_sell = AsyncMock(return_value="order-456")
    mock_order_mgr.submit_buy = AsyncMock(return_value="order-789")

    mock_trading_loop = MagicMock()
    mock_trading_loop._pending_orders = set()
    mock_trading_loop.is_symbol_pending.return_value = False

    with patch("bot.config.get_settings", return_value=settings), \
         patch("api.middleware.auth.get_settings", return_value=settings), \
         patch("api.app.get_hub") as mock_hub:
        mock_hub.return_value.start_relay = AsyncMock()
        from api.app import create_app
        app = create_app()

    app.state.trading_loop = mock_trading_loop
    app.state.portfolio = mock_portfolio
    app.state.order_mgr = mock_order_mgr
    app.state.sentiment = MagicMock(get_score=MagicMock(return_value=0.0))

    return app, settings


class TestManualClose:
    def test_close_existing_position(self):
        app, settings = _make_app_and_settings()
        client = TestClient(app)
        with patch("api.middleware.auth.get_settings", return_value=settings):
            resp = client.post("/api/positions/BTC-EUR/close")
        assert resp.status_code == 200
        data = resp.json()
        assert data["symbol"] == "BTC-EUR"
        assert data["net_pnl"] == 100.0

    def test_close_nonexistent_position(self):
        app, settings = _make_app_and_settings()
        app.state.portfolio.get_position.return_value = None
        client = TestClient(app)
        with patch("api.middleware.auth.get_settings", return_value=settings):
            resp = client.post("/api/positions/XRP-EUR/close")
        assert resp.status_code == 404

    def test_partial_close(self):
        app, settings = _make_app_and_settings()
        # Give position an id so DB update path is exercised
        app.state.portfolio.get_position.return_value["id"] = 42
        client = TestClient(app)

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("api.middleware.auth.get_settings", return_value=settings), \
             patch("api.routers.positions.get_session", return_value=mock_session), \
             patch("api.routers.positions.save_trade", new_callable=AsyncMock) as mock_save, \
             patch("api.routers.positions.update_position", new_callable=AsyncMock) as mock_update:
            resp = client.post("/api/positions/BTC-EUR/close", json={"amount": 0.25})
        assert resp.status_code == 200
        data = resp.json()
        assert data["partial"] is True
        assert data["closed_amount"] == 0.25
        assert data["remaining_amount"] == 0.25
        # Verify DB persistence was called
        mock_save.assert_called_once()
        mock_update.assert_called_once()
