# tests/test_csv_export.py
"""Tests for trade CSV export endpoint."""
from __future__ import annotations

import csv
import io
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


def _mock_trade(**overrides):
    t = MagicMock()
    t.id = overrides.get("id", 1)
    t.symbol = overrides.get("symbol", "BTC-EUR")
    t.strategy_name = overrides.get("strategy_name", "hybrid")
    t.direction = overrides.get("direction", "LONG")
    t.entry_price = overrides.get("entry_price", 50000.0)
    t.exit_price = overrides.get("exit_price", 51000.0)
    t.quantity = overrides.get("quantity", 0.1)
    t.gross_pnl = overrides.get("gross_pnl", 100.0)
    t.net_pnl = overrides.get("net_pnl", 95.0)
    t.roi_pct = overrides.get("roi_pct", 1.9)
    t.hold_seconds = overrides.get("hold_seconds", 3600)
    t.exit_reason = overrides.get("exit_reason", "take_profit")
    t.paper_trade = overrides.get("paper_trade", True)
    t.created_at = MagicMock()
    t.created_at.isoformat.return_value = "2025-06-01T12:00:00"
    t.created_at.date.return_value = MagicMock()  # For date filtering
    return t


def _make_app():
    from bot.config import Settings
    settings = Settings(
        bitvavo_api_key="", bitvavo_api_secret="",
        database_url="postgresql+asyncpg://x:x@localhost/x",
        api_key="",
    )
    with patch("bot.config.get_settings", return_value=settings), \
         patch("api.middleware.auth.get_settings", return_value=settings), \
         patch("api.app.get_hub") as mock_hub:
        mock_hub.return_value.start_relay = AsyncMock()
        from api.app import create_app
        app = create_app()
    return app, settings


class TestCSVExport:
    @patch("api.routers.trades.get_session")
    @patch("api.routers.trades.get_trades")
    def test_tax_format(self, mock_get_trades, mock_session):
        mock_session.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
        mock_session.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_get_trades.return_value = [_mock_trade()]

        app, settings = _make_app()
        client = TestClient(app)
        with patch("api.middleware.auth.get_settings", return_value=settings):
            resp = client.get("/api/trades/export?format=tax")
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]
        assert "attachment" in resp.headers.get("content-disposition", "")

        reader = csv.reader(io.StringIO(resp.text))
        headers = next(reader)
        assert "date" in headers
        assert "fees" in headers
        assert "realized_pnl" in headers
        assert "cost_basis" in headers

    @patch("api.routers.trades.get_session")
    @patch("api.routers.trades.get_trades")
    def test_full_format(self, mock_get_trades, mock_session):
        mock_session.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
        mock_session.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_get_trades.return_value = [_mock_trade()]

        app, settings = _make_app()
        client = TestClient(app)
        with patch("api.middleware.auth.get_settings", return_value=settings):
            resp = client.get("/api/trades/export?format=full")
        assert resp.status_code == 200

        reader = csv.reader(io.StringIO(resp.text))
        headers = next(reader)
        assert "id" in headers
        assert "roi_pct" in headers
