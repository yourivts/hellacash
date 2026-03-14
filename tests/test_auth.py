# tests/test_auth.py
"""Tests for API key authentication middleware."""
from __future__ import annotations

import importlib
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


def _make_app(api_key: str = "test-key-123"):
    """Create app with a specific API key configured."""
    from bot.config import Settings
    settings = Settings(
        bitvavo_api_key="", bitvavo_api_secret="",
        database_url="postgresql+asyncpg://x:x@localhost/x",
        api_key=api_key,
    )

    mock_hub = MagicMock()
    mock_hub.start_relay = AsyncMock()
    mock_hub.connect = AsyncMock()
    mock_hub.disconnect = AsyncMock()

    # Patch get_settings at every site that reads it, then reload modules so
    # the fresh settings object is captured at import time.
    with patch("bot.config.get_settings", return_value=settings), \
         patch("api.middleware.auth.get_settings", return_value=settings), \
         patch("api.app.get_hub", return_value=mock_hub):
        # Reload auth first so it captures the patched get_settings, then app.
        import api.middleware.auth as auth_module
        importlib.reload(auth_module)
        import api.app as app_module
        importlib.reload(app_module)
        app = app_module.create_app()

    return app, settings


class TestRESTAuth:
    def test_health_exempt_from_auth(self):
        app, _ = _make_app()
        client = TestClient(app)
        resp = client.get("/api/health")
        # health is exempt — should not be 401
        assert resp.status_code == 200

    def test_wrong_key_returns_401(self):
        app, _ = _make_app()
        client = TestClient(app)
        resp = client.get("/api/trades", headers={"X-API-Key": "wrong"})
        assert resp.status_code == 401

    def test_missing_key_returns_401(self):
        app, _ = _make_app()
        client = TestClient(app)
        resp = client.get("/api/trades")
        assert resp.status_code == 401

    def test_correct_key_passes_auth(self):
        app, _ = _make_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/trades", headers={"X-API-Key": "test-key-123"})
        # May be 200 or 500 (no DB), but NOT 401
        assert resp.status_code != 401

    def test_no_auth_when_key_empty(self):
        app, settings = _make_app(api_key="")
        assert settings.api_key == ""
        # Patch auth at the point require_api_key calls get_settings during the request
        with patch("api.middleware.auth.get_settings", return_value=settings):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/api/trades")
        # Should not be 401 when no key is configured
        assert resp.status_code != 401


class TestWSAuth:
    def test_ws_rejected_without_key(self):
        app, _ = _make_app()
        client = TestClient(app, raise_server_exceptions=False)
        # ws_auth closes the socket with code 4001 — TestClient raises on disconnect
        try:
            with client.websocket_connect("/ws/feed") as ws:
                # If we get here the socket should be closed by server immediately
                data = ws.receive_bytes()  # should raise or return close frame
        except Exception:
            pass  # any exception means server rejected as expected
        else:
            # If no exception, ensure we didn't get a normal connection
            pytest.fail("Expected WebSocket to be rejected but it was accepted normally")

    def test_ws_accepted_with_correct_key(self):
        app, _ = _make_app()
        client = TestClient(app, raise_server_exceptions=False)
        with client.websocket_connect("/ws/feed?api_key=test-key-123") as ws:
            pass  # connected successfully
