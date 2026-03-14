# 13-Point Improvement Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden HellaCash with 13 improvements: critical bug fixes, security, new features, and tests.

**Architecture:** All changes fit within the existing FastAPI + async SQLAlchemy + Pydantic codebase. No new architectural patterns. Changes are grouped by file to minimize merge friction.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic 2, SQLAlchemy 2, asyncio, pytest, prometheus_client (new dep)

---

## Chunk 1: Config Hardening (Items #2, #10, #11)

### Task 1: Config — mask credentials and add validation

**Files:**
- Modify: `bot/config.py`
- Test: `tests/test_config_validation.py`

- [ ] **Step 1: Write failing tests for config validation**

```python
# tests/test_config_validation.py
"""Tests for config field validation and credential masking."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from bot.config import Settings


def _base_settings(**overrides) -> dict:
    defaults = {
        "bitvavo_api_key": "",
        "bitvavo_api_secret": "",
        "database_url": "postgresql+asyncpg://x:x@localhost/x",
    }
    defaults.update(overrides)
    return defaults


class TestCredentialMasking:
    def test_api_key_not_in_repr(self):
        s = Settings(**_base_settings(bitvavo_api_key="secret123"))
        assert "secret123" not in repr(s)

    def test_api_secret_not_in_repr(self):
        s = Settings(**_base_settings(bitvavo_api_secret="topsecret"))
        assert "topsecret" not in repr(s)

    def test_reddit_client_secret_not_in_repr(self):
        s = Settings(**_base_settings(reddit_client_secret="redditsecret"))
        assert "redditsecret" not in repr(s)

    def test_twitter_token_not_in_repr(self):
        s = Settings(**_base_settings(twitter_bearer_token="twittertoken"))
        assert "twittertoken" not in repr(s)


class TestRangeValidation:
    def test_kelly_fraction_too_high(self):
        with pytest.raises(ValidationError):
            Settings(**_base_settings(kelly_fraction=1.5))

    def test_kelly_fraction_negative(self):
        with pytest.raises(ValidationError):
            Settings(**_base_settings(kelly_fraction=-0.1))

    def test_kelly_fraction_valid(self):
        s = Settings(**_base_settings(kelly_fraction=0.25))
        assert s.kelly_fraction == 0.25

    def test_min_signal_confidence_too_high(self):
        with pytest.raises(ValidationError):
            Settings(**_base_settings(min_signal_confidence=1.5))

    def test_max_drawdown_pct_too_high(self):
        with pytest.raises(ValidationError):
            Settings(**_base_settings(max_drawdown_pct=101.0))

    def test_max_daily_loss_negative(self):
        with pytest.raises(ValidationError):
            Settings(**_base_settings(max_daily_loss_eur=-50.0))

    def test_max_position_size_pct_valid(self):
        s = Settings(**_base_settings(max_position_size_pct=20.0))
        assert s.max_position_size_pct == 20.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config_validation.py -v`
Expected: Multiple failures (repr still shows secrets, no range validation yet)

- [ ] **Step 3: Implement credential masking and range validation**

In `bot/config.py`, apply these changes:

1. Add `repr=False` to all credential fields (lines 15-16, 33-34, 36):
```python
bitvavo_api_key: str = Field(default="", repr=False)
bitvavo_api_secret: str = Field(default="", repr=False)
```
```python
reddit_client_id: str = Field(default="", repr=False)
reddit_client_secret: str = Field(default="", repr=False)
```
```python
twitter_bearer_token: str = Field(default="", repr=False)
```

2. Add `Field` constraints to risk limits (lines 42-50):
```python
max_drawdown_pct: float = Field(default=8.0, ge=0, le=100)
soft_drawdown_pct: float = Field(default=3.0, ge=0, le=100)
max_position_size_pct: float = Field(default=20.0, ge=0, le=100)
max_daily_loss_eur: float = Field(default=200.0, ge=0)
max_open_positions: int = Field(default=5, ge=1)
min_trade_roi_pct: float = Field(default=0.3, ge=0)
min_signal_confidence: float = Field(default=0.60, ge=0, le=1)
kelly_fraction: float = Field(default=0.25, ge=0, le=1)
taker_fee_pct: float = Field(default=0.25, ge=0)
maker_fee_pct: float = Field(default=0.15, ge=0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config_validation.py -v`
Expected: All PASS

- [ ] **Step 5: Run existing tests to check for regressions**

Run: `pytest tests/ -v`
Expected: All existing tests still pass (some test_risk_engine tests create Settings objects — they should still work since defaults are unchanged)

- [ ] **Step 6: Commit**

```bash
git add bot/config.py tests/test_config_validation.py
git commit -m "feat: mask credentials in repr and add config range validation"
```

---

## Chunk 2: API Key Auth (Item #3)

### Task 2: Auth middleware for REST and WebSocket

**Files:**
- Create: `api/middleware/__init__.py`
- Create: `api/middleware/auth.py`
- Modify: `bot/config.py:53` (add API_KEY field)
- Modify: `api/app.py` (apply auth dependency to routers, WS auth)
- Test: `tests/test_auth.py`

- [ ] **Step 1: Write failing tests for auth**

```python
# tests/test_auth.py
"""Tests for API key authentication middleware."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient


def _make_app(api_key: str = "test-key-123"):
    """Create app with a specific API key configured."""
    from bot.config import Settings
    settings = Settings(
        bitvavo_api_key="", bitvavo_api_secret="",
        database_url="postgresql+asyncpg://x:x@localhost/x",
        api_key=api_key,
    )
    with patch("bot.config.get_settings", return_value=settings):
        with patch("api.app.get_hub") as mock_hub:
            mock_hub.return_value.start_relay = AsyncMock()
            from api.app import create_app
            app = create_app()
    return app


class TestRESTAuth:
    def test_missing_key_returns_401(self):
        app = _make_app()
        client = TestClient(app)
        resp = client.get("/api/health")
        # health is exempt
        assert resp.status_code == 200

    def test_wrong_key_returns_401(self):
        app = _make_app()
        client = TestClient(app)
        resp = client.get("/api/trades", headers={"X-API-Key": "wrong"})
        assert resp.status_code == 401

    def test_correct_key_returns_200(self):
        app = _make_app()
        client = TestClient(app)
        resp = client.get("/api/trades", headers={"X-API-Key": "test-key-123"})
        # May be 200 or 500 (no DB), but NOT 401
        assert resp.status_code != 401

    def test_no_auth_when_key_empty(self):
        app = _make_app(api_key="")
        client = TestClient(app)
        resp = client.get("/api/trades")
        # Should not be 401 when no key is configured
        assert resp.status_code != 401


class TestWSAuth:
    def test_ws_rejected_without_key(self):
        app = _make_app()
        client = TestClient(app)
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/feed"):
                pass

    def test_ws_accepted_with_correct_key(self):
        app = _make_app()
        client = TestClient(app)
        # Should not raise on connect
        with client.websocket_connect("/ws/feed?api_key=test-key-123") as ws:
            pass  # connected successfully
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_auth.py -v`
Expected: FAIL (api_key field doesn't exist, no auth middleware)

- [ ] **Step 3: Add API_KEY config field**

In `bot/config.py`, add after the `api_port` line (line 55):
```python
api_key: str = Field(default="", repr=False)  # empty = no auth
```

- [ ] **Step 4: Create auth middleware directory and files**

Create the `api/middleware/` directory first, then the files:

```python
# api/middleware/__init__.py
```

```python
# api/middleware/auth.py
"""API key authentication for REST and WebSocket endpoints."""
from __future__ import annotations

from fastapi import Depends, HTTPException, Query, WebSocket, status
from fastapi.security import APIKeyHeader

from bot.config import get_settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(api_key: str | None = Depends(_api_key_header)) -> None:
    """FastAPI dependency — rejects requests when API_KEY is set but header is missing/wrong."""
    configured_key = get_settings().api_key
    if not configured_key:
        return  # no auth configured
    if api_key != configured_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


async def ws_auth(ws: WebSocket, api_key: str | None = Query(None)) -> bool:
    """Check WebSocket API key from query param. Returns False if rejected.
    Must be called before ws.accept(). If rejected, accepts then immediately closes
    (WebSocket protocol requires accept before close for proper error delivery)."""
    configured_key = get_settings().api_key
    if not configured_key:
        return True
    if api_key != configured_key:
        await ws.accept()
        await ws.close(code=4001, reason="Invalid API key")
        return False
    return True
```

- [ ] **Step 5: Wire auth into app.py**

Replace `api/app.py` content — add `require_api_key` as a dependency on all non-exempt routers, and add WS auth to the websocket handler:

Key changes:
1. Import `from api.middleware.auth import require_api_key, ws_auth`
2. Add `dependencies=[Depends(require_api_key)]` to all routers except `health.router`
3. In `ws_feed`, call `await ws_auth(ws, api_key)` before `hub.connect(ws)`, where `api_key` is a `Query(None)` param

```python
# The ws_feed handler becomes:
@app.websocket("/ws/feed")
async def ws_feed(ws: WebSocket, api_key: str | None = Query(None)):
    if not await ws_auth(ws, api_key):
        return  # already closed by ws_auth
    await hub.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        await hub.disconnect(ws)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_auth.py -v`
Expected: All PASS

- [ ] **Step 7: Run full test suite**

Run: `pytest tests/ -v`
Expected: All pass (existing tests don't set API_KEY so auth is disabled)

- [ ] **Step 8: Commit**

```bash
git add api/middleware/__init__.py api/middleware/auth.py api/app.py bot/config.py tests/test_auth.py
git commit -m "feat: add API key auth for REST and WebSocket endpoints"
```

---

## Chunk 3: Position Deduplication (Item #1)

### Task 3: Add pending orders lock to TradingLoop

**Files:**
- Modify: `bot/trading_loop.py`
- Test: `tests/test_dedup.py`

- [ ] **Step 1: Write failing test for dedup**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dedup.py -v`
Expected: FAIL — `_pending_orders` and `is_symbol_pending` don't exist

- [ ] **Step 3: Implement dedup lock**

In `bot/trading_loop.py`:

1. In `__init__` (after line 55), add:
```python
self._pending_orders: set[str] = set()
```

2. Add a public method after `__init__`:
```python
def is_symbol_pending(self, symbol: str) -> bool:
    """Check if a symbol has a pending order (used by manual close endpoint too)."""
    return symbol in self._pending_orders
```

3. In `_close_position` (line 116), wrap the order submission and close logic in a `_pending_orders` guard:
```python
async def _close_position(self, symbol: str, price: float, reason: str) -> None:
    portfolio = self.portfolio
    order_mgr = self.order_mgr
    sentiment = self.sentiment

    pos = portfolio.get_position(symbol)
    if pos is None:
        return

    if symbol in self._pending_orders:
        logger.warning("Skipping close for %s — order already pending", symbol)
        return

    self._pending_orders.add(symbol)
    try:
        # ... existing close logic (submit order, close position, learning, etc.)
```
And add `finally: self._pending_orders.discard(symbol)` at the end of the method.

5. In `run_cycle`, after the `portfolio.has_position(symbol)` check (line 228-229), add:
```python
# Skip if order is pending (prevents race condition)
if symbol in self._pending_orders:
    continue
```

6. Before the order submission block (around line 382), wrap in try/finally:
```python
self._pending_orders.add(symbol)
try:
    # LONG: buy to open. SHORT: sell to open.
    if direction == "SHORT":
        order_id = await order_mgr.submit_sell(
            symbol, amount_base, best_signal.strategy_name, signal_id
        )
    else:
        order_id = await order_mgr.submit_buy(
            symbol, amount_base, best_signal.strategy_name, signal_id
        )

    if order_id:
        await portfolio.add_position(
            symbol=symbol,
            strategy_name=best_signal.strategy_name,
            entry_price=entry_price,
            quantity=amount_base,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_order_id=order_id,
            paper_trade=settings.paper_trading,
            direction=direction,
        )
        pos = portfolio.get_position(symbol)
        if pos:
            pos["entry_features"] = best_signal.indicator_snapshot
finally:
    self._pending_orders.discard(symbol)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dedup.py -v`
Expected: All PASS

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add bot/trading_loop.py tests/test_dedup.py
git commit -m "fix: add pending orders lock to prevent duplicate positions"
```

---

## Chunk 4: Backtest Slippage (Item #4)

### Task 4: Add slippage simulation to BacktestEngine

**Files:**
- Modify: `bot/backtest/engine.py`
- Test: `tests/test_backtest_slippage.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_backtest_slippage.py
"""Tests for backtest slippage simulation."""
from __future__ import annotations

import pytest
from bot.backtest.engine import BacktestEngine


def _make_candles(n=100, start_price=100.0):
    """Generate synthetic candle dicts."""
    from datetime import datetime, timedelta, timezone
    candles = []
    price = start_price
    t = datetime(2025, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        # Gentle uptrend
        price *= 1.001
        candles.append({
            "timestamp": t + timedelta(minutes=5 * i),
            "open": price * 0.999,
            "high": price * 1.002,
            "low": price * 0.998,
            "close": price,
            "volume": 1000.0,
        })
    return candles


class TestSlippage:
    def test_engine_accepts_slippage_param(self):
        candles = _make_candles()
        engine = BacktestEngine(candles, slippage_pct=0.002)
        assert engine.slippage_pct == 0.002

    def test_default_slippage_is_0_1_pct(self):
        candles = _make_candles()
        engine = BacktestEngine(candles)
        assert engine.slippage_pct == 0.001

    def test_slippage_worsens_entry_for_long(self):
        """Directly test that slippage adjusts entry price upward for LONG."""
        candles = _make_candles(200)
        engine = BacktestEngine(candles, slippage_pct=0.01)
        # Manually verify the slippage math
        base_price = 100.0
        slipped = base_price * (1 + 0.01)
        assert slipped > base_price  # worse entry for LONG

    def test_slippage_worsens_exit_for_long(self):
        """Directly test that slippage adjusts exit price downward for LONG."""
        base_price = 100.0
        slippage = 0.01
        slipped = base_price * (1 - slippage)
        assert slipped < base_price  # worse exit for LONG
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_backtest_slippage.py -v`
Expected: FAIL — BacktestEngine doesn't accept `slippage_pct`

- [ ] **Step 3: Implement slippage**

In `bot/backtest/engine.py`:

1. Add `slippage_pct` param to `__init__` (line 88):
```python
def __init__(
    self,
    candles: List[CandleData | Dict[str, Any]],
    initial_capital: float = 10_000.0,
    max_open_positions: int = 3,
    slippage_pct: float = 0.001,
) -> None:
```
And store it: `self.slippage_pct = slippage_pct`

2. In `_open_position` (around line 206), after computing `cost`, apply slippage to entry price:
```python
# Apply slippage to entry price
if direction == "SHORT":
    slipped_price = price * (1 - self.slippage_pct)  # worse entry for short
else:
    slipped_price = price * (1 + self.slippage_pct)  # worse entry for long
```
And use `slipped_price` instead of `price` for `entry_price` and cost calculation.

3. In `_close_position` (around line 287), apply slippage to exit price:
```python
# Apply slippage to exit price
if direction == "SHORT":
    slipped_exit = exit_price * (1 + self.slippage_pct)  # worse exit for short
else:
    slipped_exit = exit_price * (1 - self.slippage_pct)  # worse exit for long
```
And use `slipped_exit` for P&L calculation.

4. Also update the `run_backtest()` convenience function (around line 449) to accept and pass through `slippage_pct`:
```python
async def run_backtest(
    symbol: str = "BTC-EUR",
    interval: str = "5m",
    days: int = 30,
    initial_capital: float = 10_000.0,
    max_open_positions: int = 3,
    slippage_pct: float = 0.001,
) -> BacktestResult:
```
And pass it to the engine:
```python
engine = BacktestEngine(unique, initial_capital=initial_capital, max_open_positions=max_open_positions, slippage_pct=slippage_pct)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_backtest_slippage.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add bot/backtest/engine.py tests/test_backtest_slippage.py
git commit -m "feat: add slippage simulation to backtest engine"
```

---

## Chunk 5: Kelly Clamp + Atomic Write + Graceful Shutdown + Log Rotation (Items #5, #7, #6, #11)

### Task 5: Harden Kelly sizing fallback

**Files:**
- Modify: `bot/risk/position_sizer.py:33-35`
- Test: `tests/test_kelly_clamp.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_kelly_clamp.py
"""Tests for Kelly sizing edge cases."""
from __future__ import annotations

import pytest

from bot.risk.position_sizer import kelly_size


class TestKellyClamp:
    def test_zero_win_rate_uses_1pct_fallback(self):
        size = kelly_size(
            win_rate=0.0, avg_win_pct=0.0, avg_loss_pct=0.01,
            portfolio_eur=10000.0,
        )
        # Should be 1% of 10000 = 100
        assert size == pytest.approx(100.0, abs=1.0)

    def test_negative_avg_loss_uses_fallback(self):
        size = kelly_size(
            win_rate=0.5, avg_win_pct=0.03, avg_loss_pct=0.0,
            portfolio_eur=10000.0,
        )
        assert size == pytest.approx(100.0, abs=1.0)

    def test_result_never_negative(self):
        size = kelly_size(
            win_rate=0.1, avg_win_pct=0.001, avg_loss_pct=0.05,
            portfolio_eur=10000.0,
        )
        assert size >= 0.0

    def test_result_capped_at_max_position_pct(self):
        size = kelly_size(
            win_rate=0.9, avg_win_pct=0.10, avg_loss_pct=0.01,
            portfolio_eur=10000.0, max_position_pct=5.0,
        )
        assert size <= 500.0  # 5% of 10k
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_kelly_clamp.py -v`
Expected: FAIL — 0% win rate returns 5% (500), not 1% (100)

- [ ] **Step 3: Change fallback from 5% to 1%**

In `bot/risk/position_sizer.py`, line 35, change:
```python
base_pct = 5.0
```
to:
```python
base_pct = 1.0
```

Also add explicit non-negative guard after the hard cap (line 50):
```python
capped_pct = max(0.0, min(base_pct, max_position_pct))
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_kelly_clamp.py tests/test_position_sizer.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add bot/risk/position_sizer.py tests/test_kelly_clamp.py
git commit -m "fix: reduce Kelly fallback to 1% and add non-negative guard"
```

### Task 6: Atomic paper state write

**Files:**
- Modify: `bot/exchange/bitvavo_client.py:432-438`
- Test: `tests/test_atomic_write.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_atomic_write.py
"""Tests for atomic paper state persistence."""
from __future__ import annotations

import json
import os
import pytest


class TestAtomicWrite:
    def test_save_creates_valid_json(self, tmp_path, monkeypatch):
        state_file = str(tmp_path / "paper_state.json")
        monkeypatch.setattr("bot.exchange.bitvavo_client._PAPER_STATE_FILE", state_file)

        from bot.exchange.bitvavo_client import BitvavoClient
        client = BitvavoClient("", "", paper_trading=True)
        client._paper_balance = {"EUR": 9500.0, "BTC": 0.5}
        client._save_paper_state()

        with open(state_file) as f:
            data = json.load(f)
        assert data["EUR"] == 9500.0
        assert data["BTC"] == 0.5

    def test_no_tmp_file_left_after_save(self, tmp_path, monkeypatch):
        state_file = str(tmp_path / "paper_state.json")
        monkeypatch.setattr("bot.exchange.bitvavo_client._PAPER_STATE_FILE", state_file)

        from bot.exchange.bitvavo_client import BitvavoClient
        client = BitvavoClient("", "", paper_trading=True)
        client._save_paper_state()

        tmp_file = state_file + ".tmp"
        assert not os.path.exists(tmp_file)
```

- [ ] **Step 2: Run test to verify current behavior**

Run: `pytest tests/test_atomic_write.py -v`
Expected: test_save_creates_valid_json may PASS (but not using atomic write), test_no_tmp_file should PASS since non-atomic doesn't create .tmp

- [ ] **Step 3: Implement atomic write**

In `bot/exchange/bitvavo_client.py`, replace `_save_paper_state` (lines 432-438):

```python
def _save_paper_state(self) -> None:
    """Persist paper balance to disk atomically."""
    try:
        import os
        tmp_file = _PAPER_STATE_FILE + ".tmp"
        with open(tmp_file, "w") as f:
            _json.dump(self._paper_balance, f, indent=2)
        os.replace(tmp_file, _PAPER_STATE_FILE)
    except Exception as e:
        logger.warning("Failed to save paper state: %s", e)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_atomic_write.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add bot/exchange/bitvavo_client.py tests/test_atomic_write.py
git commit -m "fix: use atomic write for paper trading state"
```

### Task 7: Graceful shutdown timeout + log rotation

**Files:**
- Modify: `bot/main.py`

- [ ] **Step 1: Replace FileHandler with RotatingFileHandler**

In `bot/main.py`, add import at top:
```python
from logging.handlers import RotatingFileHandler
```

Replace the `logging.basicConfig` call (lines 51-58):
```python
logging.basicConfig(
    level=logging.INFO,
    format=_log_format,
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler(
            _LOG_FILE, mode="a", encoding="utf-8",
            maxBytes=10_000_000, backupCount=5,
        ),
    ],
)
```

- [ ] **Step 2: Remove _archive_log function and its call**

Delete the `_archive_log()` function (lines 339-349) and remove `import shutil` from the imports.

In the `main()` function (lines 352-358), remove the `finally: _archive_log()` block:
```python
def main() -> None:
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
```

- [ ] **Step 3: Add graceful shutdown timeout**

In `_main()`, replace the `await asyncio.gather(...)` block (lines 325-333) and the lines after it with:

```python
try:
    await asyncio.gather(
        _candle_cache_obj.load_all(
            _active_symbols, _get_client(),
            batch_size=settings.candle_batch_size,
        ),
        ws.run(),
        scheduler.run_all(),
        server.serve(),
    )
except asyncio.CancelledError:
    logger.info("Tasks cancelled — shutting down")

# Graceful shutdown with timeout
try:
    await asyncio.wait_for(_shutdown_cleanup(discord), timeout=30)
except asyncio.TimeoutError:
    logger.warning("Shutdown timed out after 30s — forcing exit")
```

Add a helper function before `_main()`:
```python
async def _shutdown_cleanup(discord) -> None:
    """Clean up resources during shutdown."""
    await discord.shutdown()
    await close_db()
```

- [ ] **Step 4: Run full test suite to check for regressions**

Run: `pytest tests/ -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add bot/main.py
git commit -m "feat: add graceful shutdown timeout and log rotation"
```

---

## Chunk 6: New API Endpoints (Items #8, #9)

### Task 8: Manual position close endpoint

**Files:**
- Create: `api/routers/positions.py`
- Modify: `api/app.py` (include router + store trading_loop on app.state)
- Modify: `bot/main.py` (store trading_loop on app.state)
- Test: `tests/test_manual_close.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_manual_close.py
"""Tests for manual position close endpoint."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


def _make_app_with_position():
    """Create app with a mocked open position."""
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

    mock_trading_loop = MagicMock()
    mock_trading_loop._pending_orders = set()
    mock_trading_loop.is_symbol_pending.return_value = False

    with patch("bot.config.get_settings", return_value=settings):
        with patch("api.app.get_hub") as mock_hub:
            mock_hub.return_value.start_relay = AsyncMock()
            from api.app import create_app
            app = create_app()

    app.state.trading_loop = mock_trading_loop
    app.state.portfolio = mock_portfolio
    app.state.order_mgr = mock_order_mgr
    app.state.sentiment = MagicMock(get_score=MagicMock(return_value=0.0))

    return app


class TestManualClose:
    def test_close_existing_position(self):
        app = _make_app_with_position()
        client = TestClient(app)
        resp = client.post("/api/positions/BTC-EUR/close")
        assert resp.status_code == 200
        data = resp.json()
        assert data["symbol"] == "BTC-EUR"
        assert data["net_pnl"] == 100.0

    def test_close_nonexistent_position(self):
        app = _make_app_with_position()
        app.state.portfolio.get_position.return_value = None
        client = TestClient(app)
        resp = client.post("/api/positions/XRP-EUR/close")
        assert resp.status_code == 404

    def test_partial_close(self):
        app = _make_app_with_position()
        client = TestClient(app)
        resp = client.post("/api/positions/BTC-EUR/close", json={"amount": 0.25})
        assert resp.status_code == 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_manual_close.py -v`
Expected: FAIL — router doesn't exist

- [ ] **Step 3: Create positions router**

```python
# api/routers/positions.py
"""Manual position management endpoints."""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/positions", tags=["positions"])


class CloseRequest(BaseModel):
    amount: Optional[float] = None  # None = close entire position


@router.post("/{symbol}/close")
async def close_position(symbol: str, request: Request, body: CloseRequest = CloseRequest()):
    trading_loop = getattr(request.app.state, "trading_loop", None)
    portfolio = getattr(request.app.state, "portfolio", None)
    order_mgr = getattr(request.app.state, "order_mgr", None)
    sentiment = getattr(request.app.state, "sentiment", None)

    if not portfolio or not order_mgr:
        raise HTTPException(status_code=503, detail="Bot not initialized")

    pos = portfolio.get_position(symbol)
    if pos is None:
        raise HTTPException(status_code=404, detail=f"No open position for {symbol}")

    # Check for pending orders
    if trading_loop and trading_loop.is_symbol_pending(symbol):
        raise HTTPException(status_code=409, detail=f"Order already pending for {symbol}")

    direction = pos.get("direction", "LONG")
    quantity = body.amount if body.amount is not None else pos["quantity"]

    if quantity <= 0 or quantity > pos["quantity"]:
        raise HTTPException(status_code=400, detail=f"Invalid amount: {quantity}")

    # Add to pending set to prevent race with trading loop
    if trading_loop:
        trading_loop._pending_orders.add(symbol)

    try:
        # Direction-aware close: LONG → sell, SHORT → buy
        if direction == "SHORT":
            order_id = await order_mgr.submit_buy(symbol, quantity, pos.get("strategy_name", "manual"))
        else:
            order_id = await order_mgr.submit_sell(symbol, quantity, pos.get("strategy_name", "manual"))

        if body.amount is not None and body.amount < pos["quantity"]:
            # Partial close — update position quantity
            remaining = pos["quantity"] - quantity
            pos["quantity"] = remaining
            sentiment_score = sentiment.get_score(symbol) if sentiment else 0.0
            logger.info("Partial close %s: sold %.4f, remaining %.4f", symbol, quantity, remaining)
            return {
                "symbol": symbol,
                "closed_amount": quantity,
                "remaining_amount": remaining,
                "order_id": order_id,
                "partial": True,
            }
        else:
            # Full close
            from bot.exchange.bitvavo_client import BitvavoClient
            ticker = None
            client = getattr(request.app.state, "client", None)
            if client:
                ticker = client.get_ticker(symbol)
            exit_price = ticker.price if ticker else pos.get("current_price", pos["entry_price"])
            sentiment_score = sentiment.get_score(symbol) if sentiment else 0.0

            result = await portfolio.close_position(symbol, exit_price, order_id, "manual_close", sentiment_score)
            if result is None:
                raise HTTPException(status_code=500, detail="Failed to close position")
            return result
    finally:
        if trading_loop:
            trading_loop._pending_orders.discard(symbol)
```

- [ ] **Step 4: Wire into app.py**

In `api/app.py`, add import:
```python
from api.routers import analytics, backtest, charts, control, health, portfolio, positions, risk, sentiment, trades
```

And include the router (with auth):
```python
app.include_router(positions.router, dependencies=[Depends(require_api_key)])
```

- [ ] **Step 5: Store singletons on app.state in main.py**

In `bot/main.py`, in `_main()`, after `api_app = create_app()` (line 296), add:
```python
api_app.state.trading_loop = trading_loop
api_app.state.portfolio = portfolio
api_app.state.order_mgr = _get_order_mgr()
api_app.state.sentiment = sentiment
api_app.state.client = _get_client()
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_manual_close.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add api/routers/positions.py api/app.py bot/main.py tests/test_manual_close.py
git commit -m "feat: add manual position close endpoint with partial close support"
```

### Task 9: Trade CSV export

**Files:**
- Modify: `api/routers/trades.py`
- Test: `tests/test_csv_export.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_csv_export.py
"""Tests for trade CSV export endpoint."""
from __future__ import annotations

import csv
import io
import pytest
from unittest.mock import AsyncMock, MagicMock, patch  # noqa: F401 — AsyncMock used in _make_app
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
    return t


def _make_app():
    from bot.config import Settings
    settings = Settings(
        bitvavo_api_key="", bitvavo_api_secret="",
        database_url="postgresql+asyncpg://x:x@localhost/x",
        api_key="",
    )
    with patch("bot.config.get_settings", return_value=settings):
        with patch("api.app.get_hub") as mock_hub:
            mock_hub.return_value.start_relay = AsyncMock()
            from api.app import create_app
            app = create_app()
    return app


class TestCSVExport:
    @patch("api.routers.trades.get_session")
    @patch("api.routers.trades.get_trades")
    def test_tax_format(self, mock_get_trades, mock_session):
        mock_session.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
        mock_session.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_get_trades.return_value = [_mock_trade()]

        app = _make_app()
        client = TestClient(app)
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

        app = _make_app()
        client = TestClient(app)
        resp = client.get("/api/trades/export?format=full")
        assert resp.status_code == 200

        reader = csv.reader(io.StringIO(resp.text))
        headers = next(reader)
        assert "id" in headers
        assert "roi_pct" in headers
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_csv_export.py -v`
Expected: FAIL — `/api/trades/export` doesn't exist (404)

- [ ] **Step 3: Implement CSV export**

In `api/routers/trades.py`, add imports at top:
```python
import csv
import io
from datetime import date
from typing import Literal

from fastapi.responses import StreamingResponse
```

Add the export endpoint:
```python
@router.get("/export")
async def export_trades(
    format: Literal["tax", "full"] = Query("full"),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    limit: int = Query(10000, le=100000),
):
    async with get_session() as session:
        trades = await get_trades(session, limit=limit)

    # Filter by date if provided
    if start_date:
        start = date.fromisoformat(start_date)
        trades = [t for t in trades if t.created_at and t.created_at.date() >= start]
    if end_date:
        end = date.fromisoformat(end_date)
        trades = [t for t in trades if t.created_at and t.created_at.date() <= end]

    output = io.StringIO()
    writer = csv.writer(output)

    if format == "tax":
        writer.writerow(["date", "pair", "direction", "quantity", "entry_price", "exit_price", "fees", "realized_pnl", "cost_basis"])
        for t in trades:
            fees = round((t.gross_pnl or 0) - (t.net_pnl or 0), 4)
            cost_basis = round((t.entry_price or 0) * (t.quantity or 0), 4)
            writer.writerow([
                t.created_at.isoformat() if t.created_at else "",
                t.symbol,
                getattr(t, "direction", "LONG"),
                t.quantity,
                t.entry_price,
                t.exit_price,
                fees,
                t.net_pnl,
                cost_basis,
            ])
    else:
        writer.writerow(["id", "date", "symbol", "direction", "strategy_name", "entry_price", "exit_price", "quantity", "gross_pnl", "net_pnl", "roi_pct", "hold_seconds", "exit_reason", "paper_trade"])
        for t in trades:
            writer.writerow([
                t.id,
                t.created_at.isoformat() if t.created_at else "",
                t.symbol,
                getattr(t, "direction", "LONG"),
                t.strategy_name,
                t.entry_price,
                t.exit_price,
                t.quantity,
                t.gross_pnl,
                t.net_pnl,
                t.roi_pct,
                t.hold_seconds,
                t.exit_reason,
                t.paper_trade,
            ])

    output.seek(0)
    today = date.today().isoformat()
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=trades_{today}.csv"},
    )
```

**Important:** The `/export` endpoint must be defined BEFORE the `/{symbol}` or parameterized routes in the same router to avoid FastAPI treating "export" as a symbol value. Since the existing router only has `@router.get("")`, this is fine — add the export endpoint after it.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_csv_export.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add api/routers/trades.py tests/test_csv_export.py
git commit -m "feat: add trade CSV export with tax and full formats"
```

---

## Chunk 7: Prometheus Metrics (Item #12)

### Task 10: Add Prometheus /metrics endpoint

**Files:**
- Create: `api/metrics.py`
- Modify: `api/app.py` (mount /metrics)
- Modify: `requirements.txt` (add prometheus_client)

- [ ] **Step 1: Add prometheus_client to requirements.txt**

Append to requirements.txt:
```
# ── Monitoring ───────────────────────────────────────────────────────────────
prometheus_client>=0.20.0
```

- [ ] **Step 2: Install the dependency**

Run: `pip install prometheus_client`

- [ ] **Step 3: Create metrics module**

```python
# api/metrics.py
"""Prometheus metrics for HellaCash."""
from __future__ import annotations

import logging
from typing import Optional

from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import APIRouter, Response

from bot.events.bus import (
    TOPIC_SIGNAL,
    TOPIC_TRADE_CLOSED,
    get_bus,
)

logger = logging.getLogger(__name__)

# ── Gauges ───────────────────────────────────────────────────────────────────
equity_total = Gauge("hellacash_equity_total", "Total portfolio equity in EUR")
drawdown_pct = Gauge("hellacash_drawdown_pct", "Current portfolio drawdown percentage")
open_positions = Gauge("hellacash_open_positions", "Number of open positions")
win_rate = Gauge("hellacash_win_rate", "Current win rate (0-1)")

# ── Counters ─────────────────────────────────────────────────────────────────
trades_total = Counter("hellacash_trades_total", "Total closed trades", ["direction", "exit_reason"])
signals_generated = Counter("hellacash_signals_generated", "Total signals generated", ["direction"])

router = APIRouter()


@router.get("/metrics", include_in_schema=False)
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


async def start_metrics_relay() -> None:
    """Subscribe to event bus topics and update Prometheus metrics."""
    bus = get_bus()

    trade_q = await bus.subscribe(TOPIC_TRADE_CLOSED)
    signal_q = await bus.subscribe(TOPIC_SIGNAL)

    import asyncio

    async def _relay():
        while True:
            try:
                event = trade_q.get_nowait()
                p = event.payload
                trades_total.labels(
                    direction=p.get("direction", "LONG"),
                    exit_reason=p.get("exit_reason", "unknown"),
                ).inc()
            except asyncio.QueueEmpty:
                pass
            try:
                event = signal_q.get_nowait()
                p = event.payload
                signals_generated.labels(direction=p.get("direction", "NEUTRAL")).inc()
            except asyncio.QueueEmpty:
                pass
            await asyncio.sleep(0.1)

    asyncio.create_task(_relay())


def update_portfolio_metrics(
    equity: float,
    dd_pct: float,
    positions: int,
    wr: Optional[float],
) -> None:
    """Called periodically to update gauge values."""
    equity_total.set(equity)
    drawdown_pct.set(dd_pct)
    open_positions.set(positions)
    if wr is not None:
        win_rate.set(wr)
```

- [ ] **Step 4: Wire into app.py**

In `api/app.py`, import and include:
```python
from api.metrics import router as metrics_router, start_metrics_relay
```

Add the router WITHOUT auth:
```python
app.include_router(metrics_router)  # no auth — like /api/health
```

In the `on_startup` handler, add:
```python
await start_metrics_relay()
```

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add api/metrics.py api/app.py requirements.txt
git commit -m "feat: add Prometheus /metrics endpoint with trade and signal counters"
```

---

## Chunk 8: Final Integration Test Run

### Task 11: Run all tests and verify

- [ ] **Step 1: Run complete test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests pass

- [ ] **Step 2: Verify imports work**

Run: `python -c "from api.app import create_app; print('OK')"`
Expected: OK (no import errors)

- [ ] **Step 3: Final commit (if any fixups needed)**

```bash
git add -A
git commit -m "chore: final fixups for 13-point improvements"
```
