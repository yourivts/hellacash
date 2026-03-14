# HellaCash: 13-Point Improvement Spec

## Overview

Thirteen improvements to the HellaCash trading bot spanning critical bug fixes, security hardening, new features, and tests. (Original item 5 — sentiment weight rebalancing — was dropped as `_get_active_weights()` already implements proportional redistribution.)

## 1. Position Deduplication Lock (Critical)

**Problem:** Race condition in `trading_loop.py` — if an order fills instantly, a second buy signal for the same symbol can open a duplicate position before the portfolio updates.

**Solution:**
- Add `_pending_orders: set[str]` to `TradingLoop`
- Add symbol to `_pending_orders` before `submit_buy/sell`
- Remove after position is confirmed or order fails (in a `try/finally`)
- Skip signal processing for any symbol in `_pending_orders`

**Files:** `bot/trading_loop.py`

## 2. API Key Masking (Critical)

**Problem:** Credential fields in `config.py` are missing `repr=False`, so they could leak in logs.

**Solution:** Add `repr=False` to all credential fields: `bitvavo_api_key`, `bitvavo_api_secret`, `reddit_client_id`, `reddit_client_secret`, `twitter_bearer_token`.

**Files:** `bot/config.py`

## 3. Simple API Key Auth (Critical)

**Problem:** All REST and WebSocket endpoints are open to anyone.

**Solution:**
- Add `API_KEY` field to `config.py` (default empty = no auth for dev convenience)
- Create `api/middleware/auth.py` with a FastAPI dependency that checks `X-API-Key` header for REST routes
- For WebSocket `/ws/feed`: check `api_key` query parameter (browsers can't set custom headers on WS upgrade). Auth check must happen before `ws.accept()`.
- Apply auth dependency to all REST routers
- Exempt `/api/health` and `/metrics` for external monitoring

**Files:** `bot/config.py`, new `api/middleware/auth.py`, `api/app.py`, `api/ws_hub.py`

## 4. Backtest Slippage Simulation (High)

**Problem:** `BacktestEngine` fills at exact price, making results 0.3-2% more optimistic than live.

**Solution:**
- Add `slippage_pct: float = 0.001` param (0.1% base slippage)
- Buy: `fill_price = price * (1 + slippage_pct)`, Sell: `fill_price = price * (1 - slippage_pct)`
- Volume-based scaling deferred to a follow-up (backtest candles don't carry 24h volume reliably)

**Files:** `bot/backtest/engine.py`

## 5. Kelly Sizing Clamp (High)

**Problem:** Kelly fraction edge cases — the existing code clamps `full_kelly` to `[0, 1]` and uses a 5% fallback for 0% win rate, but the fallback is too aggressive and the clamp doesn't account for `max_position_size_pct`.

**Solution:**
- In `bot/risk/position_sizer.py`, change the 5% conservative fallback to 1% for more safety
- Ensure the final `base_pct` is clamped to `[0.0, max_position_pct]` (already partially done via `min()`, add explicit `max(0.0, ...)` guard)

**Files:** `bot/risk/position_sizer.py`

## 6. Graceful Shutdown Timeout (High)

**Problem:** `main.py` can hang forever if async tasks don't cancel.

**Solution:**
- Wrap shutdown sequence in `asyncio.wait_for(timeout=30)`
- If tasks don't cancel within 30s, force-cancel and log warning

**Files:** `bot/main.py`

## 7. Atomic Paper State Write (High)

**Problem:** Crash between in-memory update and `paper_state.json` write loses balance.

**Solution:**
- Write to `paper_state.tmp.json` first
- Use `os.replace()` to atomically swap (works on Windows and Linux)

**Files:** `bot/exchange/bitvavo_client.py`

## 8. Manual Position Close Endpoint (Medium)

**Problem:** No way to close a single position via API.

**Solution:**
- `POST /api/positions/{symbol}/close` with optional `amount` body param
- If `amount` omitted, close entire position
- If `amount` provided, close that fraction and update remaining
- Direction-aware (uses correct sell/buy for long/short)
- Must add symbol to `TradingLoop._pending_orders` before submitting close to avoid race with signal processing. Requires the `TradingLoop` instance to be accessible from the API layer (via `app.state`).

**Files:** new `api/routers/positions.py`, `api/app.py`

## 9. Trade CSV Export (Medium)

**Problem:** No export for tax reporting.

**Solution:**
- `GET /api/trades/export?format=tax|full&start_date=...&end_date=...`
- `format=tax`: date, pair, side, quantity, entry_price, exit_price, fees (computed: `gross_pnl - net_pnl`), realized_pnl (`net_pnl`), cost_basis (computed: `entry_price * quantity`)
- `format=full`: all Trade model fields as-is
- Returns `Content-Disposition: attachment; filename=trades_<date>.csv` with `StreamingResponse`

**Files:** `api/routers/trades.py`

## 10. Config Range Validation (Medium)

**Problem:** Invalid config values like `KELLY_FRACTION=1.5` silently work.

**Solution:**
- `Field(ge=0, le=1)` on `KELLY_FRACTION`, `MIN_SIGNAL_CONFIDENCE`
- `Field(ge=0, le=100)` on percentage fields (`MAX_DRAWDOWN_PCT`, `SOFT_DRAWDOWN_PCT`, `MAX_POSITION_SIZE_PCT`)
- `Field(ge=0)` on `MAX_DAILY_LOSS_EUR`
- App fails fast on startup with clear Pydantic error

**Files:** `bot/config.py`

## 11. Log Rotation (Medium)

**Problem:** Logs not rotated, could fill disk. Existing `_archive_log()` function manually moves logs on shutdown, which conflicts with `RotatingFileHandler`.

**Solution:**
- Replace `FileHandler` with `RotatingFileHandler` (`maxBytes=10_000_000`, `backupCount=5`)
- Remove the `_archive_log()` function entirely — `RotatingFileHandler` manages the full log lifecycle

**Files:** `bot/main.py`

## 12. Prometheus Metrics (Medium)

**Problem:** No `/metrics` endpoint for operational monitoring.

**Solution:**
- Add `prometheus_client` dependency to `requirements.txt`
- Create `api/metrics.py` that:
  - Defines gauges: `hellacash_equity_total`, `hellacash_drawdown_pct`, `hellacash_open_positions`, `hellacash_win_rate`
  - Defines counters: `hellacash_trades_total`, `hellacash_signals_generated`
  - Subscribes to event bus topics: `TOPIC_TRADE_CLOSED` → increment `trades_total`, `TOPIC_SIGNAL` → increment `signals_generated`
  - Updates gauges from portfolio tracker on each `TOPIC_PORTFOLIO_UPDATE` (or periodic poll)
- Add `/metrics` route on the existing FastAPI app (not a separate server)
- `/metrics` is exempt from auth (like `/api/health`)

**Files:** new `api/metrics.py`, `api/app.py`, `requirements.txt`

## 13. Targeted Tests

Split into separate test files for clarity:

- `tests/test_auth.py`: 401/200/401 for missing/correct/wrong API key (REST + WS)
- `tests/test_dedup.py`: `_pending_orders` prevents duplicate order submission
- `tests/test_csv_export.py`: both `format=tax` and `format=full` produce valid CSV with correct headers
- `tests/test_manual_close.py`: full close and partial close
- `tests/test_kelly.py`: edge cases (0% win rate, negative kelly, >max size)
- `tests/test_config_validation.py`: invalid ranges raise `ValidationError`
- `tests/test_atomic_write.py`: paper state survives simulated crash

**Files:** `tests/test_auth.py`, `tests/test_dedup.py`, `tests/test_csv_export.py`, `tests/test_manual_close.py`, `tests/test_kelly.py`, `tests/test_config_validation.py`, `tests/test_atomic_write.py`
