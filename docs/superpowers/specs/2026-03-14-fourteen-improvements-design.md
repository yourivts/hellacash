# HellaCash: 14-Point Improvement Spec

## Overview

Fourteen improvements to the HellaCash trading bot spanning critical bug fixes, security hardening, new features, and tests.

## 1. Position Deduplication Lock (Critical)

**Problem:** Race condition in `trading_loop.py` — if an order fills instantly, a second buy signal for the same symbol can open a duplicate position before the portfolio updates.

**Solution:**
- Add `_pending_orders: set[str]` to `TradingLoop`
- Add symbol to `_pending_orders` before `submit_buy/sell`
- Remove after position is confirmed or order fails
- Skip signal processing for any symbol in `_pending_orders`

**Files:** `bot/trading_loop.py`

## 2. API Key Masking (Critical)

**Problem:** `bitvavo_api_key` and `bitvavo_api_secret` in `config.py` are missing `repr=False`, so they could leak in logs.

**Solution:** Add `repr=False` to both fields, matching the existing pattern on `discord_webhook_url`.

**Files:** `bot/config.py`

## 3. Simple API Key Auth (Critical)

**Problem:** All REST and WebSocket endpoints are open to anyone.

**Solution:**
- Add `API_KEY` field to `config.py` (default empty = no auth for dev convenience)
- Create `api/middleware/auth.py` with a FastAPI dependency that checks `X-API-Key` header
- Apply to all REST routers and `/ws/feed`
- Exempt `/api/health` for external monitoring

**Files:** `bot/config.py`, new `api/middleware/auth.py`, `api/app.py`, `api/ws_hub.py`

## 4. Backtest Slippage Simulation (High)

**Problem:** `BacktestEngine` fills at exact price, making results 0.3-2% more optimistic than live.

**Solution:**
- Add `slippage_pct: float = 0.001` param
- Buy: `fill_price = price * (1 + slippage_pct)`, Sell: `fill_price = price * (1 - slippage_pct)`
- Scale slippage by inverse of 24h volume

**Files:** `bot/backtest/engine.py`

## 5. Sentiment Weight Rebalancing (High)

**Problem:** When a sentiment source fails, its weight disappears rather than being redistributed. Score is skewed.

**Solution:**
- When a source is disabled, redistribute its weight proportionally across active sources
- E.g., Reddit (35%) down -> News: `40/(40+25) = 61.5%`, Fear&Greed: `25/(40+25) = 38.5%`
- Log warning when rebalancing occurs

**Files:** `bot/sentiment/aggregator.py`

## 6. Kelly Sizing Clamp (High)

**Problem:** Kelly fraction can produce negative or oversized values in edge cases.

**Solution:**
- Clamp output to `[0.0, max_position_size_pct / 100]`
- If win_rate is 0 or <10 trades, fall back to fixed minimum size (1% of equity)

**Files:** `bot/risk/position_sizer.py` or `bot/risk/engine.py`

## 7. Graceful Shutdown Timeout (High)

**Problem:** `main.py` can hang forever if async tasks don't cancel.

**Solution:**
- Wrap shutdown sequence in `asyncio.wait_for(timeout=30)`
- If tasks don't cancel within 30s, force-cancel and log warning

**Files:** `bot/main.py`

## 8. Atomic Paper State Write (High)

**Problem:** Crash between in-memory update and `paper_state.json` write loses balance.

**Solution:**
- Write to `paper_state.tmp.json` first
- Use `os.replace()` to atomically swap (works on Windows and Linux)

**Files:** `bot/exchange/bitvavo_client.py`

## 9. Manual Position Close Endpoint (Medium)

**Problem:** No way to close a single position via API.

**Solution:**
- `POST /api/positions/{symbol}/close` with optional `amount` body param
- If `amount` omitted, close entire position
- If `amount` provided, close that fraction and update remaining
- Direction-aware (uses correct sell/buy for long/short)

**Files:** new `api/routers/positions.py`, `api/app.py`

## 10. Trade CSV Export (Medium)

**Problem:** No export for tax reporting.

**Solution:**
- `GET /api/trades/export?format=tax|full&start_date=...&end_date=...`
- `format=tax`: date, pair, side, amount, price, fees, realized_pnl, cost_basis
- `format=full`: all Trade model fields
- Returns `Content-Disposition: attachment` with `StreamingResponse`

**Files:** `api/routers/trades.py`

## 11. Config Range Validation (Medium)

**Problem:** Invalid config values like `KELLY_FRACTION=1.5` silently work.

**Solution:**
- `Field(ge=0, le=1)` on `KELLY_FRACTION`, `MIN_SIGNAL_CONFIDENCE`
- `Field(ge=0, le=100)` on percentage fields
- `Field(ge=0)` on `MAX_DAILY_LOSS_EUR`
- App fails fast on startup with clear Pydantic error

**Files:** `bot/config.py`

## 12. Log Rotation (Medium)

**Problem:** Logs not rotated, could fill disk.

**Solution:**
- Replace `FileHandler` with `RotatingFileHandler`
- `maxBytes=10MB`, `backupCount=5` (keeps last 50MB)

**Files:** `bot/main.py`

## 13. Prometheus Metrics (Medium)

**Problem:** No `/metrics` endpoint for operational monitoring.

**Solution:**
- Add `prometheus_client` dependency
- Expose: `equity_total`, `drawdown_pct`, `trades_total` (counter), `open_positions`, `signals_generated` (counter), `win_rate`
- Increment from event bus subscriptions

**Files:** new `api/metrics.py`, `api/app.py`, `requirements.txt`

## 14. Targeted Tests

Tests for all new/changed code:
- Auth middleware: 401/200/401 for missing/correct/wrong key
- Position dedup: `_pending_orders` prevents duplicates
- CSV export: both formats produce valid CSV
- Manual close: full and partial
- Sentiment rebalancing: weight redistribution with 1-2 sources down
- Kelly clamp: edge cases (0% win, negative kelly, >max)
- Config validation: invalid ranges raise `ValidationError`
- Atomic write: paper state survives simulated crash

**Files:** new `tests/test_improvements.py`
