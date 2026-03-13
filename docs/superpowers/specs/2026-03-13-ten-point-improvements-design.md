# HellaCash — 10-Point Improvement Design

**Date:** 2026-03-13
**Status:** Approved
**Approach:** Grouped (Batch B) — 6 batches of related work

---

## Overview

Ten improvements to the HellaCash autonomous trading bot, grouped into 6 implementation batches for efficient delivery.

---

## Batch 1: Infrastructure — Git + .gitignore

### 1.1 Git Init
- Initialize git repository in project root
- Create initial commit with full project state

### 1.2 .gitignore
Exclude:
```
.env
.env.*
__pycache__/
*.pyc
*.pyo
*.log
models/
.pytest_cache/
*.egg-info/
dist/
build/
postgres_data/
*.db
.idea/
.vscode/
node_modules/
alembic/versions/*.pyc
```

---

## Batch 2: Refactor main.py

**Goal:** Reduce `bot/main.py` from ~717 lines to ~250 lines by extracting three modules.

### 2.1 bot/data_loader.py — CandleCache
- Move `_candle_cache`, `MAX_CANDLES`, `_cache_candle()`, `_get_df()` into `CandleCache` class
- Move `_load_all_candles()`, `_load_candles_for_symbol()`, `_lazy_load_candles()` as methods
- Move `_candles_ready`, `_candles_progress` as instance state
- Public API: `cache_candle(symbol, interval, candle)`, `get_df(symbol, interval)`, `load_all(symbols, client)`, `is_ready`, `progress`

### 2.2 bot/trading_loop.py — TradingLoop
- Move `_strategy_cycle()` and all its helpers: `_check_stops()`, `_close_position()`, `_get_trade_stats()`, `_on_candle()`, `_on_ticker()`, `_on_fill()`
- Constructor:
```python
TradingLoop(
    candle_cache: CandleCache,
    portfolio: PortfolioTracker,
    risk_engine: RiskEngine,
    order_mgr: OrderManager,
    sentiment: SentimentAggregator,
    router: StrategyRouter,
    drawdown: DrawdownGuard,
    trade_analyzer: TradeAnalyzer,
    signal_eval: SignalEvaluator,
    settings: Settings,
)
```
- Public methods: `run_cycle()`, `on_candle()`, `on_ticker()`, `on_fill()`

### 2.3 bot/scheduler.py — BotScheduler
- Move `_sentiment_loop()`, `_strategy_loop()`, `_portfolio_snapshot_loop()`, `_optimizer_loop()` into `BotScheduler` class
- Constructor:
```python
BotScheduler(
    trading_loop: TradingLoop,
    sentiment: SentimentAggregator,
    portfolio: PortfolioTracker,
    optimizer: ParamOptimizer,
    discord_notifier: Optional[DiscordNotifier],
    settings: Settings,
)
```
- `run_all()` method wraps `asyncio.gather()` of all loops (including discord notifier consume loop)
- `main.py` becomes a thin shell: init DB, init components, wire scheduler, start uvicorn

### 2.4 api/routers/control.py Update
- After refactor, `_candles_ready` and `_candles_progress` move into `CandleCache`. The control router currently accesses these as private module state from `bot.main`. Update to access via the `CandleCache` instance (exposed from `bot.main` as a public reference).

---

## Batch 3: Discord Notifications + Sentiment Circuit Breakers

### 3.1 bot/notifications/discord.py — DiscordNotifier
- Uses `aiohttp` to POST JSON to Discord webhook (new dependency, not bundled with existing packages)
- Embed formatting with color coding: red for halts/errors, green for trades, yellow for warnings
- Config additions:
  - `DISCORD_WEBHOOK_URL: str = ""` in Settings (default empty = disabled, marked `repr=False` to avoid logging the secret token)
  - `DISCORD_NOTIFY_TRADES: bool = false`
- Events that trigger notifications:
  - Drawdown circuit breaker triggered (CRITICAL)
  - Daily loss limit hit (CRITICAL)
  - Exchange API errors (WARNING, rate-limited to 1 per 5 min)
  - Trade opened/closed (INFO, only if `DISCORD_NOTIFY_TRADES=true`)
  - Bot start/stop (INFO)
- Subscribes to event bus topics: `TOPIC_RISK_HALT`, `TOPIC_TRADE_OPENED`, `TOPIC_TRADE_CLOSED`, `TOPIC_BOT_STARTED`
- **Lifecycle:** `DiscordNotifier.run()` is a long-running async task that consumes from subscribed event bus queues. It is added to `BotScheduler.run_all()`'s `asyncio.gather()`. Has an `async def shutdown()` method that closes the `aiohttp.ClientSession`.
- All sends are fire-and-forget with 5s timeout, failures logged but never block trading
- When `DISCORD_WEBHOOK_URL` is empty, `run()` returns immediately (no-op)

### 3.2 Sentiment Circuit Breakers
- In `SentimentAggregator`: wrap each scraper (Reddit, News, Fear&Greed) with failure tracking
- Track per-source: `consecutive_failures: int`, `disabled_until: Optional[datetime]`
- After 3 consecutive failures: disable source for 10 minutes, log warning
- **Reweighting formula:** When a source is disabled, redistribute its weight proportionally among remaining active sources. Example: if Reddit (0.35) is disabled, remaining weights become `news: 0.40/0.65 ≈ 0.615`, `fear_greed: 0.25/0.65 ≈ 0.385`. If all sources fail, return 0.0.
- When source recovers: reset failure counter, log info
- Publish degradation events to event bus topic `TOPIC_SENTIMENT_DEGRADED` with payload: `{source, status: "disabled"|"recovered", active_sources: [...], timestamp}`
- Dashboard: degradation events appear in the Live Signal Feed as WARNING-styled entries (no dedicated card needed)

---

## Batch 4: API Improvements

### 4.1 /api/health Endpoint
New router: `api/routers/health.py`
```json
GET /api/health
{
  "status": "healthy" | "degraded" | "unhealthy",
  "checks": {
    "database": { "status": "ok", "latency_ms": 3 },
    "exchange_api": { "status": "ok", "latency_ms": 45 },
    "websocket": { "status": "connected" | "disconnected" },
    "sentiment": { "status": "ok", "degraded_sources": [] }
  },
  "uptime_seconds": 3600,
  "version": "1.0.0"
}
```
- DB check: `SELECT 1` via async session
- Exchange check: lightweight `GET /v2/time` to Bitvavo
- WebSocket: add public `is_connected` property to `BitvavoWebSocket` (expose `_running` state)
- Suitable for Docker healthcheck and monitoring
- Status logic: "healthy" if all checks pass, "degraded" if sentiment sources are down or WS disconnected, "unhealthy" if DB or exchange unreachable

### 4.2 /api/risk/decisions Endpoint
- In-memory ring buffer (last 100 decisions) in `RiskEngine`
- Each decision stored with `gate_details` schema:
```python
{
    "timestamp": "2026-03-13T14:30:00Z",
    "symbol": "BTC-EUR",
    "side": "buy",
    "approved": false,
    "reasons": ["Signal confidence 0.45 < 0.60"],
    "gate_details": {
        "signal_confidence": {"value": 0.45, "threshold": 0.60, "passed": false},
        "drawdown": {"value": 2.1, "threshold": 8.0, "passed": true},
        "daily_loss": {"value": 50.0, "threshold": 200.0, "passed": true},
        "position_size": {"value": 15.0, "threshold": 20.0, "passed": true},
        "max_positions": {"value": 2, "threshold": 5, "passed": true},
        "min_roi": {"value": 1.5, "threshold": 0.80, "passed": true}
    }
}
```
- `GET /api/risk/decisions?limit=20` returns recent decisions
- `gate_details` is added as a new field on the `RiskDecision` dataclass

### 4.3 RiskEngine Enhancement
- Modify `approve()` to build `gate_details` dict for every gate (not just failed ones)
- Publish all decisions (approved + rejected) to event bus topic `TOPIC_RISK_DECISION`
- Store in ring buffer for API access

---

## Batch 5: Alembic + Backtesting Enhancement

### 5.1 Alembic Setup
- `alembic init alembic` with async template
- Configure `alembic/env.py` for async engine + `bot.data.models.Base.metadata`
- Set `sqlalchemy.url` to read from env/config
- Generate initial migration: `alembic revision --autogenerate -m "initial schema"`
- Add `alembic/` and `alembic.ini` to project

### 5.2 Backtesting Enhancement
- Add `POST /api/backtest/compare` endpoint:
  - Request body: `{"symbols": ["BTC-EUR", "ETH-EUR"], "days": 30, "interval": "5m", "initial_capital": 10000}`
  - Runs backtests with concurrency limited to 2 simultaneous symbols (semaphore) to respect Bitvavo rate limits
  - **Error handling:** partial results returned — if one symbol fails, its entry shows `{"symbol": "X", "error": "message"}` alongside successful results
  - Response schema:
  ```json
  {
    "results": [
      {"symbol": "BTC-EUR", "total_pnl": 150.0, "win_rate": 55.0, "sharpe_ratio": 1.2, "max_drawdown_pct": 3.5, "total_trades": 42},
      {"symbol": "ETH-EUR", "error": "No candles fetched"}
    ],
    "params": {"days": 30, "interval": "5m", "initial_capital": 10000}
  }
  ```
- Allow custom `initial_capital` and `max_open_positions` query params on existing `GET /api/backtest/{symbol}` endpoint
- Add `max_open_positions` parameter to `run_backtest()` function (forwarded to `BacktestEngine.__init__()`)

---

## Batch 6: Dashboard Risk Gate Card + Tests

### 6.1 Dashboard — Risk Gate Decisions Card
- New card in right column (below Strategy Performance): "Risk Gate Log"
- Fetches from `GET /api/risk/decisions?limit=10`
- Shows: timestamp, symbol, APPROVED/REJECTED badge, which gates failed
- Auto-refreshes every 30s
- Color coded: green for approved, red for rejected with specific gate names

### 6.2 Test Suite Expansion
All tests placed in `tests/` (flat, matching existing convention). Use `pytest` + `pytest-asyncio`, no real DB or exchange connections needed.

**test_risk_engine.py** (~8 tests):
- Each of the 6 gates individually (signal confidence, drawdown, daily loss, position size, max positions, min ROI)
- Combined: multiple gates failing at once
- Happy path: all gates pass
- Verify `gate_details` structure is populated correctly

**test_order_manager.py** (~4 tests):
- `submit_buy()` success with mocked client
- `submit_sell()` success with mocked client
- `submit_buy()` failure (client raises)
- `_record_order()` persists to DB (mocked session)

**test_strategy_router.py** (~5 tests):
- `detect_regime()` returns TRENDING for high ADX
- `detect_regime()` returns RANGING for low ADX
- `detect_regime()` returns VOLATILE for high ATR%
- `get_strategies()` returns correct set per regime
- `position_size_modifier()` returns 0.5 for VOLATILE

**test_sentiment_circuit_breaker.py** (~4 tests):
- Source disabled after 3 consecutive failures
- Source re-enabled after cooldown period
- Remaining sources reweighted when one is disabled (proportional redistribution)
- All sources failed → returns 0.0 gracefully

---

## Graceful Shutdown

All new long-running components implement `async def shutdown()`:
- `DiscordNotifier.shutdown()` — closes `aiohttp.ClientSession`
- `BotScheduler.shutdown()` — cancels all loop tasks
- `main.py` registers SIGTERM/SIGINT handler that calls shutdown on all components before closing DB

---

## Dependencies

- `aiohttp` — for Discord webhook POST (add to requirements.txt)
- `alembic` — already in requirements.txt
- No other new dependencies needed

## Files Created
- `bot/data_loader.py`
- `bot/trading_loop.py`
- `bot/scheduler.py`
- `bot/notifications/__init__.py`
- `bot/notifications/discord.py`
- `api/routers/health.py`
- `alembic/` directory + `alembic.ini`
- `tests/test_risk_engine.py`
- `tests/test_order_manager.py`
- `tests/test_strategy_router.py`
- `tests/test_sentiment_circuit_breaker.py`
- `.gitignore`

## Files Modified
- `bot/main.py` — slim down, delegate to CandleCache + TradingLoop + BotScheduler
- `bot/config.py` — add Discord settings (DISCORD_WEBHOOK_URL repr=False, DISCORD_NOTIFY_TRADES)
- `bot/risk/engine.py` — add gate_details to RiskDecision, ring buffer, event bus publishing
- `bot/sentiment/aggregator.py` — add circuit breaker logic with proportional reweighting
- `bot/events/bus.py` — add TOPIC_RISK_DECISION, TOPIC_BOT_STARTED, TOPIC_SENTIMENT_DEGRADED
- `bot/exchange/bitvavo_ws.py` — add public `is_connected` property
- `bot/backtest/engine.py` — add `max_open_positions` param to `run_backtest()`
- `api/app.py` — include health router
- `api/routers/control.py` — update to access CandleCache instance instead of private module state
- `api/routers/backtest.py` — add compare endpoint, add query params to existing endpoint
- `frontend/index.html` — add risk gate card
- `frontend/static/app.js` — add risk gate refresh logic, sentiment degradation in signal feed
- `requirements.txt` — add aiohttp
- `.env.example` — add DISCORD_WEBHOOK_URL, DISCORD_NOTIFY_TRADES
