# HellaCash 10-Point Improvements — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add git, refactor main.py, Discord notifications, sentiment circuit breakers, /health endpoint, risk gate visibility, Alembic migrations, backtest enhancements, dashboard improvements, and expanded test coverage to the HellaCash trading bot.

**Architecture:** Extract monolithic `bot/main.py` into focused modules (CandleCache, TradingLoop, BotScheduler). Add Discord webhook notifier and sentiment circuit breakers as event-bus consumers. Expose health and risk decisions via new API endpoints. Wire risk gate details into the dashboard.

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy async, asyncio, aiohttp (new), Alembic, pytest

**Spec:** `docs/superpowers/specs/2026-03-13-ten-point-improvements-design.md`

---

## Chunk 1: Infrastructure + Refactor

### Task 1: Git Init + .gitignore

**Files:**
- Create: `.gitignore`

- [ ] **Step 1: Create .gitignore**

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

- [ ] **Step 2: Init git and create initial commit**

```bash
cd C:/Users/youri/Desktop/trade
git init
git add -A
git status  # verify .env is NOT staged
git commit -m "chore: initial commit of HellaCash trading bot"
```

---

### Task 2: Event Bus — Add New Topic Constants

**Files:**
- Modify: `bot/events/bus.py:13-22`

- [ ] **Step 1: Add three new topic constants**

Add after the existing constants (line 22):

```python
TOPIC_RISK_DECISION = "risk.decision"
TOPIC_BOT_STARTED = "bot.started"
TOPIC_SENTIMENT_DEGRADED = "sentiment.degraded"
```

- [ ] **Step 2: Update main.py to use TOPIC_BOT_STARTED constant**

In `bot/main.py:114`, replace the string literal `"bot.started"` with `TOPIC_BOT_STARTED`. Add the import in the existing bus import block (line 24-30).

- [ ] **Step 3: Commit**

```bash
git add bot/events/bus.py bot/main.py
git commit -m "feat: add event bus topics for risk decisions, bot lifecycle, sentiment degradation"
```

---

### Task 3: Extract CandleCache — bot/data_loader.py

**Files:**
- Create: `bot/data_loader.py`
- Modify: `bot/main.py`

- [ ] **Step 1: Create bot/data_loader.py**

```python
"""Candle cache and historical data loader."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)

MAX_CANDLES = 500


class CandleCache:
    """In-memory cache of OHLCV candle data per symbol/interval."""

    def __init__(self) -> None:
        self._cache: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self._ready = False
        self._progress: Dict[str, Any] = {"loaded": 0, "total": 0, "done": False}

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def progress(self) -> Dict[str, Any]:
        return self._progress

    def cache_candle(self, symbol: str, interval: str, candle: Dict[str, Any]) -> None:
        self._cache.setdefault(symbol, {}).setdefault(interval, [])
        buf = self._cache[symbol][interval]
        buf.append(candle)
        if len(buf) > MAX_CANDLES:
            buf.pop(0)

    def get_df(self, symbol: str, interval: str) -> pd.DataFrame:
        rows = self._cache.get(symbol, {}).get(interval, [])
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        return df

    def has_enough(self, symbol: str, interval: str = "5m", minimum: int = 30) -> bool:
        return len(self._cache.get(symbol, {}).get(interval, [])) >= minimum

    async def load_all(self, symbols: List[str], client: Any, batch_size: int = 5) -> None:
        """Load candles for ALL symbols in concurrent batches."""
        loop = asyncio.get_running_loop()
        total = len(symbols)
        self._progress = {"loaded": 0, "total": total, "done": False}

        for batch_start in range(0, total, batch_size):
            batch = symbols[batch_start:batch_start + batch_size]
            tasks = []
            for symbol in batch:
                if self.has_enough(symbol):
                    continue
                tasks.append(self._load_symbol(client, symbol, loop))
            if tasks:
                await asyncio.gather(*tasks)
            self._progress["loaded"] = min(batch_start + batch_size, total)
            if self._progress["loaded"] % 50 == 0 or self._progress["loaded"] == total:
                logger.info("Candle loading progress: %d/%d", self._progress["loaded"], total)
            await asyncio.sleep(1)

        self._progress["done"] = True
        self._ready = True
        logger.info("All candles loaded (%d pairs)", total)

    async def lazy_load(self, symbol: str, client: Any) -> None:
        """Fetch candles from REST API if not yet cached."""
        if self.has_enough(symbol):
            return
        loop = asyncio.get_running_loop()
        await self._load_symbol(client, symbol, loop)

    async def _load_symbol(self, client: Any, symbol: str, loop) -> None:
        for interval in ["5m", "1h"]:
            try:
                candles = await loop.run_in_executor(
                    None, lambda s=symbol, iv=interval: client.get_candles(s, iv, limit=200)
                )
                for c in candles:
                    self.cache_candle(symbol, interval, {
                        "symbol": c.symbol, "interval": c.interval,
                        "timestamp": c.timestamp, "open": c.open,
                        "high": c.high, "low": c.low,
                        "close": c.close, "volume": c.volume,
                    })
            except Exception as e:
                logger.error("Failed to load candles for %s %s: %s", symbol, interval, e)
```

- [ ] **Step 2: Update main.py — replace candle cache globals with CandleCache instance**

Remove from `bot/main.py`:
- Lines 56-76: `_candle_cache`, `MAX_CANDLES`, `_cache_candle()`, `_get_df()`
- Lines 363-383: `_lazy_load_candles()`
- Lines 560-606: `_load_candles_for_symbol()`, `_load_all_candles()`
- Lines 84-85: `_candles_ready`, `_candles_progress`

Add after the imports:

```python
from bot.data_loader import CandleCache

_candle_cache_obj = CandleCache()
```

Replace all references:
- `_cache_candle(...)` → `_candle_cache_obj.cache_candle(...)`
- `_get_df(...)` → `_candle_cache_obj.get_df(...)`
- `_candles_ready` → `_candle_cache_obj.is_ready`
- `_candles_progress` → `_candle_cache_obj.progress`
- `_load_all_candles()` → `_candle_cache_obj.load_all(_active_symbols, _get_client(), get_settings().candle_batch_size)`
- `_lazy_load_candles(symbol)` → `_candle_cache_obj.lazy_load(symbol, _get_client())`

Add public accessor:

```python
def get_candle_cache() -> CandleCache:
    return _candle_cache_obj
```

- [ ] **Step 3: Update api/routers/control.py**

Replace lines 22-23:

```python
        "candles_ready": bot_main._candles_ready,
        "candles_progress": bot_main._candles_progress,
```

With:

```python
        "candles_ready": bot_main.get_candle_cache().is_ready,
        "candles_progress": bot_main.get_candle_cache().progress,
```

- [ ] **Step 4: Run existing tests to verify no regression**

```bash
cd C:/Users/youri/Desktop/trade
python -m pytest tests/ -v
```

Expected: all existing tests pass.

- [ ] **Step 5: Commit**

```bash
git add bot/data_loader.py bot/main.py api/routers/control.py
git commit -m "refactor: extract CandleCache from main.py into bot/data_loader.py"
```

---

### Task 4: Extract TradingLoop — bot/trading_loop.py

**Files:**
- Create: `bot/trading_loop.py`
- Modify: `bot/main.py`

- [ ] **Step 1: Create bot/trading_loop.py**

Move these functions from `bot/main.py` into a `TradingLoop` class:
- `_on_candle()` (line ~220)
- `_on_ticker()` (line ~229)
- `_on_fill()` (line ~237)
- `_check_stops()` (line ~244)
- `_close_position()` (line ~272)
- `_get_trade_stats()` (line ~320)
- `_strategy_cycle()` (line ~386)

Constructor takes:
```python
def __init__(
    self,
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
) -> None:
```

Public methods: `async def run_cycle()`, `async def on_candle(data)`, `async def on_ticker(data)`, `async def on_fill(data)`

The methods are identical to the current implementations in `main.py`, just using `self.portfolio` instead of `_get_portfolio()`, etc. The `_trade_stats_cache` and `_trade_stats_ts` become instance attributes.

- [ ] **Step 2: Update main.py**

Remove the moved functions. Create TradingLoop instance after singletons are initialized:

```python
_trading_loop: Optional[TradingLoop] = None

def _get_trading_loop() -> TradingLoop:
    global _trading_loop
    if _trading_loop is None:
        _trading_loop = TradingLoop(
            candle_cache=_candle_cache_obj,
            portfolio=_get_portfolio(),
            risk_engine=_get_risk_engine(),
            order_mgr=_get_order_mgr(),
            sentiment=_get_sentiment(),
            router=_get_router(),
            drawdown=_get_drawdown(),
            trade_analyzer=_get_trade_analyzer(),
            signal_eval=_get_signal_eval(),
            settings=get_settings(),
        )
    return _trading_loop
```

Update WebSocket callbacks in `_main()` to use:
```python
trading_loop = _get_trading_loop()
ws = BitvavoWebSocket(
    api_key=settings.bitvavo_api_key,
    api_secret=settings.bitvavo_api_secret,
    on_candle=trading_loop.on_candle,
    on_ticker=trading_loop.on_ticker,
    on_fill=trading_loop.on_fill,
)
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/ -v
```

- [ ] **Step 4: Commit**

```bash
git add bot/trading_loop.py bot/main.py
git commit -m "refactor: extract TradingLoop from main.py into bot/trading_loop.py"
```

---

### Task 5: Extract BotScheduler — bot/scheduler.py

**Files:**
- Create: `bot/scheduler.py`
- Modify: `bot/main.py`

- [ ] **Step 1: Create bot/scheduler.py**

```python
"""Async loop scheduler for all periodic bot tasks."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

from bot.config import Settings

logger = logging.getLogger(__name__)


class BotScheduler:
    """Manages all periodic async loops and coordinates shutdown."""

    def __init__(
        self,
        run_cycle: Callable,
        sentiment_cycle: Callable,
        portfolio_snapshot: Callable,
        optimizer_run: Callable,
        is_running: Callable[[], bool],
        active_symbols: Callable[[], list],
        settings: Settings,
        discord_run: Optional[Callable] = None,
    ) -> None:
        self._run_cycle = run_cycle
        self._sentiment_cycle = sentiment_cycle
        self._portfolio_snapshot = portfolio_snapshot
        self._optimizer_run = optimizer_run
        self._is_running = is_running
        self._active_symbols = active_symbols
        self._settings = settings
        self._discord_run = discord_run
        self._tasks: list[asyncio.Task] = []

    async def run_all(self) -> None:
        """Start all loops. Call with asyncio.gather alongside other tasks."""
        coros = [
            self._strategy_loop(self._settings.strategy_cycle_secs),
            self._sentiment_loop(self._settings.sentiment_poll_interval_secs),
            self._portfolio_snapshot_loop(300),
            self._optimizer_loop(24),
        ]
        if self._discord_run:
            coros.append(self._discord_run())
        self._tasks = [asyncio.create_task(c) for c in coros]
        await asyncio.gather(*self._tasks)

    async def _strategy_loop(self, interval_secs: int) -> None:
        while True:
            if self._is_running():
                await self._run_cycle()
            await asyncio.sleep(interval_secs)

    async def _sentiment_loop(self, interval_secs: int) -> None:
        while True:
            symbols = self._active_symbols()
            if symbols:
                try:
                    await self._sentiment_cycle(symbols)
                except Exception as e:
                    logger.error("Sentiment loop error: %s", e)
            await asyncio.sleep(interval_secs)

    async def _portfolio_snapshot_loop(self, interval_secs: int) -> None:
        while True:
            if self._is_running():
                try:
                    await self._portfolio_snapshot()
                except Exception as e:
                    logger.error("Snapshot error: %s", e)
            await asyncio.sleep(interval_secs)

    async def _optimizer_loop(self, interval_hours: int) -> None:
        while True:
            await asyncio.sleep(interval_hours * 3600)
            if self._is_running():
                try:
                    await self._optimizer_run()
                except Exception as e:
                    logger.error("Optimizer error: %s", e)

    async def shutdown(self) -> None:
        """Cancel all running loop tasks."""
        for task in self._tasks:
            task.cancel()
        logger.info("BotScheduler shutdown complete")
```

- [ ] **Step 2: Update main.py — replace inline loops with BotScheduler**

Remove `_sentiment_loop()`, `_strategy_loop()`, `_portfolio_snapshot_loop()`, `_optimizer_loop()` from `main.py`.

In `_main()`, replace the inline loop functions and `asyncio.gather()` call with:

```python
scheduler = BotScheduler(
    run_cycle=_get_trading_loop().run_cycle,
    sentiment_cycle=_get_sentiment().run_cycle,
    portfolio_snapshot=_get_portfolio().snapshot,
    optimizer_run=lambda: _get_param_optimizer().optimize_hybrid(
        min_trades=settings.optimizer_min_trades
    ),
    is_running=is_running,
    active_symbols=get_active_symbols,
    settings=settings,
)

await asyncio.gather(
    _candle_cache_obj.load_all(_active_symbols, _get_client(), settings.candle_batch_size),
    ws.run(),
    scheduler.run_all(),
    server.serve(),
)
```

- [ ] **Step 3: Add graceful shutdown handler to main.py**

Add signal handler in `_main()` before the gather:

```python
import signal as _signal

async def _shutdown():
    logger.info("Shutting down...")
    await stop_bot()
    await scheduler.shutdown()
    await ws.stop()
    await close_db()

loop = asyncio.get_running_loop()
for sig in (_signal.SIGTERM, _signal.SIGINT):
    try:
        loop.add_signal_handler(sig, lambda: asyncio.create_task(_shutdown()))
    except NotImplementedError:
        pass  # Windows doesn't support add_signal_handler
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/ -v
```

- [ ] **Step 5: Commit**

```bash
git add bot/scheduler.py bot/main.py
git commit -m "refactor: extract BotScheduler from main.py, add graceful shutdown"
```

---

## Chunk 2: Discord + Sentiment + Risk Engine

### Task 6: Config — Add Discord Settings

**Files:**
- Modify: `bot/config.py:52-53`
- Modify: `.env.example`

- [ ] **Step 1: Add Discord settings to bot/config.py**

Add after the API settings section (after line 55):

```python
    # ── Discord notifications ──────────────────────────────────────────────
    discord_webhook_url: str = ""
    discord_notify_trades: bool = False
```

Add `repr=False` to the webhook field to prevent logging the secret. Use Pydantic field:

```python
from pydantic import Field, field_validator
# ...
    discord_webhook_url: str = Field(default="", repr=False)
```

- [ ] **Step 2: Add to .env.example**

Append:

```
# ── Discord notifications ─────────────────────────────────────────────────
DISCORD_WEBHOOK_URL=
DISCORD_NOTIFY_TRADES=false
```

- [ ] **Step 3: Commit**

```bash
git add bot/config.py .env.example
git commit -m "feat: add Discord notification settings to config"
```

---

### Task 7: RiskEngine — Gate Details + Ring Buffer

**Files:**
- Modify: `bot/risk/engine.py`
- Create: `tests/test_risk_engine.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_risk_engine.py`:

```python
"""Tests for bot.risk.engine.RiskEngine."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from bot.config import Settings
from bot.risk.engine import RiskEngine, RiskDecision


def _make_settings(**overrides) -> Settings:
    defaults = {
        "bitvavo_api_key": "", "bitvavo_api_secret": "",
        "database_url": "postgresql+asyncpg://x:x@localhost/x",
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _approve_defaults(**overrides) -> dict:
    """Default args that pass all gates."""
    base = dict(
        symbol="BTC-EUR", side="buy", position_size_eur=1000.0,
        signal_confidence=0.75, expected_roi_pct=5.0,
        portfolio_equity_eur=10000.0, open_position_count=0,
        daily_loss_eur=0.0, current_drawdown_pct=0.0,
    )
    base.update(overrides)
    return base


class TestAllGatesPass:
    def test_all_gates_pass(self):
        engine = RiskEngine(_make_settings())
        decision = engine.approve(**_approve_defaults())
        assert decision.approved is True
        assert decision.reasons == []
        assert "signal_confidence" in decision.gate_details
        assert decision.gate_details["signal_confidence"]["passed"] is True


class TestSignalConfidenceGate:
    def test_low_confidence_rejected(self):
        engine = RiskEngine(_make_settings())
        decision = engine.approve(**_approve_defaults(signal_confidence=0.40))
        assert decision.approved is False
        assert decision.gate_details["signal_confidence"]["passed"] is False
        assert decision.gate_details["signal_confidence"]["value"] == 0.40


class TestDrawdownGate:
    def test_high_drawdown_rejected(self):
        engine = RiskEngine(_make_settings())
        decision = engine.approve(**_approve_defaults(current_drawdown_pct=9.0))
        assert decision.approved is False
        assert decision.gate_details["drawdown"]["passed"] is False


class TestDailyLossGate:
    def test_daily_loss_exceeded(self):
        engine = RiskEngine(_make_settings())
        decision = engine.approve(**_approve_defaults(daily_loss_eur=250.0))
        assert decision.approved is False
        assert decision.gate_details["daily_loss"]["passed"] is False


class TestPositionSizeGate:
    def test_oversized_position(self):
        engine = RiskEngine(_make_settings())
        decision = engine.approve(**_approve_defaults(position_size_eur=3000.0))
        assert decision.approved is False
        assert decision.gate_details["position_size"]["passed"] is False


class TestMaxPositionsGate:
    def test_too_many_positions(self):
        engine = RiskEngine(_make_settings())
        decision = engine.approve(**_approve_defaults(open_position_count=5))
        assert decision.approved is False
        assert decision.gate_details["max_positions"]["passed"] is False


class TestMinROIGate:
    def test_low_roi(self):
        engine = RiskEngine(_make_settings())
        decision = engine.approve(**_approve_defaults(expected_roi_pct=0.1))
        assert decision.approved is False
        assert decision.gate_details["min_roi"]["passed"] is False


class TestMultipleGatesFail:
    def test_multiple_gates_fail(self):
        engine = RiskEngine(_make_settings())
        decision = engine.approve(**_approve_defaults(
            signal_confidence=0.40, current_drawdown_pct=9.0
        ))
        assert decision.approved is False
        assert len(decision.reasons) == 2
        assert decision.gate_details["signal_confidence"]["passed"] is False
        assert decision.gate_details["drawdown"]["passed"] is False


class TestRingBuffer:
    def test_recent_decisions_stored(self):
        engine = RiskEngine(_make_settings())
        engine.approve(**_approve_defaults())
        engine.approve(**_approve_defaults(signal_confidence=0.40))
        decisions = engine.get_recent_decisions(limit=10)
        assert len(decisions) == 2
        assert decisions[0]["approved"] is False  # most recent first
        assert decisions[1]["approved"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_risk_engine.py -v
```

Expected: FAIL (gate_details not yet on RiskDecision, get_recent_decisions doesn't exist)

- [ ] **Step 3: Implement gate_details + ring buffer in RiskEngine**

Modify `bot/risk/engine.py`:

```python
"""Risk engine — mandatory gate before every order."""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from bot.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class RiskDecision:
    approved: bool
    reasons: List[str] = field(default_factory=list)
    gate_details: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __str__(self) -> str:
        if self.approved:
            return "APPROVED"
        return "REJECTED: " + "; ".join(self.reasons)


class RiskEngine:
    """
    Validates every proposed trade against hard risk limits.
    Risk limits come from Settings and are NEVER modified by the learning system.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._recent_decisions: deque = deque(maxlen=100)

    def approve(
        self,
        symbol: str,
        side: str,
        position_size_eur: float,
        signal_confidence: float,
        expected_roi_pct: float,
        portfolio_equity_eur: float,
        open_position_count: int,
        daily_loss_eur: float,
        current_drawdown_pct: float,
        taker_fee_pct: float = 0.25,
    ) -> RiskDecision:
        """Run all risk checks. Returns RiskDecision(approved, reasons, gate_details)."""
        reasons: List[str] = []
        gate_details: Dict[str, Dict[str, Any]] = {}

        # 1. Signal confidence
        passed = signal_confidence >= self.settings.min_signal_confidence
        gate_details["signal_confidence"] = {
            "value": signal_confidence,
            "threshold": self.settings.min_signal_confidence,
            "passed": passed,
        }
        if not passed:
            reasons.append(
                f"Signal confidence {signal_confidence:.2f} < {self.settings.min_signal_confidence}"
            )

        # 2. Drawdown circuit breaker
        passed = current_drawdown_pct < self.settings.max_drawdown_pct
        gate_details["drawdown"] = {
            "value": current_drawdown_pct,
            "threshold": self.settings.max_drawdown_pct,
            "passed": passed,
        }
        if not passed:
            reasons.append(
                f"Drawdown {current_drawdown_pct:.1f}% >= limit {self.settings.max_drawdown_pct}%"
            )

        # 3. Daily loss limit
        passed = daily_loss_eur < self.settings.max_daily_loss_eur
        gate_details["daily_loss"] = {
            "value": daily_loss_eur,
            "threshold": self.settings.max_daily_loss_eur,
            "passed": passed,
        }
        if not passed:
            reasons.append(
                f"Daily loss EUR{daily_loss_eur:.2f} >= limit EUR{self.settings.max_daily_loss_eur:.2f}"
            )

        # 4. Position concentration
        position_pct = (position_size_eur / portfolio_equity_eur * 100.0) if portfolio_equity_eur > 0 else 100.0
        passed = position_pct <= self.settings.max_position_size_pct
        gate_details["position_size"] = {
            "value": round(position_pct, 1),
            "threshold": self.settings.max_position_size_pct,
            "passed": passed,
        }
        if not passed:
            reasons.append(
                f"Position {position_pct:.1f}% > max {self.settings.max_position_size_pct}%"
            )

        # 5. Max open positions
        passed = not (side == "buy" and open_position_count >= self.settings.max_open_positions)
        gate_details["max_positions"] = {
            "value": open_position_count,
            "threshold": self.settings.max_open_positions,
            "passed": passed,
        }
        if not passed:
            reasons.append(
                f"Open positions {open_position_count} >= max {self.settings.max_open_positions}"
            )

        # 6. Minimum ROI potential
        min_roi = self.settings.min_trade_roi_pct + 2 * taker_fee_pct
        passed = expected_roi_pct >= min_roi
        gate_details["min_roi"] = {
            "value": expected_roi_pct,
            "threshold": round(min_roi, 2),
            "passed": passed,
        }
        if not passed:
            reasons.append(
                f"Expected ROI {expected_roi_pct:.2f}% < required {min_roi:.2f}%"
            )

        approved = len(reasons) == 0
        decision = RiskDecision(approved=approved, reasons=reasons, gate_details=gate_details)

        if not approved:
            logger.info("RiskEngine REJECTED %s %s: %s", side, symbol, "; ".join(reasons))
        else:
            logger.debug("RiskEngine approved %s %s (confidence=%.2f)", side, symbol, signal_confidence)

        # Store in ring buffer
        self._recent_decisions.appendleft({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "side": side,
            "approved": approved,
            "reasons": reasons,
            "gate_details": gate_details,
        })

        return decision

    def get_recent_decisions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return the most recent risk decisions (newest first)."""
        return list(self._recent_decisions)[:limit]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_risk_engine.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add bot/risk/engine.py tests/test_risk_engine.py
git commit -m "feat: add gate_details and ring buffer to RiskEngine with tests"
```

---

### Task 8: Discord Notifier

**Files:**
- Create: `bot/notifications/__init__.py`
- Create: `bot/notifications/discord.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Add aiohttp to requirements.txt**

Add after the websockets line:

```
aiohttp>=3.9.0
```

- [ ] **Step 2: Create bot/notifications/__init__.py**

Empty file.

- [ ] **Step 3: Create bot/notifications/discord.py**

```python
"""Discord webhook notifier — fire-and-forget alerts."""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from bot.config import Settings
from bot.events.bus import (
    TOPIC_BOT_STARTED,
    TOPIC_RISK_HALT,
    TOPIC_TRADE_CLOSED,
    TOPIC_TRADE_OPENED,
    get_bus,
)

logger = logging.getLogger(__name__)

# Embed colors
COLOR_CRITICAL = 0xEF4444  # red
COLOR_WARNING = 0xF59E0B   # yellow
COLOR_INFO = 0x3B82F6      # blue
COLOR_SUCCESS = 0x10B981   # green


class DiscordNotifier:
    """Consumes event bus messages and sends Discord webhook embeds."""

    def __init__(self, settings: Settings) -> None:
        self._webhook_url = settings.discord_webhook_url
        self._notify_trades = settings.discord_notify_trades
        self._session = None
        self._last_error_sent: float = 0.0
        self._error_cooldown = 300  # 5 minutes

    @property
    def enabled(self) -> bool:
        return bool(self._webhook_url)

    async def run(self) -> None:
        """Main loop: subscribe to topics and relay to Discord."""
        if not self.enabled:
            return

        try:
            import aiohttp
            self._session = aiohttp.ClientSession()
        except ImportError:
            logger.warning("aiohttp not installed — Discord notifications disabled")
            return

        bus = get_bus()
        queues = {}
        topics = [TOPIC_RISK_HALT, TOPIC_BOT_STARTED]
        if self._notify_trades:
            topics.extend([TOPIC_TRADE_OPENED, TOPIC_TRADE_CLOSED])

        for topic in topics:
            queues[topic] = await bus.subscribe(topic)

        logger.info("Discord notifier started (topics: %s)", [t for t in topics])

        while True:
            for topic, queue in queues.items():
                try:
                    event = queue.get_nowait()
                    await self._handle_event(topic, event.payload)
                except asyncio.QueueEmpty:
                    pass
            await asyncio.sleep(1)

    async def _handle_event(self, topic: str, data: Dict[str, Any]) -> None:
        if topic == TOPIC_RISK_HALT:
            await self._send_embed(
                title="CIRCUIT BREAKER TRIGGERED",
                description=data.get("reason", "Trading halted"),
                color=COLOR_CRITICAL,
            )
        elif topic == TOPIC_BOT_STARTED:
            count = len(data.get("symbols", []))
            await self._send_embed(
                title="Bot Started",
                description=f"Trading activated with {count} symbols",
                color=COLOR_INFO,
            )
        elif topic == TOPIC_TRADE_OPENED:
            await self._send_embed(
                title=f"Trade Opened: {data.get('symbol', '?')}",
                description=(
                    f"Strategy: {data.get('strategy_name', '?')}\n"
                    f"Price: EUR{data.get('fill_price', 0):.2f}\n"
                    f"Amount: {data.get('filled_amount', 0):.4f}"
                ),
                color=COLOR_SUCCESS,
            )
        elif topic == TOPIC_TRADE_CLOSED:
            pnl = data.get("net_pnl", 0)
            color = COLOR_SUCCESS if pnl >= 0 else COLOR_CRITICAL
            await self._send_embed(
                title=f"Trade Closed: {data.get('symbol', '?')}",
                description=(
                    f"P&L: EUR{pnl:.2f}\n"
                    f"Reason: {data.get('reason', '?')}"
                ),
                color=color,
            )

    async def send_error(self, message: str) -> None:
        """Rate-limited error notification."""
        now = time.time()
        if now - self._last_error_sent < self._error_cooldown:
            return
        self._last_error_sent = now
        await self._send_embed(
            title="Exchange Error",
            description=message,
            color=COLOR_WARNING,
        )

    async def _send_embed(
        self, title: str, description: str, color: int
    ) -> None:
        if not self._session:
            return
        payload = {
            "embeds": [{
                "title": title,
                "description": description,
                "color": color,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "footer": {"text": "HellaCash Bot"},
            }]
        }
        try:
            async with self._session.post(
                self._webhook_url, json=payload, timeout=5
            ) as resp:
                if resp.status >= 400:
                    logger.warning("Discord webhook returned %d", resp.status)
        except Exception as e:
            logger.debug("Discord send failed: %s", e)

    async def shutdown(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None
```

- [ ] **Step 4: Wire into main.py**

In `_main()`, after creating the scheduler, add:

```python
from bot.notifications.discord import DiscordNotifier

discord = DiscordNotifier(settings)
```

Pass `discord_run=discord.run` to the BotScheduler constructor.

Add `await discord.shutdown()` in the `_shutdown()` handler.

- [ ] **Step 5: Commit**

```bash
git add bot/notifications/ requirements.txt bot/main.py
git commit -m "feat: add Discord webhook notifier with event bus integration"
```

---

### Task 9: Sentiment Circuit Breakers

**Files:**
- Modify: `bot/sentiment/aggregator.py`
- Create: `tests/test_sentiment_circuit_breaker.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_sentiment_circuit_breaker.py`:

```python
"""Tests for sentiment source circuit breakers."""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from bot.sentiment.aggregator import SentimentAggregator, SOURCE_WEIGHTS


def _make_aggregator() -> SentimentAggregator:
    settings = MagicMock()
    settings.reddit_client_id = ""
    settings.reddit_client_secret = ""
    settings.reddit_user_agent = "test"
    agg = SentimentAggregator(settings)
    return agg


class TestCircuitBreakerDisablesAfterFailures:
    def test_source_disabled_after_3_failures(self):
        agg = _make_aggregator()
        for _ in range(3):
            agg._record_failure("reddit")
        assert agg._is_source_disabled("reddit") is True

    def test_source_not_disabled_after_2_failures(self):
        agg = _make_aggregator()
        for _ in range(2):
            agg._record_failure("reddit")
        assert agg._is_source_disabled("reddit") is False


class TestCircuitBreakerRecovery:
    def test_source_recovers_after_cooldown(self):
        agg = _make_aggregator()
        for _ in range(3):
            agg._record_failure("reddit")
        assert agg._is_source_disabled("reddit") is True
        # Simulate cooldown expired
        agg._source_disabled_until["reddit"] = datetime.now(timezone.utc) - timedelta(seconds=1)
        assert agg._is_source_disabled("reddit") is False


class TestReweighting:
    def test_reweight_with_one_source_disabled(self):
        agg = _make_aggregator()
        weights = agg._get_active_weights(disabled={"reddit"})
        # news=0.40, fear_greed=0.25 → renormalized to sum=1
        assert abs(weights["news"] - 0.40 / 0.65) < 0.01
        assert abs(weights["fear_greed"] - 0.25 / 0.65) < 0.01
        assert "reddit" not in weights

    def test_all_sources_disabled_returns_empty(self):
        agg = _make_aggregator()
        weights = agg._get_active_weights(disabled={"news", "reddit", "fear_greed"})
        assert weights == {}


class TestRecordSuccess:
    def test_success_resets_failure_count(self):
        agg = _make_aggregator()
        for _ in range(2):
            agg._record_failure("news")
        agg._record_success("news")
        assert agg._source_failures.get("news", 0) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_sentiment_circuit_breaker.py -v
```

- [ ] **Step 3: Add circuit breaker methods to SentimentAggregator**

Add to `bot/sentiment/aggregator.py` in `__init__`:

```python
        self._source_failures: Dict[str, int] = {}
        self._source_disabled_until: Dict[str, datetime] = {}
        self._max_failures = 3
        self._cooldown_secs = 600  # 10 minutes
```

Add new methods:

```python
    def _record_failure(self, source: str) -> None:
        self._source_failures[source] = self._source_failures.get(source, 0) + 1
        if self._source_failures[source] >= self._max_failures:
            until = datetime.now(timezone.utc) + timedelta(seconds=self._cooldown_secs)
            self._source_disabled_until[source] = until
            logger.warning("Sentiment source '%s' disabled until %s", source, until.isoformat())
            # Publish degradation event
            try:
                active = [s for s in SOURCE_WEIGHTS if not self._is_source_disabled(s)]
                asyncio.get_running_loop().create_task(
                    self._bus.publish(TOPIC_SENTIMENT_DEGRADED, {
                        "source": source, "status": "disabled",
                        "active_sources": active,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                )
            except RuntimeError:
                pass  # no running loop (e.g. in tests)

    def _record_success(self, source: str) -> None:
        was_disabled = source in self._source_disabled_until
        self._source_failures[source] = 0
        self._source_disabled_until.pop(source, None)
        if was_disabled:
            try:
                active = [s for s in SOURCE_WEIGHTS if not self._is_source_disabled(s)]
                asyncio.get_running_loop().create_task(
                    self._bus.publish(TOPIC_SENTIMENT_DEGRADED, {
                        "source": source, "status": "recovered",
                        "active_sources": active,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                )
            except RuntimeError:
                pass

    def _is_source_disabled(self, source: str) -> bool:
        until = self._source_disabled_until.get(source)
        if until is None:
            return False
        if datetime.now(timezone.utc) >= until:
            self._source_disabled_until.pop(source, None)
            self._source_failures[source] = 0
            logger.info("Sentiment source '%s' recovered (cooldown expired)", source)
            return False
        return True

    def _get_active_weights(self, disabled: set) -> Dict[str, float]:
        active = {k: v for k, v in SOURCE_WEIGHTS.items() if k not in disabled}
        if not active:
            return {}
        total = sum(active.values())
        return {k: v / total for k, v in active.items()}
```

Add `from datetime import timedelta` and `import asyncio` to the imports. Add `TOPIC_SENTIMENT_DEGRADED` to the existing bus import from `bot.events.bus`.

- [ ] **Step 4: Wrap scraper calls in run_cycle with try/except + circuit breaker**

In `run_cycle()`, wrap the prefetch and per-asset scraping calls. For example, wrap `self._fear_greed.get_score` with:

```python
        # Fear & Greed
        if self._is_source_disabled("fear_greed"):
            fear_greed_score = 0.0
        else:
            try:
                fear_greed_score = await loop.run_in_executor(None, self._fear_greed.get_score)
                self._record_success("fear_greed")
            except Exception as e:
                logger.error("Fear & Greed scraper failed: %s", e)
                self._record_failure("fear_greed")
                fear_greed_score = 0.0
```

Similarly wrap Reddit and News fetches. In the scoring loop, use `_get_active_weights()` instead of `SOURCE_WEIGHTS` directly.

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/test_sentiment_circuit_breaker.py -v
```

Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add bot/sentiment/aggregator.py tests/test_sentiment_circuit_breaker.py
git commit -m "feat: add circuit breakers to sentiment scrapers with proportional reweighting"
```

---

## Chunk 3: API Endpoints + Dashboard

### Task 10: WebSocket is_connected Property

**Files:**
- Modify: `bot/exchange/bitvavo_ws.py:44`

- [ ] **Step 1: Add property**

Add after line 45 (`self._ws = None`):

```python
    @property
    def is_connected(self) -> bool:
        return self._running and self._ws is not None
```

- [ ] **Step 2: Commit**

```bash
git add bot/exchange/bitvavo_ws.py
git commit -m "feat: add is_connected property to BitvavoWebSocket"
```

---

### Task 11: Health Endpoint

**Files:**
- Create: `api/routers/health.py`
- Modify: `api/app.py`

- [ ] **Step 1: Create api/routers/health.py**

```python
"""Health check endpoint for monitoring and Docker healthchecks."""
from __future__ import annotations

import asyncio
import time
import urllib.request

from fastapi import APIRouter
from sqlalchemy import text

from bot.data.database import get_session

router = APIRouter(tags=["health"])

_start_time = time.time()


@router.get("/api/health")
async def health_check():
    checks = {}
    overall = "healthy"

    # Database
    try:
        t0 = time.time()
        async with get_session() as session:
            await session.execute(text("SELECT 1"))
        checks["database"] = {"status": "ok", "latency_ms": round((time.time() - t0) * 1000)}
    except Exception as e:
        checks["database"] = {"status": "error", "error": str(e)}
        overall = "unhealthy"

    # Exchange API (run in executor to avoid blocking event loop)
    try:
        t0 = time.time()
        loop = asyncio.get_running_loop()
        def _check_exchange():
            req = urllib.request.Request(
                "https://api.bitvavo.com/v2/time",
                headers={"User-Agent": "hellacash/1.0"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                resp.read()
        await loop.run_in_executor(None, _check_exchange)
        checks["exchange_api"] = {"status": "ok", "latency_ms": round((time.time() - t0) * 1000)}
    except Exception as e:
        checks["exchange_api"] = {"status": "error", "error": str(e)}
        overall = "unhealthy"

    # WebSocket
    try:
        import bot.main as bot_main
        ws_connected = getattr(bot_main, '_ws_instance', None)
        if ws_connected and hasattr(ws_connected, 'is_connected'):
            checks["websocket"] = {"status": "connected" if ws_connected.is_connected else "disconnected"}
        else:
            checks["websocket"] = {"status": "unknown"}
        if checks["websocket"]["status"] == "disconnected":
            if overall == "healthy":
                overall = "degraded"
    except Exception:
        checks["websocket"] = {"status": "unknown"}

    # Sentiment sources
    try:
        import bot.main as bot_main
        sentiment = bot_main._get_sentiment()
        degraded = [
            src for src in ["news", "reddit", "fear_greed"]
            if sentiment._is_source_disabled(src)
        ]
        checks["sentiment"] = {
            "status": "degraded" if degraded else "ok",
            "degraded_sources": degraded,
        }
        if degraded and overall == "healthy":
            overall = "degraded"
    except Exception:
        checks["sentiment"] = {"status": "unknown", "degraded_sources": []}

    return {
        "status": overall,
        "checks": checks,
        "uptime_seconds": round(time.time() - _start_time),
        "version": "1.0.0",
    }
```

- [ ] **Step 2: Register in api/app.py**

Add import:
```python
from api.routers import analytics, backtest, charts, control, health, portfolio, sentiment, trades
```

Add after the other router includes:
```python
    app.include_router(health.router)
```

- [ ] **Step 3: Expose WebSocket instance from main.py**

In `bot/main.py`, in `_main()`, after creating the WebSocket instance, store it as a module-level reference:

```python
global _ws_instance
_ws_instance = ws
```

Add at module level:
```python
_ws_instance = None
```

- [ ] **Step 4: Commit**

```bash
git add api/routers/health.py api/app.py bot/main.py
git commit -m "feat: add /api/health endpoint with DB, exchange, WS, sentiment checks"
```

---

### Task 12: Risk Decisions API Endpoint

**Files:**
- Create: `api/routers/risk.py`
- Modify: `api/app.py`

- [ ] **Step 1: Create api/routers/risk.py**

```python
"""Risk gate decision log endpoint."""
from __future__ import annotations

from fastapi import APIRouter, Query

import bot.main as bot_main

router = APIRouter(prefix="/api/risk", tags=["risk"])


@router.get("/decisions")
async def get_risk_decisions(
    limit: int = Query(20, ge=1, le=100, description="Number of recent decisions"),
):
    """Return recent risk gate decisions (newest first)."""
    engine = bot_main._get_risk_engine()
    return engine.get_recent_decisions(limit=limit)
```

- [ ] **Step 2: Register in api/app.py**

Add to imports and include:
```python
from api.routers import analytics, backtest, charts, control, health, portfolio, risk, sentiment, trades
# ...
    app.include_router(risk.router)
```

- [ ] **Step 3: Commit**

```bash
git add api/routers/risk.py api/app.py
git commit -m "feat: add /api/risk/decisions endpoint for risk gate visibility"
```

---

### Task 13: Strategy Router Tests

**Files:**
- Create: `tests/test_strategy_router.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for bot.strategy.router."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bot.strategy.router import Regime, StrategyRouter, detect_regime
from bot.strategy.hybrid import HybridStrategy
from bot.strategy.trend_following import TrendFollowingStrategy
from bot.strategy.breakout import BreakoutStrategy
from bot.strategy.mean_reversion import MeanReversionStrategy


def _make_df(n: int = 100, trend: str = "flat", volatility: float = 0.01) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    np.random.seed(42)
    close = [100.0]
    for i in range(1, n):
        if trend == "up":
            drift = 0.002
        elif trend == "down":
            drift = -0.002
        else:
            drift = 0.0
        close.append(close[-1] * (1 + drift + np.random.normal(0, volatility)))
    close = np.array(close)
    high = close * (1 + np.random.uniform(0, volatility, n))
    low = close * (1 - np.random.uniform(0, volatility, n))
    open_ = close * (1 + np.random.normal(0, volatility / 2, n))
    volume = np.random.uniform(100, 1000, n)
    idx = pd.date_range("2026-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


class TestDetectRegime:
    def test_trending_market(self):
        df = _make_df(100, trend="up", volatility=0.005)
        regime = detect_regime(df)
        assert regime in (Regime.TRENDING, Regime.UNKNOWN)

    def test_volatile_market(self):
        df = _make_df(100, trend="flat", volatility=0.05)
        regime = detect_regime(df)
        assert regime in (Regime.VOLATILE, Regime.UNKNOWN)

    def test_ranging_market(self):
        df = _make_df(100, trend="flat", volatility=0.002)
        regime = detect_regime(df)
        assert regime in (Regime.RANGING, Regime.UNKNOWN)

    def test_insufficient_data(self):
        df = _make_df(10)
        assert detect_regime(df) == Regime.UNKNOWN


class TestStrategyRouter:
    def test_trending_includes_trend_strategy(self):
        router = StrategyRouter()
        strategies = router.get_strategies(Regime.TRENDING)
        types = [type(s) for s in strategies]
        assert TrendFollowingStrategy in types
        assert HybridStrategy in types

    def test_ranging_includes_mean_reversion(self):
        router = StrategyRouter()
        strategies = router.get_strategies(Regime.RANGING)
        types = [type(s) for s in strategies]
        assert MeanReversionStrategy in types

    def test_volatile_position_modifier(self):
        router = StrategyRouter()
        assert router.position_size_modifier(Regime.VOLATILE) == 0.5
        assert router.position_size_modifier(Regime.TRENDING) == 1.0
```

- [ ] **Step 2: Run tests**

```bash
python -m pytest tests/test_strategy_router.py -v
```

Expected: all PASS (these test existing code)

- [ ] **Step 3: Commit**

```bash
git add tests/test_strategy_router.py
git commit -m "test: add strategy router and regime detection tests"
```

---

### Task 14: Order Manager Tests

**Files:**
- Create: `tests/test_order_manager.py`

- [ ] **Step 1: Write tests**

```python
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
            "orderId": "abc123",
            "market": "BTC-EUR",
            "side": "buy",
            "orderType": "market",
            "status": "filled",
            "price": "50000",
            "amount": "0.1",
            "filledAmount": "0.1",
            "feePaid": "1.25",
            "feeCurrency": "EUR",
            "paper": True,
        }
        with patch("bot.exchange.order_manager.get_session") as mock_gs, \
             patch("bot.exchange.order_manager.save_order") as mock_save:
            mock_ctx = AsyncMock()
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)
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
            "orderId": "def456",
            "market": "BTC-EUR",
            "side": "sell",
            "orderType": "market",
            "status": "filled",
            "price": "51000",
            "amount": "0.1",
            "filledAmount": "0.1",
            "feePaid": "1.28",
            "feeCurrency": "EUR",
        }
        with patch("bot.exchange.order_manager.get_session") as mock_gs, \
             patch("bot.exchange.order_manager.save_order") as mock_save:
            mock_ctx = AsyncMock()
            mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)
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
```

- [ ] **Step 2: Run tests**

```bash
python -m pytest tests/test_order_manager.py -v
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_order_manager.py
git commit -m "test: add order manager tests with mocked client and DB"
```

---

## Chunk 4: Alembic + Backtest + Dashboard

### Task 15: Alembic Setup

**Files:**
- Create: `alembic.ini`
- Create: `alembic/env.py`
- Create: `alembic/script.py.mako`
- Create: `alembic/versions/` (directory)

- [ ] **Step 1: Initialize Alembic**

```bash
cd C:/Users/youri/Desktop/trade
pip install alembic 2>/dev/null  # already in requirements.txt
python -m alembic init alembic
```

- [ ] **Step 2: Configure alembic.ini**

In `alembic.ini`, set sqlalchemy.url to empty (we'll use env.py to load it):

```ini
sqlalchemy.url =
```

- [ ] **Step 3: Configure alembic/env.py for async**

Replace `alembic/env.py` with:

```python
import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

from bot.config import get_settings
from bot.data.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.database_url)


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations():
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

- [ ] **Step 4: Generate initial migration** (requires running PostgreSQL)

```bash
python -m alembic revision --autogenerate -m "initial schema"
```

Note: this step requires the PostgreSQL DB to be running. If unavailable, create an empty migration and fill in later.

- [ ] **Step 5: Commit**

```bash
git add alembic/ alembic.ini
git commit -m "feat: add Alembic migration setup with async engine config"
```

---

### Task 16: Backtest Enhancement

**Files:**
- Modify: `bot/backtest/engine.py:419-424`
- Modify: `api/routers/backtest.py`

- [ ] **Step 1: Add max_open_positions param to run_backtest()**

In `bot/backtest/engine.py`, update `run_backtest()` signature (line ~419):

```python
async def run_backtest(
    symbol: str = "BTC-EUR",
    interval: str = "5m",
    days: int = 30,
    initial_capital: float = 10_000.0,
    max_open_positions: int = 3,
) -> BacktestResult:
```

And pass it through on line ~511:
```python
    engine = BacktestEngine(unique, initial_capital=initial_capital, max_open_positions=max_open_positions)
```

- [ ] **Step 2: Add query params to existing backtest endpoint**

In `api/routers/backtest.py`, update the endpoint:

```python
@router.get("/{symbol}")
async def backtest_symbol(
    symbol: str,
    interval: str = Query("5m", description="Candle interval"),
    days: int = Query(30, ge=1, le=365),
    initial_capital: float = Query(10_000.0, ge=100),
    max_open_positions: int = Query(3, ge=1, le=10),
):
    if "-" not in symbol:
        symbol = symbol[:3] + "-" + symbol[3:]
    symbol = symbol.upper()

    result = await run_backtest(
        symbol=symbol, interval=interval, days=days,
        initial_capital=initial_capital, max_open_positions=max_open_positions,
    )
    # ... rest unchanged
```

- [ ] **Step 3: Add compare endpoint**

Add to `api/routers/backtest.py`:

```python
import asyncio
from pydantic import BaseModel
from typing import List, Optional


class CompareRequest(BaseModel):
    symbols: List[str]
    days: int = 30
    interval: str = "5m"
    initial_capital: float = 10_000.0
    max_open_positions: int = 3


@router.post("/compare")
async def backtest_compare(req: CompareRequest):
    """Run backtests for multiple symbols with rate-limited concurrency."""
    sem = asyncio.Semaphore(2)

    async def run_one(symbol: str) -> dict:
        if "-" not in symbol:
            symbol = symbol[:3] + "-" + symbol[3:]
        symbol = symbol.upper()
        async with sem:
            try:
                result = await run_backtest(
                    symbol=symbol, interval=req.interval, days=req.days,
                    initial_capital=req.initial_capital,
                    max_open_positions=req.max_open_positions,
                )
                return {
                    "symbol": symbol,
                    "total_pnl": result.total_pnl,
                    "total_return_pct": result.total_return_pct,
                    "win_rate": result.win_rate,
                    "total_trades": result.total_trades,
                    "sharpe_ratio": result.sharpe_ratio,
                    "max_drawdown_pct": result.max_drawdown_pct,
                }
            except Exception as e:
                return {"symbol": symbol, "error": str(e)}

    tasks = [run_one(s) for s in req.symbols]
    results = await asyncio.gather(*tasks)

    return {
        "results": results,
        "params": {
            "days": req.days,
            "interval": req.interval,
            "initial_capital": req.initial_capital,
        },
    }
```

- [ ] **Step 4: Commit**

```bash
git add bot/backtest/engine.py api/routers/backtest.py
git commit -m "feat: add backtest compare endpoint and custom params support"
```

---

### Task 17: Dashboard — Risk Gate Card

**Files:**
- Modify: `frontend/index.html`
- Modify: `frontend/static/app.js`

- [ ] **Step 1: Add risk gate card to index.html**

In `frontend/index.html`, add a new card after the Strategy Performance card (after line ~191, before the closing `</aside>`):

```html
    <div class="card p-4">
      <div class="text-xs text-gray-500 mb-3">Risk Gate Log</div>
      <div id="risk-gate-log" class="flex flex-col gap-1 max-h-48 overflow-y-auto text-xs">
        <div class="text-gray-600 italic">No decisions yet</div>
      </div>
    </div>
```

- [ ] **Step 2: Add refresh logic to app.js**

Add to `frontend/static/app.js`:

```javascript
// ── Risk Gate Decisions ───────────────────────────────────────────────

async function refreshRiskGates() {
  try {
    const res = await fetch(`${API}/api/risk/decisions?limit=10`);
    const decisions = await res.json();
    const el = document.getElementById('risk-gate-log');
    if (!decisions.length) {
      el.innerHTML = '<div class="text-gray-600 italic">No decisions yet</div>';
      return;
    }
    el.innerHTML = decisions.map(d => {
      const time = d.timestamp ? d.timestamp.slice(11, 19) : '';
      const badge = d.approved
        ? '<span class="px-1.5 py-0.5 rounded bg-green-900 text-green-400">APPROVED</span>'
        : '<span class="px-1.5 py-0.5 rounded bg-red-900 text-red-400">REJECTED</span>';
      const failedGates = d.gate_details
        ? Object.entries(d.gate_details)
            .filter(([_, g]) => !g.passed)
            .map(([name, _]) => name.replace('_', ' '))
            .join(', ')
        : '';
      const gateInfo = failedGates ? `<span class="text-red-400 ml-1">${failedGates}</span>` : '';
      return `<div class="flex items-center gap-2 py-0.5 border-b border-gray-800">
        <span class="text-gray-600">${time}</span>
        ${badge}
        <span class="text-gray-400">${d.symbol || ''}</span>
        ${gateInfo}
      </div>`;
    }).join('');
  } catch (e) { console.error('refreshRiskGates', e); }
}
```

Add `refreshRiskGates()` to the `init()` Promise.all and to the 30s periodic refresh:

In init():
```javascript
    loadMarkets(), refreshStatus(), refreshPortfolio(), refreshAnalytics(),
    loadCandles(), loadEquityHistory(), refreshTrades(), refreshSentiment(),
    refreshStrategies(), refreshRiskGates(),
```

In the 30s interval:
```javascript
      refreshPortfolio(), refreshAnalytics(), loadEquityHistory(),
      refreshStrategies(), refreshRiskGates(),
```

- [ ] **Step 3: Add sentiment degradation to signal feed**

In the WebSocket `onmessage` handler in `app.js`, add a case for degradation:

```javascript
      if (type === 'sentiment.degraded') {
        addSignalFeedItem({direction: 'WARNING', symbol: d.source || '', strategy_name: `Sentiment ${d.status}`});
      }
```

- [ ] **Step 4: Commit**

```bash
git add frontend/index.html frontend/static/app.js
git commit -m "feat: add risk gate log card and sentiment degradation alerts to dashboard"
```

---

### Task 18: Final Integration + Verify

- [ ] **Step 1: Run full test suite**

```bash
cd C:/Users/youri/Desktop/trade
python -m pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 2: Verify imports work (no circular deps)**

```bash
python -c "from bot.main import main; print('OK')"
python -c "from api.app import create_app; print('OK')"
python -c "from bot.data_loader import CandleCache; print('OK')"
python -c "from bot.trading_loop import TradingLoop; print('OK')"
python -c "from bot.scheduler import BotScheduler; print('OK')"
python -c "from bot.notifications.discord import DiscordNotifier; print('OK')"
```

- [ ] **Step 3: Final commit**

```bash
git add -A
git status  # verify nothing unexpected
git commit -m "chore: final integration of all 10 improvements"
```
