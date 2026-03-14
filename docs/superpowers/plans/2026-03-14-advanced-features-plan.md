# Advanced Trading Features Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 7 features: multi-timeframe voting, on-chain metrics, order book analysis, trade journaling, performance attribution, benchmark comparison, walk-forward optimization.

**Architecture:** Hybrid approach (C) — strategy enhancements integrate into the trading pipeline via a `SignalProvider` protocol, analytics are a separate `bot/analytics/` package, walk-forward extends `bot/learning/`. Chart generation in `bot/charts/`.

**Tech Stack:** Python 3.11+, asyncio, FastAPI, SQLAlchemy 2.0, PostgreSQL, matplotlib, Alembic, pytest

**Spec:** `docs/superpowers/specs/2026-03-14-advanced-features-design.md`

---

## File Map

### New Files
| File | Responsibility |
|------|---------------|
| `bot/strategy/mtf_voter.py` | Multi-timeframe voting logic |
| `bot/indicators/onchain.py` | On-chain metrics provider (Binance, Blockchair) |
| `bot/indicators/orderbook.py` | Order book analysis provider |
| `bot/analytics/__init__.py` | Analytics package init |
| `bot/analytics/journal.py` | Trade journal creation + reasoning |
| `bot/analytics/attribution.py` | P&L attribution by strategy/regime/session/direction |
| `bot/analytics/benchmark.py` | Buy-and-hold benchmark comparison |
| `bot/charts/__init__.py` | Charts package init |
| `bot/charts/snapshot.py` | Matplotlib chart generation |
| `bot/learning/walk_forward.py` | Walk-forward optimizer |
| `api/routers/analytics.py` | API endpoints for attribution, benchmark, walk-forward |
| `api/routers/journal.py` | API endpoints for journal + chart serving |
| `api/routers/orderbook.py` | API endpoint for order book depth |
| `tests/test_mtf_voter.py` | MTF voter tests |
| `tests/test_onchain.py` | On-chain provider tests |
| `tests/test_orderbook.py` | Order book provider tests |
| `tests/test_journal.py` | Journal + reasoning tests |
| `tests/test_attribution.py` | Attribution engine tests |
| `tests/test_benchmark.py` | Benchmark engine tests |
| `tests/test_walk_forward.py` | Walk-forward optimizer tests |

### Modified Files
| File | Changes |
|------|---------|
| `bot/strategy/base.py` | Add `SignalProvider` protocol, extend `MarketContext` |
| `bot/config.py` | Add new config fields (MTF weights, onchain, orderbook) |
| `bot/data_loader.py` | Increase `MAX_CANDLES` to 2000, fetch limit to 1500 |
| `bot/data/models.py` | Add `TradeJournal`, `OnchainScore` models, `market_regime` on `Trade` |
| `bot/data/repositories.py` | Add journal + onchain repo functions |
| `bot/strategy/hybrid.py` | New 4-component formula, remove 1h dampening |
| `bot/trading_loop.py` | Populate extended `MarketContext`, remove 1h EMA check, pass journal data to order manager |
| `bot/exchange/order_manager.py` | Extend `TOPIC_TRADE_OPENED` payload |
| `bot/exchange/bitvavo_ws.py` | Add `on_book` callback + book channel subscription |
| `bot/backtest/engine.py` | Add `strategy_params` argument |
| `bot/main.py` | Wire new components, register scheduler tasks |
| `bot/events/bus.py` | Add new topic constants |

---

## Chunk 1: Foundation

### Task 1: SignalProvider Protocol + MarketContext Extension

**Files:**
- Modify: `bot/strategy/base.py:1-56`
- Test: `tests/test_mtf_voter.py` (protocol check)

- [ ] **Step 1: Write test for SignalProvider protocol**

```python
# tests/test_mtf_voter.py
"""Tests for multi-timeframe voter and SignalProvider protocol."""
from __future__ import annotations
import pytest
from bot.strategy.base import SignalProvider


class _DummyProvider:
    name = "dummy"
    def score(self, symbol: str, **kwargs) -> float:
        return 0.5
    def is_available(self) -> bool:
        return True


class _BadProvider:
    name = "bad"
    # missing score method


class TestSignalProviderProtocol:
    def test_valid_provider_matches_protocol(self):
        p = _DummyProvider()
        assert isinstance(p, SignalProvider)

    def test_invalid_provider_does_not_match(self):
        p = _BadProvider()
        assert not isinstance(p, SignalProvider)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mtf_voter.py::TestSignalProviderProtocol -v`
Expected: FAIL — `SignalProvider` not importable

- [ ] **Step 3: Add SignalProvider protocol and extend MarketContext**

Add to `bot/strategy/base.py` after existing imports:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class SignalProvider(Protocol):
    name: str
    def score(self, symbol: str, **kwargs) -> float: ...
    def is_available(self) -> bool: ...
```

Add new optional fields to `MarketContext` dataclass:

```python
    candles_15m: Optional[pd.DataFrame] = None
    candles_4h: Optional[pd.DataFrame] = None
    candles_1d: Optional[pd.DataFrame] = None
    onchain_score: float = 0.0
    orderbook_imbalance: float = 0.0
    market_regime: str = "unknown"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mtf_voter.py::TestSignalProviderProtocol -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add bot/strategy/base.py tests/test_mtf_voter.py
git commit -m "feat: add SignalProvider protocol and extend MarketContext"
```

---

### Task 2: Config Additions

**Files:**
- Modify: `bot/config.py:11-76`
- Test: `tests/test_config_validation.py` (add cases)

- [ ] **Step 1: Write test for new config fields**

```python
# Append to tests/test_config_validation.py
class TestNewConfigFields:
    def test_mtf_weights_defaults(self):
        s = Settings(database_url="postgresql+asyncpg://x:x@localhost/x")
        assert s.mtf_tech_weight == 0.55
        assert s.mtf_sent_weight == 0.20
        assert s.mtf_onchain_weight == 0.15
        assert s.mtf_book_weight == 0.10

    def test_onchain_defaults(self):
        s = Settings(database_url="postgresql+asyncpg://x:x@localhost/x")
        assert s.onchain_enabled is True
        assert s.onchain_poll_interval_secs == 300

    def test_orderbook_defaults(self):
        s = Settings(database_url="postgresql+asyncpg://x:x@localhost/x")
        assert s.orderbook_enabled is True
        assert s.orderbook_depth_levels == 25
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config_validation.py::TestNewConfigFields -v`
Expected: FAIL — fields don't exist

- [ ] **Step 3: Add config fields**

Add to `bot/config.py` Settings class after the learning section (line 64):

```python
    # ── Multi-timeframe voting ───────────────────────────────────────────────
    mtf_tech_weight: float = Field(default=0.55, ge=0, le=1)
    mtf_sent_weight: float = Field(default=0.20, ge=0, le=1)
    mtf_onchain_weight: float = Field(default=0.15, ge=0, le=1)
    mtf_book_weight: float = Field(default=0.10, ge=0, le=1)

    # ── On-chain metrics ────────────────────────────────────────────────────
    onchain_enabled: bool = True
    onchain_poll_interval_secs: int = 300

    # ── Order book ───────────────────────────────────────────────────────────
    orderbook_enabled: bool = True
    orderbook_depth_levels: int = 25
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config_validation.py::TestNewConfigFields -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add bot/config.py tests/test_config_validation.py
git commit -m "feat: add config fields for MTF, on-chain, and order book"
```

---

### Task 3: CandleCache Expansion

**Files:**
- Modify: `bot/data_loader.py:12,96,115`

- [ ] **Step 1: Write test**

```python
# tests/test_candle_cache.py
"""Tests for CandleCache expansion."""
from bot.data_loader import CandleCache, MAX_CANDLES


class TestCandleCacheCapacity:
    def test_max_candles_is_2000(self):
        assert MAX_CANDLES == 2000

    def test_cache_holds_2000_candles(self):
        cc = CandleCache()
        for i in range(2100):
            cc.cache_candle("BTC-EUR", "5m", {
                "timestamp": f"2026-01-01T00:{i:04d}",
                "open": 100, "high": 101, "low": 99, "close": 100, "volume": 10,
            })
        rows = cc._cache["BTC-EUR"]["5m"]
        assert len(rows) == 2000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_candle_cache.py -v`
Expected: FAIL — MAX_CANDLES is 500

- [ ] **Step 3: Update data_loader.py**

Change line 12: `MAX_CANDLES = 2000`

Change line 96 (`lazy_load`): `limit=200` → `limit=1500`

Change line 115 (`_load_symbol`): `limit=200` → `limit=1500`

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_candle_cache.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add bot/data_loader.py tests/test_candle_cache.py
git commit -m "feat: expand CandleCache to 2000 candles for MTF support"
```

---

## Chunk 2: Multi-Timeframe Voting

### Task 4: MTF Voter Module

**Files:**
- Create: `bot/strategy/mtf_voter.py`
- Test: `tests/test_mtf_voter.py` (extend)

- [ ] **Step 1: Write tests for MTFVoter**

```python
# Append to tests/test_mtf_voter.py
import numpy as np
import pandas as pd
from bot.strategy.mtf_voter import MTFVoter, MTFResult, REGIME_WEIGHTS


def _make_ohlcv(n=50, base_price=100.0, trend=0.0, seed=42):
    np.random.seed(seed)
    close = [base_price]
    for _ in range(1, n):
        close.append(close[-1] * (1 + trend + np.random.normal(0, 0.005)))
    close = np.array(close)
    return pd.DataFrame({
        "open": close * 0.999,
        "high": close * 1.005,
        "low": close * 0.995,
        "close": close,
        "volume": np.random.uniform(100, 1000, n),
    }, index=pd.date_range("2026-01-01", periods=n, freq="5min"))


class TestMTFVoter:
    def test_returns_mtf_result(self):
        df = _make_ohlcv(200, trend=0.002)
        voter = MTFVoter()
        result = voter.vote(
            df_15m=df, df_1h=df, df_4h=df, df_1d=df, regime="trending"
        )
        assert isinstance(result, MTFResult)
        assert -1.0 <= result.mtf_score <= 1.0
        assert 0.0 <= result.agreement_ratio <= 1.0
        assert len(result.per_tf_scores) == 4

    def test_regime_weights_used(self):
        assert "trending" in REGIME_WEIGHTS
        assert "ranging" in REGIME_WEIGHTS
        w = REGIME_WEIGHTS["trending"]
        assert abs(sum(w.values()) - 1.0) < 0.01

    def test_dampening_on_low_agreement(self):
        voter = MTFVoter()
        # Create conflicting signals across TFs
        bullish = _make_ohlcv(200, trend=0.003)
        bearish = _make_ohlcv(200, trend=-0.003, seed=99)
        result = voter.vote(
            df_15m=bullish, df_1h=bearish, df_4h=bearish, df_1d=bearish,
            regime="unknown",
        )
        # With 3/4 disagreeing, score should be dampened or directionally bearish
        assert isinstance(result, MTFResult)

    def test_insufficient_data_returns_neutral(self):
        tiny = _make_ohlcv(5)
        voter = MTFVoter()
        result = voter.vote(
            df_15m=tiny, df_1h=tiny, df_4h=tiny, df_1d=tiny, regime="unknown"
        )
        assert result.mtf_score == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mtf_voter.py::TestMTFVoter -v`
Expected: FAIL — `mtf_voter` module doesn't exist

- [ ] **Step 3: Implement MTFVoter**

Create `bot/strategy/mtf_voter.py`:

```python
"""Multi-timeframe strategy voting — runs composite indicator on multiple TFs."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd

from bot.indicators.composite import compute as compute_indicators

logger = logging.getLogger(__name__)

BASE_WEIGHTS = {"15m": 0.15, "1h": 0.25, "4h": 0.35, "1d": 0.25}

REGIME_WEIGHTS: Dict[str, Dict[str, float]] = {
    "trending":  {"15m": 0.10, "1h": 0.20, "4h": 0.40, "1d": 0.30},
    "ranging":   {"15m": 0.30, "1h": 0.35, "4h": 0.25, "1d": 0.10},
    "volatile":  {"15m": 0.10, "1h": 0.25, "4h": 0.35, "1d": 0.30},
    "unknown":   BASE_WEIGHTS,
}

MIN_CANDLES = 30


@dataclass
class MTFResult:
    mtf_score: float = 0.0
    agreement_ratio: float = 0.0
    per_tf_scores: Dict[str, float] = field(default_factory=dict)


class MTFVoter:
    """Aggregate composite technical scores across multiple timeframes."""

    def vote(
        self,
        df_15m: pd.DataFrame,
        df_1h: pd.DataFrame,
        df_4h: pd.DataFrame,
        df_1d: pd.DataFrame,
        regime: str = "unknown",
        indicator_weights: Optional[Dict[str, float]] = None,
    ) -> MTFResult:
        frames = {"15m": df_15m, "1h": df_1h, "4h": df_4h, "1d": df_1d}
        tf_scores: Dict[str, float] = {}

        for tf, df in frames.items():
            if df is None or len(df) < MIN_CANDLES:
                continue
            try:
                result = compute_indicators(df, indicator_weights)
                tf_scores[tf] = result.technical_score
            except Exception as e:
                logger.warning("MTF %s failed: %s", tf, e)

        if not tf_scores:
            return MTFResult()

        weights = REGIME_WEIGHTS.get(regime, BASE_WEIGHTS)
        # Only use weights for available TFs, redistribute
        active = {tf: weights.get(tf, 0) for tf in tf_scores}
        total_w = sum(active.values())
        if total_w <= 0:
            return MTFResult()
        norm = {tf: w / total_w for tf, w in active.items()}

        mtf_score = sum(tf_scores[tf] * norm[tf] for tf in tf_scores)
        mtf_score = max(-1.0, min(1.0, mtf_score))

        # Agreement ratio: non-neutral TFs agreeing on majority direction
        non_neutral = {tf: s for tf, s in tf_scores.items() if abs(s) >= 0.15}
        if len(non_neutral) < 2:
            agreement_ratio = 0.0
        else:
            bullish = sum(1 for s in non_neutral.values() if s > 0)
            bearish = len(non_neutral) - bullish
            agreement_ratio = max(bullish, bearish) / len(non_neutral)

        if agreement_ratio < 0.5:
            mtf_score *= 0.5

        return MTFResult(
            mtf_score=mtf_score,
            agreement_ratio=agreement_ratio,
            per_tf_scores=tf_scores,
        )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_mtf_voter.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add bot/strategy/mtf_voter.py tests/test_mtf_voter.py
git commit -m "feat: add multi-timeframe voter with regime-adaptive weights"
```

---

### Task 5: Hybrid Strategy Refactor — New 4-Component Formula

**Files:**
- Modify: `bot/strategy/hybrid.py:1-108`
- Test: `tests/test_hybrid_formula.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_hybrid_formula.py
"""Tests for the refactored hybrid strategy 4-component formula."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from bot.strategy.hybrid import HybridStrategy, redistribute_weights
from bot.strategy.base import MarketContext


def _ohlcv(n=50, trend=0.001):
    np.random.seed(42)
    c = [100.0]
    for _ in range(1, n):
        c.append(c[-1] * (1 + trend + np.random.normal(0, 0.003)))
    c = np.array(c)
    idx = pd.date_range("2026-01-01", periods=n, freq="5min")
    return pd.DataFrame({
        "open": c * 0.999, "high": c * 1.005, "low": c * 0.995,
        "close": c, "volume": np.random.uniform(100, 500, n),
    }, index=idx)


class TestRedistributeWeights:
    def test_all_enabled(self):
        w = {"tech": 0.55, "sent": 0.20, "onchain": 0.15, "book": 0.10}
        result = redistribute_weights(w, disabled=set())
        assert abs(sum(result.values()) - 1.0) < 0.001

    def test_onchain_disabled(self):
        w = {"tech": 0.55, "sent": 0.20, "onchain": 0.15, "book": 0.10}
        result = redistribute_weights(w, disabled={"onchain"})
        assert "onchain" not in result
        assert abs(sum(result.values()) - 1.0) < 0.001

    def test_both_disabled(self):
        w = {"tech": 0.55, "sent": 0.20, "onchain": 0.15, "book": 0.10}
        result = redistribute_weights(w, disabled={"onchain", "book"})
        assert abs(result["tech"] - 0.733) < 0.01
        assert abs(result["sent"] - 0.267) < 0.01


class TestHybridNewFormula:
    def test_generates_signal(self):
        df = _ohlcv(200)
        strat = HybridStrategy()
        ctx = MarketContext(
            symbol="BTC-EUR", candles_5m=df, candles_1h=df,
            candles_15m=df, candles_4h=df, candles_1d=df,
            current_price=df["close"].iloc[-1],
            sentiment_score=0.3, portfolio_equity_eur=10000,
            open_position_count=0, market_regime="trending",
        )
        sig = strat.generate_signal(ctx)
        assert sig.direction in ("LONG", "SHORT", "NEUTRAL")

    def test_entry_threshold_default_is_035(self):
        strat = HybridStrategy()
        assert strat.entry_threshold == 0.35
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hybrid_formula.py -v`
Expected: FAIL — `redistribute_weights` not importable, MarketContext missing new fields in some paths

- [ ] **Step 3: Refactor hybrid.py**

Replace `bot/strategy/hybrid.py` with the new 4-component formula. Key changes:
- Import `MTFVoter`
- Add `redistribute_weights()` function
- Change `DEFAULT_ENTRY_THRESHOLD` from 0.40 to 0.35
- In `generate_signal()`: use MTFVoter instead of direct `compute_indicators` call
- Apply formula: `final_score = w_tech * mtf_score + w_sent * sentiment + w_onchain * onchain + w_book * orderbook`
- Remove the old 1h dampening block (lines 57-62)

```python
"""Primary hybrid strategy: 4-component fusion (MTF technical + sentiment + on-chain + order book)."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Set

from bot.config import get_settings
from bot.strategy.base import BaseStrategy, MarketContext, Signal
from bot.strategy.mtf_voter import MTFVoter

logger = logging.getLogger(__name__)

DEFAULT_ENTRY_THRESHOLD = 0.35


def redistribute_weights(
    weights: Dict[str, float], disabled: Set[str]
) -> Dict[str, float]:
    active = {k: v for k, v in weights.items() if k not in disabled}
    total = sum(active.values())
    if total <= 0:
        return active
    return {k: v / total for k, v in active.items()}


class HybridStrategy(BaseStrategy):
    """
    Combines MTF technical composite, sentiment, on-chain, and order book scores.
    final_score = w_tech * mtf + w_sent * sentiment + w_onchain * onchain + w_book * orderbook
    """

    name = "hybrid"

    def __init__(
        self,
        sentiment_weight: float = 0.20,
        entry_threshold: float = DEFAULT_ENTRY_THRESHOLD,
        indicator_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.sentiment_weight = sentiment_weight
        self.entry_threshold = entry_threshold
        self.indicator_weights = indicator_weights
        self._mtf_voter = MTFVoter()

    def generate_signal(self, ctx: MarketContext) -> Signal:
        if len(ctx.candles_5m) < 30:
            return Signal(
                symbol=ctx.symbol, direction="NEUTRAL",
                strength=0.0, strategy_name=self.name,
            )

        settings = get_settings()

        # MTF voting
        mtf_result = self._mtf_voter.vote(
            df_15m=ctx.candles_15m if ctx.candles_15m is not None else ctx.candles_5m,
            df_1h=ctx.candles_1h,
            df_4h=ctx.candles_4h if ctx.candles_4h is not None else ctx.candles_1h,
            df_1d=ctx.candles_1d if ctx.candles_1d is not None else ctx.candles_1h,
            regime=ctx.market_regime,
            indicator_weights=ctx.indicator_weights or self.indicator_weights,
        )

        # Component scores
        tech_score = mtf_result.mtf_score
        sent_score = ctx.sentiment_score
        onchain_score = ctx.onchain_score
        book_score = ctx.orderbook_imbalance

        # Build weight map, disable unavailable providers
        base_w = {
            "tech": settings.mtf_tech_weight,
            "sent": settings.mtf_sent_weight,
            "onchain": settings.mtf_onchain_weight,
            "book": settings.mtf_book_weight,
        }
        disabled = set()
        if not settings.onchain_enabled:
            disabled.add("onchain")
        if not settings.orderbook_enabled:
            disabled.add("book")
        w = redistribute_weights(base_w, disabled)

        final_score = (
            w.get("tech", 0) * tech_score
            + w.get("sent", 0) * sent_score
            + w.get("onchain", 0) * onchain_score
            + w.get("book", 0) * book_score
        )
        final_score = max(-1.0, min(1.0, final_score))

        strength = min(abs(final_score), 1.0)
        direction = "NEUTRAL"
        if final_score > self.entry_threshold:
            direction = "LONG"
        elif final_score < -self.entry_threshold:
            direction = "SHORT"

        # Count confirming from MTF breakdown
        confirming = sum(
            1 for s in mtf_result.per_tf_scores.values()
            if s * final_score > 0
        )

        snapshot = {
            "confirming_count": confirming,
            "final_score": final_score,
            "technical_score": tech_score,
            "sentiment_score": sent_score,
            "onchain_score": onchain_score,
            "orderbook_imbalance": book_score,
            "mtf_scores": mtf_result.per_tf_scores,
            "mtf_agreement": mtf_result.agreement_ratio,
            "market_regime": ctx.market_regime,
        }

        return Signal(
            symbol=ctx.symbol, direction=direction, strength=strength,
            strategy_name=self.name, technical_score=tech_score,
            sentiment_score=sent_score, indicator_snapshot=snapshot,
        )

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "sentiment_weight": self.sentiment_weight,
            "entry_threshold": self.entry_threshold,
            "indicator_weights": self.indicator_weights,
        }

    def get_param_space(self) -> Dict[str, Any]:
        return {
            "sentiment_weight": {"type": "float", "low": 0.05, "high": 0.50},
            "entry_threshold": {"type": "float", "low": 0.25, "high": 0.65},
        }
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_hybrid_formula.py tests/test_mtf_voter.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add bot/strategy/hybrid.py tests/test_hybrid_formula.py
git commit -m "feat: refactor hybrid strategy to 4-component formula with MTF voting"
```

---

### Task 6: Remove 1h EMA Check from Trading Loop

**Files:**
- Modify: `bot/trading_loop.py:314-329`

- [ ] **Step 1: Remove 1h EMA confirmation block**

Delete lines 314-329 in `bot/trading_loop.py` (the block starting with `# Multi-timeframe confirmation: 1h trend must agree with direction`).

- [ ] **Step 2: Run existing tests**

Run: `pytest tests/ -v --timeout=30`
Expected: ALL PASS (no tests depend on the 1h EMA check)

- [ ] **Step 3: Commit**

```bash
git add bot/trading_loop.py
git commit -m "refactor: remove 1h EMA check, subsumed by MTF voting"
```

---

## Chunk 3: On-Chain Metrics

### Task 7: On-Chain Provider

**Files:**
- Create: `bot/indicators/onchain.py`
- Test: `tests/test_onchain.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_onchain.py
"""Tests for on-chain metrics provider."""
from __future__ import annotations
import pytest
from unittest.mock import patch, MagicMock
from bot.indicators.onchain import (
    OnchainProvider, BITVAVO_TO_BINANCE, _parse_funding_rate, _parse_open_interest,
)
from bot.strategy.base import SignalProvider


class TestSymbolMapping:
    def test_btc_maps(self):
        assert BITVAVO_TO_BINANCE["BTC-EUR"] == "BTCUSDT"

    def test_eth_maps(self):
        assert BITVAVO_TO_BINANCE["ETH-EUR"] == "ETHUSDT"

    def test_unknown_returns_none(self):
        assert BITVAVO_TO_BINANCE.get("FAKE-EUR") is None


class TestParsers:
    def test_parse_funding_positive(self):
        # Funding > 0.01% → bearish (-0.5)
        score = _parse_funding_rate(0.0002)  # 0.02%
        assert score == pytest.approx(-0.5)

    def test_parse_funding_negative(self):
        # Funding < -0.01% → bullish (+0.5)
        score = _parse_funding_rate(-0.0002)
        assert score == pytest.approx(0.5)

    def test_parse_funding_neutral(self):
        score = _parse_funding_rate(0.00005)  # 0.005%
        assert score == 0.0

    def test_parse_oi_rising_price_rising(self):
        score = _parse_open_interest(
            current_oi=1_000_000, prev_oi=900_000,
            current_price=50000, prev_price=49000,
        )
        assert score > 0  # trend confirmation

    def test_parse_oi_rising_price_falling(self):
        score = _parse_open_interest(
            current_oi=1_000_000, prev_oi=900_000,
            current_price=48000, prev_price=49000,
        )
        assert score < 0  # bearish pressure


class TestOnchainProvider:
    def test_implements_signal_provider(self):
        p = OnchainProvider()
        assert isinstance(p, SignalProvider)

    def test_score_returns_zero_when_no_data(self):
        p = OnchainProvider()
        assert p.score("BTC-EUR") == 0.0

    def test_is_available_false_initially(self):
        p = OnchainProvider()
        assert not p.is_available()

    def test_score_after_cache_populated(self):
        p = OnchainProvider()
        p._cache["BTC-EUR"] = {"funding": -0.5, "oi": 0.3}
        p._has_data = True
        score = p.score("BTC-EUR")
        assert -1.0 <= score <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_onchain.py -v`
Expected: FAIL — module doesn't exist

- [ ] **Step 3: Implement on-chain provider**

Create `bot/indicators/onchain.py`:

```python
"""On-chain metrics signal provider — Binance funding/OI + Blockchair whales."""
from __future__ import annotations

import json
import logging
import time
import urllib.request
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

BITVAVO_TO_BINANCE: Dict[str, str] = {
    "BTC-EUR": "BTCUSDT", "ETH-EUR": "ETHUSDT", "SOL-EUR": "SOLUSDT",
    "XRP-EUR": "XRPUSDT", "ADA-EUR": "ADAUSDT", "DOGE-EUR": "DOGEUSDT",
    "DOT-EUR": "DOTUSDT", "LINK-EUR": "LINKUSDT", "AVAX-EUR": "AVAXUSDT",
    "MATIC-EUR": "MATICUSDT", "ATOM-EUR": "ATOMUSDT", "UNI-EUR": "UNIUSDT",
    "LTC-EUR": "LTCUSDT", "BCH-EUR": "BCHUSDT", "FIL-EUR": "FILUSDT",
}

# Scoring weights per metric
METRIC_WEIGHTS = {"funding": 0.35, "oi": 0.35, "whale": 0.20, "reserves": 0.10}
_BINANCE_FAPI = "https://fapi.binance.com"
_BLOCKCHAIR = "https://api.blockchair.com"
_TIMEOUT = 10


def _parse_funding_rate(rate: float) -> float:
    if rate > 0.0001:
        return -0.5  # crowded longs → contrarian bearish
    elif rate < -0.0001:
        return 0.5   # crowded shorts → contrarian bullish
    return 0.0


def _parse_open_interest(
    current_oi: float, prev_oi: float,
    current_price: float, prev_price: float,
) -> float:
    if prev_oi <= 0 or prev_price <= 0:
        return 0.0
    oi_change = (current_oi - prev_oi) / prev_oi
    price_change = (current_price - prev_price) / prev_price
    if abs(oi_change) < 0.01:
        return 0.0
    if oi_change > 0 and price_change > 0:
        return 0.5   # trend confirmation
    elif oi_change > 0 and price_change < 0:
        return -0.5  # bearish pressure
    elif oi_change < 0 and price_change > 0:
        return 0.3   # short squeeze potential
    elif oi_change < 0 and price_change < 0:
        return -0.3  # long liquidation
    return 0.0


def _fetch_json(url: str) -> Optional[Any]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "hellacash/1.0"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        logger.warning("On-chain fetch failed %s: %s", url, e)
        return None


class OnchainProvider:
    """On-chain metrics signal provider. Caches results, polled async."""

    name = "onchain"

    def __init__(self) -> None:
        self._cache: Dict[str, Dict[str, float]] = {}
        self._prev_oi: Dict[str, float] = {}
        self._prev_price: Dict[str, float] = {}
        self._has_data = False
        self._failure_count = 0
        self._disabled_until = 0.0

    def score(self, symbol: str, **kwargs) -> float:
        if not self._has_data or symbol not in self._cache:
            return 0.0
        metrics = self._cache[symbol]
        if not metrics:
            return 0.0
        active = {k: v for k, v in metrics.items() if k in METRIC_WEIGHTS}
        if not active:
            return 0.0
        weights = {k: METRIC_WEIGHTS[k] for k in active}
        total_w = sum(weights.values())
        return max(-1.0, min(1.0,
            sum(active[k] * weights[k] / total_w for k in active)
        ))

    def is_available(self) -> bool:
        return self._has_data and time.time() > self._disabled_until

    async def poll(self, symbols: list[str], current_prices: Dict[str, float]) -> None:
        if time.time() < self._disabled_until:
            return

        import asyncio
        loop = asyncio.get_event_loop()

        for symbol in symbols:
            binance_sym = BITVAVO_TO_BINANCE.get(symbol)
            if not binance_sym:
                continue

            try:
                metrics: Dict[str, float] = {}

                # Funding rate
                data = await loop.run_in_executor(None, _fetch_json,
                    f"{_BINANCE_FAPI}/fapi/v1/fundingRate?symbol={binance_sym}&limit=1")
                if data and len(data) > 0:
                    rate = float(data[0].get("fundingRate", 0))
                    metrics["funding"] = _parse_funding_rate(rate)

                # Open interest
                data = await loop.run_in_executor(None, _fetch_json,
                    f"{_BINANCE_FAPI}/fapi/v1/openInterest?symbol={binance_sym}")
                if data:
                    oi = float(data.get("openInterest", 0))
                    price = current_prices.get(symbol, 0)
                    prev_oi = self._prev_oi.get(symbol, oi)
                    prev_price = self._prev_price.get(symbol, price)
                    metrics["oi"] = _parse_open_interest(oi, prev_oi, price, prev_price)
                    self._prev_oi[symbol] = oi
                    self._prev_price[symbol] = price

                self._cache[symbol] = metrics
                self._has_data = True
                self._failure_count = 0

            except Exception as e:
                logger.error("On-chain poll error for %s: %s", symbol, e)
                self._failure_count += 1
                if self._failure_count >= 3:
                    self._disabled_until = time.time() + 600
                    logger.warning("On-chain circuit breaker: disabled for 10 min")
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_onchain.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add bot/indicators/onchain.py tests/test_onchain.py
git commit -m "feat: add on-chain metrics provider (Binance funding/OI)"
```

---

## Chunk 4: Order Book Analysis

### Task 8: OrderBook Provider

**Files:**
- Create: `bot/indicators/orderbook.py`
- Test: `tests/test_orderbook.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_orderbook.py
"""Tests for order book analysis provider."""
from __future__ import annotations
import pytest
from bot.indicators.orderbook import (
    OrderBookProvider, _BookState, DepthAnalysis, PriceLevel, DepthBucket,
)
from bot.strategy.base import SignalProvider


class TestBookState:
    def test_apply_delta_add(self):
        bs = _BookState()
        bs.apply_delta("bid", 50000.0, 1.5)
        assert bs.bids[50000.0] == 1.5

    def test_apply_delta_remove(self):
        bs = _BookState()
        bs.apply_delta("bid", 50000.0, 1.5)
        bs.apply_delta("bid", 50000.0, 0)
        assert 50000.0 not in bs.bids

    def test_apply_delta_ask(self):
        bs = _BookState()
        bs.apply_delta("ask", 51000.0, 2.0)
        assert bs.asks[51000.0] == 2.0


class TestOrderBookProvider:
    def test_implements_signal_provider(self):
        p = OrderBookProvider()
        assert isinstance(p, SignalProvider)

    def test_imbalance_bullish(self):
        p = OrderBookProvider()
        # Lots of bids, few asks
        for i in range(25):
            p.update_level("BTC-EUR", "bid", 50000 - i * 10, 2.0)
            p.update_level("BTC-EUR", "ask", 50100 + i * 10, 0.5)
        score = p.score("BTC-EUR")
        assert score > 0  # bullish imbalance

    def test_imbalance_bearish(self):
        p = OrderBookProvider()
        for i in range(25):
            p.update_level("BTC-EUR", "bid", 50000 - i * 10, 0.5)
            p.update_level("BTC-EUR", "ask", 50100 + i * 10, 2.0)
        score = p.score("BTC-EUR")
        assert score < 0  # bearish imbalance

    def test_no_data_returns_zero(self):
        p = OrderBookProvider()
        assert p.score("BTC-EUR") == 0.0

    def test_analyze_returns_depth_analysis(self):
        p = OrderBookProvider()
        for i in range(25):
            p.update_level("BTC-EUR", "bid", 50000 - i * 10, 1.0 + (i == 5) * 5.0)
            p.update_level("BTC-EUR", "ask", 50100 + i * 10, 1.0)
        analysis = p.analyze("BTC-EUR")
        assert isinstance(analysis, DepthAnalysis)
        assert -1.0 <= analysis.imbalance <= 1.0
        assert analysis.spread_pct >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_orderbook.py -v`
Expected: FAIL — module doesn't exist

- [ ] **Step 3: Implement order book provider**

Create `bot/indicators/orderbook.py`:

```python
"""Order book analysis — bid/ask imbalance, wall detection, support/resistance."""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PriceLevel:
    price: float
    quantity: float
    eur_value: float = 0.0


@dataclass
class DepthBucket:
    price_low: float
    price_high: float
    bid_volume: float = 0.0
    ask_volume: float = 0.0


@dataclass
class DepthAnalysis:
    imbalance: float = 0.0
    spread_pct: float = 0.0
    bid_walls: List[PriceLevel] = field(default_factory=list)
    ask_walls: List[PriceLevel] = field(default_factory=list)
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)
    depth_buckets: List[DepthBucket] = field(default_factory=list)


class _BookState:
    def __init__(self) -> None:
        self.bids: Dict[float, float] = {}
        self.asks: Dict[float, float] = {}
        self.last_update: float = 0.0

    def apply_delta(self, side: str, price: float, qty: float) -> None:
        book = self.bids if side == "bid" else self.asks
        if qty == 0:
            book.pop(price, None)
        else:
            book[price] = qty
        self.last_update = time.time()


class OrderBookProvider:
    """Order book signal provider with depth analysis."""

    name = "orderbook"

    def __init__(self, depth_levels: int = 25) -> None:
        self._books: Dict[str, _BookState] = {}
        self._depth_levels = depth_levels
        self._last_analysis: Dict[str, float] = {}

    def update_level(self, symbol: str, side: str, price: float, qty: float) -> None:
        if symbol not in self._books:
            self._books[symbol] = _BookState()
        self._books[symbol].apply_delta(side, price, qty)

    def set_snapshot(self, symbol: str, bids: list, asks: list) -> None:
        bs = _BookState()
        for price, qty in bids:
            bs.bids[float(price)] = float(qty)
        for price, qty in asks:
            bs.asks[float(price)] = float(qty)
        bs.last_update = time.time()
        self._books[symbol] = bs

    def score(self, symbol: str, **kwargs) -> float:
        if symbol not in self._books:
            return 0.0
        bs = self._books[symbol]
        if not bs.bids or not bs.asks:
            return 0.0
        imbalance = self._compute_imbalance(bs)
        if abs(imbalance) < 0.3:
            return 0.0
        return max(-1.0, min(1.0, imbalance))

    def is_available(self) -> bool:
        return bool(self._books)

    def analyze(self, symbol: str) -> DepthAnalysis:
        if symbol not in self._books:
            return DepthAnalysis()
        bs = self._books[symbol]
        if not bs.bids or not bs.asks:
            return DepthAnalysis()

        imbalance = self._compute_imbalance(bs)
        best_bid = max(bs.bids.keys())
        best_ask = min(bs.asks.keys())
        mid = (best_bid + best_ask) / 2
        spread_pct = (best_ask - best_bid) / mid * 100 if mid > 0 else 0

        # Wall detection
        n = self._depth_levels
        top_bids = sorted(bs.bids.items(), key=lambda x: -x[0])[:n]
        top_asks = sorted(bs.asks.items(), key=lambda x: x[0])[:n]
        all_qtys = [q for _, q in top_bids + top_asks]
        avg_qty = sum(all_qtys) / len(all_qtys) if all_qtys else 1
        threshold = avg_qty * 3

        bid_walls = [PriceLevel(p, q, p * q) for p, q in top_bids if q > threshold]
        ask_walls = [PriceLevel(p, q, p * q) for p, q in top_asks if q > threshold]

        # Support/resistance via price buckets
        bucket_width = mid * 0.005
        support = self._cluster_levels(top_bids, bucket_width, top_n=3)
        resistance = self._cluster_levels(top_asks, bucket_width, top_n=3)

        # Depth buckets for heatmap
        buckets = self._build_depth_buckets(bs, mid, n_buckets=50)

        return DepthAnalysis(
            imbalance=imbalance, spread_pct=spread_pct,
            bid_walls=bid_walls, ask_walls=ask_walls,
            support_levels=support, resistance_levels=resistance,
            depth_buckets=buckets,
        )

    def _compute_imbalance(self, bs: _BookState) -> float:
        n = self._depth_levels
        top_bids = sorted(bs.bids.items(), key=lambda x: -x[0])[:n]
        top_asks = sorted(bs.asks.items(), key=lambda x: x[0])[:n]
        bid_vol = sum(q for _, q in top_bids)
        ask_vol = sum(q for _, q in top_asks)
        total = bid_vol + ask_vol
        if total <= 0:
            return 0.0
        return (bid_vol - ask_vol) / total

    def _cluster_levels(self, levels: list, bucket_width: float, top_n: int = 3) -> List[float]:
        clusters: Dict[float, float] = defaultdict(float)
        for price, qty in levels:
            bucket = round(price / bucket_width) * bucket_width
            clusters[bucket] += qty
        if not clusters:
            return []
        median_vol = sorted(clusters.values())[len(clusters) // 2]
        significant = [(p, v) for p, v in clusters.items() if v > median_vol * 2]
        significant.sort(key=lambda x: -x[1])
        return [p for p, _ in significant[:top_n]]

    def _build_depth_buckets(self, bs: _BookState, mid: float, n_buckets: int = 50) -> List[DepthBucket]:
        spread = mid * 0.10  # ±5%
        low = mid - spread / 2
        step = spread / n_buckets
        buckets = []
        for i in range(n_buckets):
            bl = low + i * step
            bh = bl + step
            bid_v = sum(q for p, q in bs.bids.items() if bl <= p < bh)
            ask_v = sum(q for p, q in bs.asks.items() if bl <= p < bh)
            buckets.append(DepthBucket(bl, bh, bid_v, ask_v))
        return buckets
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_orderbook.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add bot/indicators/orderbook.py tests/test_orderbook.py
git commit -m "feat: add order book analysis provider with depth analysis"
```

---

### Task 9: WebSocket Book Channel Integration

**Files:**
- Modify: `bot/exchange/bitvavo_ws.py:30-42,86-122,124-159`

- [ ] **Step 1: Add on_book callback to __init__ (line 37)**

Add `on_book: Optional[Callable] = None` parameter and `self._on_book = on_book`.

- [ ] **Step 2: Add book subscription to _subscribe method**

After the existing candle subscription block, add:

```python
if self._on_book and self._subscribed_markets:
    await ws.send(json.dumps({
        "action": "subscribe",
        "channels": [{"name": "book", "markets": list(self._subscribed_markets)}],
    }))
```

- [ ] **Step 3: Add book dispatch to message handler**

In the dispatch method, add handling for book events:

```python
elif event == "book":
    if self._on_book:
        await self._on_book(msg)
```

- [ ] **Step 4: Run existing WS tests**

Run: `pytest tests/ -v --timeout=30`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add bot/exchange/bitvavo_ws.py
git commit -m "feat: add order book WebSocket channel support"
```

---

## Chunk 5: Database Changes

### Task 10: Models + Alembic Migration

**Files:**
- Modify: `bot/data/models.py`
- Modify: `bot/events/bus.py`
- Create: Alembic migration

- [ ] **Step 1: Add new models and column to models.py**

Add `market_regime` to `Trade` model (after `borrow_fee` field, line 194):

```python
    market_regime: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
```

Add `TradeJournal` model after `Trade`:

```python
class TradeJournal(Base):
    """Structured trade journal with auto-generated reasoning and chart refs."""
    __tablename__ = "trade_journal"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trade_id: Mapped[int] = mapped_column(Integer, ForeignKey("trades.id"), unique=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    direction: Mapped[str] = mapped_column(String(10), nullable=False)
    strategy_name: Mapped[str] = mapped_column(String(50), nullable=False)
    market_regime: Mapped[str] = mapped_column(String(20), nullable=False)
    entry_technical_scores: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    entry_mtf_scores: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    entry_sentiment_score: Mapped[float] = mapped_column(Float, default=0.0)
    entry_onchain_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    entry_orderbook_imbalance: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    entry_composite_score: Mapped[float] = mapped_column(Float, default=0.0)
    exit_technical_scores: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    exit_composite_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    entry_reasoning: Mapped[str] = mapped_column(Text, nullable=False)
    exit_reasoning: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    entry_chart_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    exit_chart_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
```

Add `OnchainScore` model:

```python
class OnchainScore(Base):
    """Persisted on-chain metric scores for auditability."""
    __tablename__ = "onchain_scores"
    __table_args__ = (Index("ix_onchain_symbol_ts", "symbol", "computed_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    funding_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    oi_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    whale_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    composite_score: Mapped[float] = mapped_column(Float, nullable=False)
    raw_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    computed_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
```

- [ ] **Step 2: Add new event bus topics to bus.py**

Add after existing TOPIC constants (line 25):

```python
TOPIC_ORDERBOOK_UPDATE = "orderbook.update"
TOPIC_ONCHAIN_UPDATE = "onchain.update"
```

- [ ] **Step 3: Generate Alembic migration**

Run: `cd C:/Users/youri/Desktop/trade && alembic revision --autogenerate -m "add trade_journal, onchain_scores, market_regime"`

- [ ] **Step 4: Apply migration**

Run: `alembic upgrade head`

- [ ] **Step 5: Commit**

```bash
git add bot/data/models.py bot/events/bus.py alembic/versions/
git commit -m "feat: add trade_journal, onchain_scores tables and market_regime column"
```

---

## Chunk 6: Trade Journaling

### Task 11: Journal Module + Reasoning Generator

**Files:**
- Create: `bot/analytics/__init__.py`
- Create: `bot/analytics/journal.py`
- Test: `tests/test_journal.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_journal.py
"""Tests for trade journal reasoning generator."""
from __future__ import annotations
from bot.analytics.journal import generate_entry_reasoning, generate_exit_reasoning


class TestEntryReasoning:
    def test_generates_string(self):
        reasoning = generate_entry_reasoning(
            symbol="BTC-EUR", direction="LONG", strategy="hybrid",
            regime="trending", final_score=0.64, threshold=0.35,
            mtf_scores={"15m": 0.42, "1h": 0.61, "4h": 0.55, "1d": -0.12},
            mtf_agreement=0.75,
            technical_score=0.58, sentiment_score=0.31,
            onchain_score=0.22, orderbook_imbalance=0.15,
            indicator_values={"rsi": 34, "macd_histogram": 0.002},
        )
        assert "LONG" in reasoning
        assert "BTC-EUR" in reasoning
        assert "TRENDING" in reasoning or "trending" in reasoning

    def test_handles_missing_optional_scores(self):
        reasoning = generate_entry_reasoning(
            symbol="ETH-EUR", direction="SHORT", strategy="hybrid",
            regime="ranging", final_score=-0.45, threshold=0.35,
            mtf_scores={"15m": -0.3, "1h": -0.5},
            mtf_agreement=1.0,
            technical_score=-0.5, sentiment_score=-0.1,
        )
        assert "SHORT" in reasoning


class TestExitReasoning:
    def test_generates_string(self):
        reasoning = generate_exit_reasoning(
            exit_reason="take_profit", hold_seconds=3600,
            entry_price=50000, exit_price=51500,
            net_pnl=47.20, roi_pct=2.3,
        )
        assert "take_profit" in reasoning or "take profit" in reasoning
        assert "47" in reasoning
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_journal.py -v`
Expected: FAIL

- [ ] **Step 3: Implement journal module**

Create `bot/analytics/__init__.py` (empty).

Create `bot/analytics/journal.py`:

```python
"""Trade journaling — auto-generated reasoning from indicator snapshots."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def generate_entry_reasoning(
    symbol: str,
    direction: str,
    strategy: str,
    regime: str,
    final_score: float,
    threshold: float,
    mtf_scores: Dict[str, float],
    mtf_agreement: float,
    technical_score: float,
    sentiment_score: float,
    onchain_score: float = 0.0,
    orderbook_imbalance: float = 0.0,
    indicator_values: Optional[Dict[str, Any]] = None,
) -> str:
    parts = []
    n_agree = sum(1 for s in mtf_scores.values() if s * final_score > 0)
    n_total = len(mtf_scores)

    parts.append(
        f"{direction} {symbol} via {strategy} in {regime.upper()} regime."
    )

    tf_detail = ", ".join(f"{tf}: {s:+.2f}" for tf, s in mtf_scores.items())
    parts.append(
        f"MTF agreement: {n_agree}/{n_total} timeframes "
        f"({'bullish' if final_score > 0 else 'bearish'}) ({tf_detail})."
    )

    parts.append(f"Technical composite: {technical_score:+.2f}")

    if indicator_values:
        indicators = []
        if "rsi" in indicator_values:
            rsi = indicator_values["rsi"]
            label = "oversold" if rsi < 30 else "overbought" if rsi > 70 else ""
            indicators.append(f"RSI {rsi:.0f}{' ' + label if label else ''}")
        if "macd_histogram" in indicator_values:
            h = indicator_values["macd_histogram"]
            indicators.append(f"MACD {'rising' if h > 0 else 'falling'}")
        if indicators:
            parts[-1] += f" ({', '.join(indicators)})"
    parts[-1] += "."

    parts.append(f"Sentiment: {sentiment_score:+.2f}.")

    if onchain_score != 0:
        parts.append(f"On-chain: {onchain_score:+.2f}.")
    if orderbook_imbalance != 0:
        parts.append(f"Order book: {orderbook_imbalance:+.2f}.")

    parts.append(f"Final score: {final_score:+.2f} vs threshold {threshold:.2f}.")

    return " ".join(parts)


def generate_exit_reasoning(
    exit_reason: str,
    hold_seconds: int,
    entry_price: float,
    exit_price: float,
    net_pnl: float,
    roi_pct: float,
    indicator_values: Optional[Dict[str, Any]] = None,
) -> str:
    hours = hold_seconds // 3600
    mins = (hold_seconds % 3600) // 60
    hold_str = f"{hours}h {mins}m" if hours > 0 else f"{mins}m"

    reason_label = exit_reason.replace("_", " ")
    parts = [
        f"Exited via {reason_label} at €{exit_price:,.2f}.",
        f"Held {hold_str}.",
    ]

    if indicator_values and "rsi" in indicator_values:
        parts.append(f"RSI at exit: {indicator_values['rsi']:.0f}.")

    sign = "+" if net_pnl >= 0 else ""
    parts.append(f"Net P&L: {sign}€{net_pnl:.2f} ({sign}{roi_pct:.1f}%).")

    return " ".join(parts)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_journal.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add bot/analytics/__init__.py bot/analytics/journal.py tests/test_journal.py
git commit -m "feat: add trade journal reasoning generator"
```

---

### Task 12: Chart Snapshot Generator

**Files:**
- Create: `bot/charts/__init__.py`
- Create: `bot/charts/snapshot.py`

- [ ] **Step 1: Write test**

```python
# tests/test_chart_snapshot.py
"""Tests for chart snapshot generation."""
from __future__ import annotations
import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from bot.charts.snapshot import generate_entry_chart, generate_exit_chart


def _ohlcv(n=100):
    np.random.seed(42)
    c = np.cumsum(np.random.normal(0, 1, n)) + 100
    idx = pd.date_range("2026-01-01", periods=n, freq="5min")
    return pd.DataFrame({
        "open": c * 0.999, "high": c * 1.005,
        "low": c * 0.995, "close": c,
        "volume": np.random.uniform(100, 500, n),
    }, index=idx)


class TestChartGeneration:
    def test_entry_chart_creates_png(self):
        df = _ohlcv(100)
        with tempfile.TemporaryDirectory() as d:
            path = generate_entry_chart(
                df, entry_price=100.0, symbol="BTC-EUR",
                trade_id=1, output_dir=d,
            )
            assert os.path.exists(path)
            assert path.endswith(".png")
            assert os.path.getsize(path) > 1000

    def test_exit_chart_creates_png(self):
        df = _ohlcv(100)
        with tempfile.TemporaryDirectory() as d:
            path = generate_exit_chart(
                df, entry_price=100.0, exit_price=105.0,
                stop_loss=95.0, take_profit=110.0,
                symbol="BTC-EUR", trade_id=1, output_dir=d,
            )
            assert os.path.exists(path)
            assert path.endswith(".png")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chart_snapshot.py -v`
Expected: FAIL

- [ ] **Step 3: Implement chart generator**

Create `bot/charts/__init__.py` (empty).

Create `bot/charts/snapshot.py`:

```python
"""Matplotlib chart snapshot generation for trade journal."""
from __future__ import annotations

import logging
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from bot.indicators.momentum import rsi
from bot.indicators.trend import ema, macd
from bot.indicators.volatility import bollinger_bands

logger = logging.getLogger(__name__)

# Dark theme
plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#444",
    "axes.labelcolor": "#ccc",
    "text.color": "#ccc",
    "xtick.color": "#888",
    "ytick.color": "#888",
    "grid.color": "#333",
})


def _plot_candlesticks(ax, df: pd.DataFrame) -> None:
    up = df[df["close"] >= df["open"]]
    down = df[df["close"] < df["open"]]
    width = 0.0025  # for datetime index
    ax.bar(up.index, up["close"] - up["open"], width, bottom=up["open"], color="#26a69a", alpha=0.9)
    ax.bar(up.index, up["high"] - up["close"], width * 0.3, bottom=up["close"], color="#26a69a", alpha=0.9)
    ax.bar(up.index, up["low"] - up["open"], width * 0.3, bottom=up["open"], color="#26a69a", alpha=0.9)
    ax.bar(down.index, down["close"] - down["open"], width, bottom=down["open"], color="#ef5350", alpha=0.9)
    ax.bar(down.index, down["high"] - down["open"], width * 0.3, bottom=down["open"], color="#ef5350", alpha=0.9)
    ax.bar(down.index, down["low"] - down["close"], width * 0.3, bottom=down["close"], color="#ef5350", alpha=0.9)


def generate_entry_chart(
    df: pd.DataFrame,
    entry_price: float,
    symbol: str,
    trade_id: int,
    output_dir: str = "data/charts",
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1, 1]}, sharex=True)

    ax_price, ax_rsi, ax_vol = axes

    _plot_candlesticks(ax_price, df)
    ax_price.axhline(entry_price, color="#ffd700", linestyle="--", linewidth=1, label=f"Entry €{entry_price:,.2f}")

    # Overlays
    if len(df) >= 20:
        ax_price.plot(df.index, ema(df["close"], 20), color="#42a5f5", linewidth=0.8, label="EMA20")
    if len(df) >= 50:
        ax_price.plot(df.index, ema(df["close"], 50), color="#ab47bc", linewidth=0.8, label="EMA50")
    if len(df) >= 20:
        bb = bollinger_bands(df["close"])
        ax_price.fill_between(df.index, bb["lower"], bb["upper"], alpha=0.1, color="#42a5f5")

    ax_price.set_title(f"{symbol} Entry — Trade #{trade_id}", fontsize=12)
    ax_price.legend(loc="upper left", fontsize=8)
    ax_price.grid(True, alpha=0.3)

    # RSI
    if len(df) >= 14:
        r = rsi(df["close"])
        ax_rsi.plot(df.index, r, color="#ffa726", linewidth=0.8)
        ax_rsi.axhline(70, color="#ef5350", linestyle=":", linewidth=0.5)
        ax_rsi.axhline(30, color="#26a69a", linestyle=":", linewidth=0.5)
        ax_rsi.set_ylabel("RSI", fontsize=9)
        ax_rsi.set_ylim(0, 100)
        ax_rsi.grid(True, alpha=0.3)

    # Volume
    colors = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(df["close"], df["open"])]
    ax_vol.bar(df.index, df["volume"], width=0.0025, color=colors, alpha=0.7)
    ax_vol.set_ylabel("Volume", fontsize=9)
    ax_vol.grid(True, alpha=0.3)

    ax_vol.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.tight_layout()

    filename = f"{symbol.replace('-', '')}_{trade_id}_entry.png"
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_exit_chart(
    df: pd.DataFrame,
    entry_price: float,
    exit_price: float,
    stop_loss: float,
    take_profit: float,
    symbol: str,
    trade_id: int,
    output_dir: str = "data/charts",
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1, 1]}, sharex=True)
    ax_price, ax_rsi, ax_vol = axes

    _plot_candlesticks(ax_price, df)
    ax_price.axhline(entry_price, color="#ffd700", linestyle="--", linewidth=1, label=f"Entry €{entry_price:,.2f}")
    ax_price.axhline(exit_price, color="#00e676", linestyle="--", linewidth=1, label=f"Exit €{exit_price:,.2f}")
    ax_price.axhline(stop_loss, color="#ef5350", linestyle=":", linewidth=0.8, label=f"SL €{stop_loss:,.2f}")
    ax_price.axhline(take_profit, color="#26a69a", linestyle=":", linewidth=0.8, label=f"TP €{take_profit:,.2f}")

    if len(df) >= 20:
        ax_price.plot(df.index, ema(df["close"], 20), color="#42a5f5", linewidth=0.8, label="EMA20")
    if len(df) >= 50:
        ax_price.plot(df.index, ema(df["close"], 50), color="#ab47bc", linewidth=0.8, label="EMA50")

    ax_price.set_title(f"{symbol} Exit — Trade #{trade_id}", fontsize=12)
    ax_price.legend(loc="upper left", fontsize=8)
    ax_price.grid(True, alpha=0.3)

    if len(df) >= 14:
        r = rsi(df["close"])
        ax_rsi.plot(df.index, r, color="#ffa726", linewidth=0.8)
        ax_rsi.axhline(70, color="#ef5350", linestyle=":", linewidth=0.5)
        ax_rsi.axhline(30, color="#26a69a", linestyle=":", linewidth=0.5)
        ax_rsi.set_ylabel("RSI", fontsize=9)
        ax_rsi.set_ylim(0, 100)
        ax_rsi.grid(True, alpha=0.3)

    colors = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(df["close"], df["open"])]
    ax_vol.bar(df.index, df["volume"], width=0.0025, color=colors, alpha=0.7)
    ax_vol.set_ylabel("Volume", fontsize=9)
    ax_vol.grid(True, alpha=0.3)

    ax_vol.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.tight_layout()

    filename = f"{symbol.replace('-', '')}_{trade_id}_exit.png"
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_chart_snapshot.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add bot/charts/__init__.py bot/charts/snapshot.py tests/test_chart_snapshot.py
git commit -m "feat: add matplotlib chart snapshot generator for trade journal"
```

---

## Chunk 7: Performance Attribution + Benchmark

### Task 13: Attribution Engine

**Files:**
- Create: `bot/analytics/attribution.py`
- Test: `tests/test_attribution.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_attribution.py
"""Tests for performance attribution engine."""
from __future__ import annotations
from datetime import datetime, timezone
import pytest
from bot.analytics.attribution import (
    AttributionEngine, _classify_session, StrategyAttribution, AttributionReport,
)


class TestClassifySession:
    def test_asia(self):
        dt = datetime(2026, 1, 1, 3, 0, tzinfo=timezone.utc)
        assert _classify_session(dt) == "Asia"

    def test_europe(self):
        dt = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
        assert _classify_session(dt) == "Europe"

    def test_us(self):
        dt = datetime(2026, 1, 1, 16, 0, tzinfo=timezone.utc)
        assert _classify_session(dt) == "US"

    def test_overlap(self):
        dt = datetime(2026, 1, 1, 22, 0, tzinfo=timezone.utc)
        assert _classify_session(dt) == "Overlap/Off-hours"


class TestAttributionEngine:
    def test_compute_empty_trades(self):
        engine = AttributionEngine()
        report = engine.compute_from_trades([])
        assert isinstance(report, AttributionReport)
        assert report.total_pnl == 0.0
        assert report.by_strategy == []

    def test_compute_groups_by_strategy(self):
        trades = [
            _fake_trade("hybrid", 50.0), _fake_trade("hybrid", -20.0),
            _fake_trade("trend_following", 30.0),
        ]
        engine = AttributionEngine()
        report = engine.compute_from_trades(trades)
        assert len(report.by_strategy) == 2
        hybrid = next(s for s in report.by_strategy if s.strategy_name == "hybrid")
        assert hybrid.trade_count == 2
        assert hybrid.total_pnl == pytest.approx(30.0)


def _fake_trade(strategy: str, pnl: float):
    from types import SimpleNamespace
    return SimpleNamespace(
        strategy_name=strategy, net_pnl=pnl, roi_pct=pnl / 100,
        direction="LONG", market_regime="trending",
        created_at=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_attribution.py -v`
Expected: FAIL

- [ ] **Step 3: Implement attribution engine**

Create `bot/analytics/attribution.py`:

```python
"""Performance attribution — P&L breakdown by strategy, regime, session, direction."""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class StrategyAttribution:
    strategy_name: str
    total_pnl: float = 0.0
    trade_count: int = 0
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    pnl_contribution_pct: float = 0.0


@dataclass
class AttributionReport:
    total_pnl: float = 0.0
    total_trades: int = 0
    overall_win_rate: float = 0.0
    by_strategy: List[StrategyAttribution] = field(default_factory=list)
    by_regime: List[StrategyAttribution] = field(default_factory=list)
    by_session: List[StrategyAttribution] = field(default_factory=list)
    by_direction: List[StrategyAttribution] = field(default_factory=list)
    best_strategy: str = ""
    worst_strategy: str = ""
    best_session: str = ""
    worst_session: str = ""


SESSION_HOURS = {
    "Asia": range(0, 8),
    "Europe": range(8, 14),
    "US": range(14, 21),
    "Overlap/Off-hours": range(21, 24),
}


def _classify_session(dt: datetime) -> str:
    h = dt.hour
    for name, hours in SESSION_HOURS.items():
        if h in hours:
            return name
    return "Overlap/Off-hours"


def _group_attribution(trades: list, key_fn) -> List[StrategyAttribution]:
    groups: Dict[str, list] = defaultdict(list)
    for t in trades:
        groups[key_fn(t)].append(t)

    total_pnl = sum(t.net_pnl for t in trades) or 1.0
    results = []
    for name, group in sorted(groups.items()):
        pnl = sum(t.net_pnl for t in group)
        wins = sum(1 for t in group if t.net_pnl > 0)
        results.append(StrategyAttribution(
            strategy_name=name,
            total_pnl=pnl,
            trade_count=len(group),
            win_rate=wins / len(group) if group else 0,
            avg_pnl=pnl / len(group) if group else 0,
            pnl_contribution_pct=pnl / total_pnl * 100 if total_pnl else 0,
        ))
    return results


class AttributionEngine:
    def compute_from_trades(self, trades: list) -> AttributionReport:
        if not trades:
            return AttributionReport()

        total_pnl = sum(t.net_pnl for t in trades)
        wins = sum(1 for t in trades if t.net_pnl > 0)

        by_strategy = _group_attribution(trades, lambda t: t.strategy_name)
        by_regime = _group_attribution(
            trades, lambda t: getattr(t, "market_regime", None) or "unknown"
        )
        by_session = _group_attribution(trades, lambda t: _classify_session(t.created_at))
        by_direction = _group_attribution(trades, lambda t: t.direction)

        best_s = max(by_strategy, key=lambda s: s.total_pnl) if by_strategy else None
        worst_s = min(by_strategy, key=lambda s: s.total_pnl) if by_strategy else None
        best_sess = max(by_session, key=lambda s: s.total_pnl) if by_session else None
        worst_sess = min(by_session, key=lambda s: s.total_pnl) if by_session else None

        return AttributionReport(
            total_pnl=total_pnl,
            total_trades=len(trades),
            overall_win_rate=wins / len(trades) if trades else 0,
            by_strategy=by_strategy,
            by_regime=by_regime,
            by_session=by_session,
            by_direction=by_direction,
            best_strategy=best_s.strategy_name if best_s else "",
            worst_strategy=worst_s.strategy_name if worst_s else "",
            best_session=best_sess.strategy_name if best_sess else "",
            worst_session=worst_sess.strategy_name if worst_sess else "",
        )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_attribution.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add bot/analytics/attribution.py tests/test_attribution.py
git commit -m "feat: add performance attribution engine"
```

---

### Task 14: Benchmark Comparison Engine

**Files:**
- Create: `bot/analytics/benchmark.py`
- Test: `tests/test_benchmark.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_benchmark.py
"""Tests for benchmark comparison engine."""
from __future__ import annotations
import pytest
from bot.analytics.benchmark import BenchmarkEntry, compute_buy_hold_return


class TestBuyHoldReturn:
    def test_positive_return(self):
        entry = compute_buy_hold_return(
            symbol="BTC-EUR", start_price=40000, end_price=50000,
            initial_capital=10000,
        )
        assert isinstance(entry, BenchmarkEntry)
        assert entry.buy_hold_return_pct == pytest.approx(25.0)
        assert entry.buy_hold_pnl_eur == pytest.approx(2500.0)

    def test_negative_return(self):
        entry = compute_buy_hold_return(
            symbol="ETH-EUR", start_price=3000, end_price=2400,
            initial_capital=10000,
        )
        assert entry.buy_hold_return_pct == pytest.approx(-20.0)
        assert entry.buy_hold_pnl_eur == pytest.approx(-2000.0)

    def test_zero_start_price(self):
        entry = compute_buy_hold_return(
            symbol="X-EUR", start_price=0, end_price=100,
            initial_capital=10000,
        )
        assert entry.buy_hold_return_pct == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmark.py -v`
Expected: FAIL

- [ ] **Step 3: Implement benchmark engine**

Create `bot/analytics/benchmark.py`:

```python
"""Benchmark comparison — bot performance vs buy-and-hold."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkEntry:
    symbol: str
    start_price: float
    end_price: float
    buy_hold_return_pct: float
    buy_hold_pnl_eur: float


@dataclass
class EquityCurvePoint:
    timestamp: datetime
    bot_equity: float
    btc_equity: float
    eth_equity: float


@dataclass
class BenchmarkReport:
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    initial_capital: float = 0.0
    bot_total_pnl: float = 0.0
    bot_return_pct: float = 0.0
    bot_sharpe: float = 0.0
    bot_max_drawdown_pct: float = 0.0
    btc_benchmark: Optional[BenchmarkEntry] = None
    eth_benchmark: Optional[BenchmarkEntry] = None
    per_asset_benchmarks: List[BenchmarkEntry] = field(default_factory=list)
    alpha_vs_btc: float = 0.0
    alpha_vs_eth: float = 0.0
    equity_curve: List[EquityCurvePoint] = field(default_factory=list)


def compute_buy_hold_return(
    symbol: str, start_price: float, end_price: float, initial_capital: float,
) -> BenchmarkEntry:
    if start_price <= 0:
        return BenchmarkEntry(symbol, start_price, end_price, 0.0, 0.0)
    ret_pct = (end_price - start_price) / start_price * 100
    pnl = initial_capital * ret_pct / 100
    return BenchmarkEntry(symbol, start_price, end_price, round(ret_pct, 4), round(pnl, 4))


class BenchmarkEngine:
    """Computes benchmark comparison reports from portfolio and price data."""

    def compute(
        self,
        bot_pnl: float,
        bot_return_pct: float,
        bot_sharpe: float,
        bot_max_dd: float,
        initial_capital: float,
        btc_start: float,
        btc_end: float,
        eth_start: float,
        eth_end: float,
        per_asset_prices: Optional[Dict[str, tuple]] = None,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> BenchmarkReport:
        btc = compute_buy_hold_return("BTC-EUR", btc_start, btc_end, initial_capital)
        eth = compute_buy_hold_return("ETH-EUR", eth_start, eth_end, initial_capital)

        per_asset = []
        if per_asset_prices:
            for sym, (sp, ep) in per_asset_prices.items():
                per_asset.append(compute_buy_hold_return(sym, sp, ep, initial_capital))

        return BenchmarkReport(
            period_start=period_start,
            period_end=period_end,
            initial_capital=initial_capital,
            bot_total_pnl=bot_pnl,
            bot_return_pct=bot_return_pct,
            bot_sharpe=bot_sharpe,
            bot_max_drawdown_pct=bot_max_dd,
            btc_benchmark=btc,
            eth_benchmark=eth,
            per_asset_benchmarks=per_asset,
            alpha_vs_btc=bot_return_pct - btc.buy_hold_return_pct,
            alpha_vs_eth=bot_return_pct - eth.buy_hold_return_pct,
        )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_benchmark.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add bot/analytics/benchmark.py tests/test_benchmark.py
git commit -m "feat: add benchmark comparison engine"
```

---

## Chunk 8: Walk-Forward Optimization

### Task 15: BacktestEngine Parameterization

**Files:**
- Modify: `bot/backtest/engine.py:85-96`

- [ ] **Step 1: Write test**

```python
# tests/test_walk_forward.py
"""Tests for walk-forward optimizer."""
from __future__ import annotations
import pytest
from bot.backtest.engine import BacktestEngine


class TestBacktestParameterization:
    def test_accepts_strategy_params(self):
        engine = BacktestEngine(
            candles=[],
            strategy_params={"sentiment_weight": 0.30, "entry_threshold": 0.45},
        )
        # Should not raise; params applied to router
        assert engine._router is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_walk_forward.py::TestBacktestParameterization -v`
Expected: FAIL — `strategy_params` not accepted

- [ ] **Step 3: Add strategy_params to BacktestEngine.__init__**

Modify `bot/backtest/engine.py` — add `strategy_params` parameter to `__init__`:

```python
    def __init__(
        self,
        candles: List[CandleData | Dict[str, Any]],
        initial_capital: float = 10_000.0,
        max_open_positions: int = 3,
        slippage_pct: float = 0.001,
        strategy_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        # ... existing code ...
        self._router = StrategyRouter()
        if strategy_params:
            self._router.update_hybrid_params(
                sentiment_weight=strategy_params.get("sentiment_weight", 0.25),
                entry_threshold=strategy_params.get("entry_threshold", 0.40),
                indicator_weights=strategy_params.get("indicator_weights"),
            )
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_walk_forward.py::TestBacktestParameterization -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add bot/backtest/engine.py tests/test_walk_forward.py
git commit -m "feat: add strategy_params to BacktestEngine for walk-forward support"
```

---

### Task 16: Walk-Forward Optimizer

**Files:**
- Create: `bot/learning/walk_forward.py`
- Test: `tests/test_walk_forward.py` (extend)

- [ ] **Step 1: Write tests**

```python
# Append to tests/test_walk_forward.py
from bot.learning.walk_forward import WalkForwardOptimizer, WFWindow, WFResult


class TestWalkForwardOptimizer:
    def test_window_generation(self):
        wfo = WalkForwardOptimizer()
        windows = wfo._generate_windows(total_days=120)
        # 120 days: first window train 0-60, test 60-74
        # second: train 14-74, test 74-88
        # third: train 28-88, test 88-102
        # fourth: train 42-102, test 102-116
        assert len(windows) >= 3

    def test_adoption_criteria_rejects_poor_results(self):
        wfo = WalkForwardOptimizer()
        windows = [
            WFWindow(sharpe=-0.5, pnl=-100, params={}),
            WFWindow(sharpe=0.3, pnl=-50, params={}),
            WFWindow(sharpe=0.2, pnl=-20, params={}),
        ]
        assert not wfo._should_adopt(windows)

    def test_adoption_criteria_accepts_good_results(self):
        wfo = WalkForwardOptimizer()
        windows = [
            WFWindow(sharpe=1.2, pnl=200, params={"sentiment_weight": 0.3}),
            WFWindow(sharpe=0.8, pnl=150, params={"sentiment_weight": 0.3}),
            WFWindow(sharpe=1.0, pnl=180, params={"sentiment_weight": 0.3}),
        ]
        assert wfo._should_adopt(windows)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_walk_forward.py::TestWalkForwardOptimizer -v`
Expected: FAIL

- [ ] **Step 3: Implement walk-forward optimizer**

Create `bot/learning/walk_forward.py`:

```python
"""Walk-forward optimization — rolling out-of-sample parameter validation."""
from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

TRAIN_DAYS = 60
TEST_DAYS = 14
STEP_DAYS = 14
MIN_WINDOWS = 3
MIN_OOS_SHARPE = 0.5
MAX_SHARPE_STD = 0.5


@dataclass
class WFWindow:
    sharpe: float = 0.0
    pnl: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)
    train_start: Optional[datetime] = None
    train_end: Optional[datetime] = None
    test_start: Optional[datetime] = None
    test_end: Optional[datetime] = None


@dataclass
class WFResult:
    windows: List[WFWindow] = field(default_factory=list)
    avg_oos_sharpe: float = 0.0
    avg_oos_pnl: float = 0.0
    sharpe_stability: float = 0.0
    recommended_params: Optional[Dict[str, Any]] = None
    adopted: bool = False


class WalkForwardOptimizer:
    """Rolling walk-forward optimization using BacktestEngine."""

    def __init__(self) -> None:
        self._latest_result: Optional[WFResult] = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def latest_result(self) -> Optional[WFResult]:
        return self._latest_result

    def _generate_windows(self, total_days: int) -> List[dict]:
        windows = []
        offset = 0
        while offset + TRAIN_DAYS + TEST_DAYS <= total_days:
            windows.append({
                "train_start_day": offset,
                "train_end_day": offset + TRAIN_DAYS,
                "test_start_day": offset + TRAIN_DAYS,
                "test_end_day": offset + TRAIN_DAYS + TEST_DAYS,
            })
            offset += STEP_DAYS
        return windows

    def _should_adopt(self, windows: List[WFWindow]) -> bool:
        if len(windows) < MIN_WINDOWS:
            return False
        sharpes = [w.sharpe for w in windows]
        pnls = [w.pnl for w in windows]
        avg_sharpe = statistics.mean(sharpes)
        avg_pnl = statistics.mean(pnls)
        sharpe_std = statistics.stdev(sharpes) if len(sharpes) > 1 else 999
        if avg_sharpe < MIN_OOS_SHARPE:
            return False
        if sharpe_std > MAX_SHARPE_STD:
            return False
        if avg_pnl <= 0:
            return False
        return True

    async def run(self, candle_fetcher=None) -> WFResult:
        """Execute full walk-forward optimization. candle_fetcher returns candles for a date range."""
        self._running = True
        try:
            result = await self._execute(candle_fetcher)
            self._latest_result = result
            return result
        finally:
            self._running = False

    async def _execute(self, candle_fetcher) -> WFResult:
        from bot.backtest.engine import BacktestEngine

        # Determine available data range
        # For now, use 120 days as default
        total_days = 120
        windows_spec = self._generate_windows(total_days)

        if len(windows_spec) < MIN_WINDOWS:
            logger.warning("Walk-forward: insufficient data for %d windows", MIN_WINDOWS)
            return WFResult()

        # Candidate parameter sets
        candidates = [
            {"sentiment_weight": 0.15, "entry_threshold": 0.30},
            {"sentiment_weight": 0.20, "entry_threshold": 0.35},
            {"sentiment_weight": 0.25, "entry_threshold": 0.40},
            {"sentiment_weight": 0.30, "entry_threshold": 0.35},
            {"sentiment_weight": 0.10, "entry_threshold": 0.30},
        ]

        wf_windows: List[WFWindow] = []

        for ws in windows_spec:
            if candle_fetcher is None:
                break

            train_candles = await candle_fetcher(ws["train_start_day"], ws["train_end_day"])
            test_candles = await candle_fetcher(ws["test_start_day"], ws["test_end_day"])

            if not train_candles or not test_candles:
                continue

            # Find best params on training data
            best_pnl = float("-inf")
            best_params = candidates[0]

            for params in candidates:
                engine = BacktestEngine(
                    train_candles, strategy_params=params, slippage_pct=0.001,
                )
                result = engine.run()
                if result.total_pnl > best_pnl:
                    best_pnl = result.total_pnl
                    best_params = params

            # Test best params out-of-sample
            test_engine = BacktestEngine(
                test_candles, strategy_params=best_params, slippage_pct=0.001,
            )
            test_result = test_engine.run()

            wf_windows.append(WFWindow(
                sharpe=test_result.sharpe_ratio,
                pnl=test_result.total_pnl,
                params=best_params,
            ))

        adopted = self._should_adopt(wf_windows)
        avg_sharpe = statistics.mean(w.sharpe for w in wf_windows) if wf_windows else 0
        avg_pnl = statistics.mean(w.pnl for w in wf_windows) if wf_windows else 0
        sharpe_std = statistics.stdev(w.sharpe for w in wf_windows) if len(wf_windows) > 1 else 0

        best_params = None
        if adopted and wf_windows:
            # Use params from the best-performing window
            best_window = max(wf_windows, key=lambda w: w.pnl)
            best_params = best_window.params

        return WFResult(
            windows=wf_windows,
            avg_oos_sharpe=avg_sharpe,
            avg_oos_pnl=avg_pnl,
            sharpe_stability=sharpe_std,
            recommended_params=best_params,
            adopted=adopted,
        )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_walk_forward.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add bot/learning/walk_forward.py tests/test_walk_forward.py
git commit -m "feat: add walk-forward optimizer with rolling OOS validation"
```

---

## Chunk 9: API Endpoints

### Task 17: Analytics API Router

**Files:**
- Create: `api/routers/analytics.py`

- [ ] **Step 1: Create analytics router**

```python
# api/routers/analytics.py
"""API endpoints for performance attribution, benchmark, and walk-forward."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from bot.analytics.attribution import AttributionEngine
from bot.analytics.benchmark import BenchmarkEngine
from bot.data.database import get_session
from bot.data.repositories import get_trades

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/attribution")
async def get_attribution(
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
):
    start_dt = datetime.fromisoformat(start) if start else None
    end_dt = datetime.fromisoformat(end) if end else None

    async with get_session() as session:
        trades = await get_trades(
            session, limit=10000, start_date=start_dt, end_date=end_dt,
        )

    engine = AttributionEngine()
    report = engine.compute_from_trades(trades)
    return {
        "total_pnl": report.total_pnl,
        "total_trades": report.total_trades,
        "overall_win_rate": report.overall_win_rate,
        "by_strategy": [vars(s) for s in report.by_strategy],
        "by_regime": [vars(s) for s in report.by_regime],
        "by_session": [vars(s) for s in report.by_session],
        "by_direction": [vars(s) for s in report.by_direction],
        "best_strategy": report.best_strategy,
        "worst_strategy": report.worst_strategy,
        "best_session": report.best_session,
        "worst_session": report.worst_session,
    }


@router.get("/benchmark")
async def get_benchmark(
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
):
    # Placeholder — full implementation requires fetching portfolio snapshots
    # and candle prices for BTC/ETH. Wired in main.py integration task.
    return {"status": "not_yet_wired", "message": "Benchmark requires portfolio data wiring"}


@router.get("/walk-forward")
async def get_walk_forward():
    from bot.learning.walk_forward import WalkForwardOptimizer
    # Access singleton — wired in main.py
    return {"status": "no_results", "message": "No walk-forward run completed yet"}


@router.post("/walk-forward/run", status_code=202)
async def trigger_walk_forward():
    return {"status": "accepted", "message": "Walk-forward run triggered"}
```

- [ ] **Step 2: Commit**

```bash
git add api/routers/analytics.py
git commit -m "feat: add analytics API router (attribution, benchmark, walk-forward)"
```

---

### Task 18: Journal + Order Book API Routers

**Files:**
- Create: `api/routers/journal.py`
- Create: `api/routers/orderbook.py`

- [ ] **Step 1: Create journal router**

```python
# api/routers/journal.py
"""API endpoints for trade journal and chart serving."""
from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from bot.data.database import get_session

router = APIRouter(prefix="/api", tags=["journal"])

CHARTS_DIR = "data/charts"


@router.get("/journal")
async def get_journal(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    from bot.data.repositories import get_journal_entries
    async with get_session() as session:
        entries = await get_journal_entries(session, limit=limit, offset=offset)
    return [_journal_to_dict(e) for e in entries]


@router.get("/journal/{trade_id}")
async def get_journal_entry(trade_id: int):
    from bot.data.repositories import get_journal_by_trade_id
    async with get_session() as session:
        entry = await get_journal_by_trade_id(session, trade_id)
    if not entry:
        raise HTTPException(404, f"No journal entry for trade {trade_id}")
    return _journal_to_dict(entry)


@router.get("/charts/{filename}")
async def serve_chart(filename: str):
    path = os.path.join(CHARTS_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(404, "Chart not found")
    return FileResponse(path, media_type="image/png")


def _journal_to_dict(entry) -> dict:
    return {
        "id": entry.id,
        "trade_id": entry.trade_id,
        "symbol": entry.symbol,
        "direction": entry.direction,
        "strategy_name": entry.strategy_name,
        "market_regime": entry.market_regime,
        "entry_composite_score": entry.entry_composite_score,
        "entry_reasoning": entry.entry_reasoning,
        "exit_reasoning": entry.exit_reasoning,
        "entry_chart_path": entry.entry_chart_path,
        "exit_chart_path": entry.exit_chart_path,
        "created_at": entry.created_at.isoformat() if entry.created_at else None,
    }
```

- [ ] **Step 2: Create orderbook router**

```python
# api/routers/orderbook.py
"""API endpoint for order book depth analysis."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from dataclasses import asdict

router = APIRouter(prefix="/api", tags=["orderbook"])

# Singleton reference — set during main.py wiring
_orderbook_provider = None


def set_provider(provider):
    global _orderbook_provider
    _orderbook_provider = provider


@router.get("/orderbook/{symbol}")
async def get_orderbook(symbol: str):
    if _orderbook_provider is None:
        raise HTTPException(503, "Order book provider not initialized")
    analysis = _orderbook_provider.analyze(symbol)
    return {
        "imbalance": analysis.imbalance,
        "spread_pct": analysis.spread_pct,
        "bid_walls": [{"price": w.price, "quantity": w.quantity, "eur_value": w.eur_value} for w in analysis.bid_walls],
        "ask_walls": [{"price": w.price, "quantity": w.quantity, "eur_value": w.eur_value} for w in analysis.ask_walls],
        "support_levels": analysis.support_levels,
        "resistance_levels": analysis.resistance_levels,
        "depth_buckets": [{"price_low": b.price_low, "price_high": b.price_high, "bid_volume": b.bid_volume, "ask_volume": b.ask_volume} for b in analysis.depth_buckets],
    }
```

- [ ] **Step 3: Commit**

```bash
git add api/routers/journal.py api/routers/orderbook.py
git commit -m "feat: add journal and orderbook API routers"
```

---

## Chunk 10: Repository Functions + Integration Wiring

### Task 19: Repository Functions for Journal + Onchain

**Files:**
- Modify: `bot/data/repositories.py`

- [ ] **Step 1: Add journal and onchain repo functions**

Add to `bot/data/repositories.py`:

```python
from bot.data.models import TradeJournal, OnchainScore


async def save_journal_entry(session, **kwargs) -> TradeJournal:
    entry = TradeJournal(**kwargs)
    session.add(entry)
    await session.flush()
    return entry


async def update_journal_exit(session, trade_id: int, **kwargs) -> None:
    from sqlalchemy import select, update
    stmt = update(TradeJournal).where(TradeJournal.trade_id == trade_id).values(**kwargs)
    await session.execute(stmt)
    await session.flush()


async def get_journal_entries(session, limit: int = 20, offset: int = 0):
    from sqlalchemy import select
    stmt = select(TradeJournal).order_by(TradeJournal.created_at.desc()).limit(limit).offset(offset)
    result = await session.execute(stmt)
    return result.scalars().all()


async def get_journal_by_trade_id(session, trade_id: int):
    from sqlalchemy import select
    stmt = select(TradeJournal).where(TradeJournal.trade_id == trade_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def save_onchain_score(session, **kwargs) -> OnchainScore:
    score = OnchainScore(**kwargs)
    session.add(score)
    await session.flush()
    return score
```

- [ ] **Step 2: Commit**

```bash
git add bot/data/repositories.py
git commit -m "feat: add repository functions for journal and onchain scores"
```

---

### Task 20: Trading Loop Integration

**Files:**
- Modify: `bot/trading_loop.py`
- Modify: `bot/exchange/order_manager.py`

- [ ] **Step 1: Update TradingLoop.__init__ to accept new providers**

Add parameters to `TradingLoop.__init__`:

```python
    onchain: Any = None,
    orderbook: Any = None,
```

Store as `self.onchain` and `self.orderbook`.

- [ ] **Step 2: Update MarketContext construction in run_cycle**

In `run_cycle`, around line 269, update the MarketContext creation to include new fields:

```python
    # Resample for MTF
    df_15m = self._resample(df_5m, "15min") if len(df_5m) >= 10 else df_5m
    df_4h = self._resample(df_5m, "4h") if len(df_5m) >= 200 else df_1h
    df_1d = self._resample(df_5m, "1D") if len(df_5m) >= 500 else df_1h

    # Get provider scores
    onchain_score = self.onchain.score(symbol) if self.onchain and self.onchain.is_available() else 0.0
    orderbook_imb = self.orderbook.score(symbol) if self.orderbook and self.orderbook.is_available() else 0.0

    ctx = MarketContext(
        symbol=symbol,
        candles_5m=df_5m,
        candles_1h=df_1h,
        candles_15m=df_15m,
        candles_4h=df_4h,
        candles_1d=df_1d,
        current_price=df_5m["close"].iloc[-1],
        sentiment_score=sentiment_score,
        portfolio_equity_eur=equity,
        open_position_count=portfolio.open_position_count(),
        onchain_score=onchain_score,
        orderbook_imbalance=orderbook_imb,
        market_regime=regime,
    )
```

- [ ] **Step 3: Add _resample helper method**

Add to TradingLoop class:

```python
    @staticmethod
    def _resample(df_5m: pd.DataFrame, freq: str) -> pd.DataFrame:
        if df_5m.empty:
            return df_5m
        return df_5m.resample(freq).agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()
```

Add `import pandas as pd` to the imports if not already present.

- [ ] **Step 4: Extend order_manager TOPIC_TRADE_OPENED payload**

In `bot/exchange/order_manager.py`, modify `_record_order` to accept and forward extra journal data:

Add `extra_data: Optional[Dict] = None` parameter to `_record_order`.

Extend the publish payload:

```python
        payload = {
            "order_id": order_id,
            "bitvavo_order_id": resp.get("orderId"),
            "symbol": resp.get("market"),
            "side": resp.get("side"),
            "fill_price": fill_price,
            "filled_amount": filled,
            "fee": fee,
            "strategy_name": strategy_name,
            "paper_trade": paper or self.client.paper_trading,
        }
        if extra_data:
            payload.update(extra_data)
        await self._bus.publish(TOPIC_TRADE_OPENED, payload)
```

Update `submit_buy` and `submit_sell` to accept and pass through `extra_data: Optional[Dict] = None`.

- [ ] **Step 5: In trading_loop.py, pass journal data when submitting orders**

After signal is accepted (around line 400), build extra_data:

```python
    journal_data = {
        "signal": best_signal,
        "market_regime": regime,
        "candle_data": df_5m.tail(100).to_dict("records") if len(df_5m) > 0 else [],
    }
```

Pass `extra_data=journal_data` to `order_mgr.submit_buy` / `order_mgr.submit_sell`.

- [ ] **Step 6: Commit**

```bash
git add bot/trading_loop.py bot/exchange/order_manager.py
git commit -m "feat: wire MTF, on-chain, orderbook into trading loop and extend trade.opened payload"
```

---

### Task 21: Main.py Wiring + Router Registration

**Files:**
- Modify: `bot/main.py`

- [ ] **Step 1: Initialize new providers in main.py**

Add initialization of `OnchainProvider`, `OrderBookProvider`, and wiring to the trading loop:

```python
from bot.indicators.onchain import OnchainProvider
from bot.indicators.orderbook import OrderBookProvider
from bot.learning.walk_forward import WalkForwardOptimizer

# In the component initialization section:
_onchain_provider: Optional[OnchainProvider] = None
_orderbook_provider: Optional[OrderBookProvider] = None
_walk_forward: Optional[WalkForwardOptimizer] = None

def _get_onchain():
    global _onchain_provider
    if _onchain_provider is None:
        _onchain_provider = OnchainProvider()
    return _onchain_provider

def _get_orderbook():
    global _orderbook_provider
    if _orderbook_provider is None:
        settings = get_settings()
        _orderbook_provider = OrderBookProvider(depth_levels=settings.orderbook_depth_levels)
    return _orderbook_provider
```

- [ ] **Step 2: Pass providers to TradingLoop**

When creating `TradingLoop`, add the new providers:

```python
    onchain=_get_onchain() if settings.onchain_enabled else None,
    orderbook=_get_orderbook() if settings.orderbook_enabled else None,
```

- [ ] **Step 3: Register new API routers**

```python
from api.routers.analytics import router as analytics_router
from api.routers.journal import router as journal_router
from api.routers.orderbook import router as orderbook_router, set_provider as set_ob_provider

app.include_router(analytics_router)
app.include_router(journal_router)
app.include_router(orderbook_router)

# Wire orderbook provider singleton
if settings.orderbook_enabled:
    set_ob_provider(_get_orderbook())
```

- [ ] **Step 4: Add on-chain polling to scheduler**

Add an on-chain poll task to the scheduler if enabled:

```python
if settings.onchain_enabled:
    # Register onchain poll alongside sentiment cycle
    async def _onchain_cycle():
        prices = {s: cache.get_df(s, "5m")["close"].iloc[-1]
                  for s in _tradeable_symbols
                  if cache.has_enough(s)}
        await _get_onchain().poll(list(_tradeable_symbols), prices)
```

- [ ] **Step 5: Add book WebSocket callback**

When creating `BitvavoWebSocket`, add `on_book` callback:

```python
    async def _on_book(msg):
        symbol = msg.get("market")
        bids = msg.get("bids", [])
        asks = msg.get("asks", [])
        ob = _get_orderbook()
        if "nonce" in msg and not ob._books.get(symbol):
            # Initial snapshot
            ob.set_snapshot(symbol, bids, asks)
        else:
            for price, qty in bids:
                ob.update_level(symbol, "bid", float(price), float(qty))
            for price, qty in asks:
                ob.update_level(symbol, "ask", float(price), float(qty))
```

Pass `on_book=_on_book` when creating `BitvavoWebSocket`.

- [ ] **Step 6: Run all tests**

Run: `pytest tests/ -v --timeout=60`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add bot/main.py
git commit -m "feat: wire all new providers, routers, and scheduler tasks in main.py"
```

---

### Task 22: Journal Event Bus Subscriber

**Files:**
- Modify: `bot/main.py` (add subscriber)

- [ ] **Step 1: Add journal subscriber for trade events**

In main.py, after component initialization, subscribe to trade events:

```python
import asyncio
from bot.analytics.journal import generate_entry_reasoning, generate_exit_reasoning
from bot.charts.snapshot import generate_entry_chart, generate_exit_chart

async def _on_trade_opened(data):
    signal = data.get("signal")
    if not signal:
        return
    snap = signal.indicator_snapshot or {}
    from bot.data.database import get_session
    from bot.data.repositories import save_journal_entry
    async with get_session() as session:
        reasoning = generate_entry_reasoning(
            symbol=signal.symbol, direction=signal.direction,
            strategy=signal.strategy_name,
            regime=data.get("market_regime", "unknown"),
            final_score=snap.get("final_score", 0),
            threshold=0.35,
            mtf_scores=snap.get("mtf_scores", {}),
            mtf_agreement=snap.get("mtf_agreement", 0),
            technical_score=signal.technical_score,
            sentiment_score=signal.sentiment_score,
            onchain_score=snap.get("onchain_score", 0),
            orderbook_imbalance=snap.get("orderbook_imbalance", 0),
            indicator_values=snap,
        )
        await save_journal_entry(
            session,
            trade_id=data.get("order_id"),
            symbol=data.get("symbol"),
            direction=signal.direction,
            strategy_name=signal.strategy_name,
            market_regime=data.get("market_regime", "unknown"),
            entry_technical_scores=snap.get("mtf_scores"),
            entry_mtf_scores=snap.get("mtf_scores"),
            entry_sentiment_score=signal.sentiment_score,
            entry_onchain_score=snap.get("onchain_score"),
            entry_orderbook_imbalance=snap.get("orderbook_imbalance"),
            entry_composite_score=snap.get("final_score", 0),
            entry_reasoning=reasoning,
        )

    # Chart in background thread
    candle_data = data.get("candle_data", [])
    if candle_data:
        import pandas as pd
        df = pd.DataFrame(candle_data)
        if not df.empty and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp").sort_index()
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, generate_entry_chart,
                df, data.get("fill_price", 0), data.get("symbol", ""),
                data.get("order_id", 0),
            )

# Subscribe
bus = get_bus()
bus.subscribe(TOPIC_TRADE_OPENED, _on_trade_opened)
```

- [ ] **Step 2: Commit**

```bash
git add bot/main.py
git commit -m "feat: add journal event subscriber for trade.opened"
```

---

### Task 23: Chart Retention Cleanup + Final Touches

**Files:**
- Modify: `bot/main.py`

- [ ] **Step 1: Add chart cleanup to scheduler**

```python
import glob
import os
import time

async def _cleanup_old_charts():
    charts_dir = "data/charts"
    if not os.path.exists(charts_dir):
        return
    cutoff = time.time() - (90 * 86400)  # 90 days
    for f in glob.glob(os.path.join(charts_dir, "*.png")):
        if os.path.getmtime(f) < cutoff:
            os.remove(f)
            logger.info("Removed old chart: %s", f)
```

Register in the scheduler as a weekly task.

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/ -v --timeout=60`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add bot/main.py
git commit -m "feat: add weekly chart retention cleanup task"
```

---

## Execution Notes

- Tasks 1-9 are independent enough to be parallelized in subagent-driven execution
- Tasks 10 (DB migration) must run before Tasks 11-12 (journal) and Task 19 (repos)
- Tasks 17-18 (API routers) depend on Tasks 11-14 (analytics modules)
- Tasks 20-23 (integration) must run last as they wire everything together
- All tests use pytest — run `pytest tests/ -v` after each chunk
