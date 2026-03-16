# Dynamic Tradeable Universe Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Only trade strategy-symbol combinations that pass walk-forward validation, each with independently optimized parameters.

**Architecture:** New `AdoptedUniverse` class stores adopted strategy-symbol combos with per-combo params. Walk-forward optimizes each strategy independently per symbol using a new `target_strategy` filter on `BacktestEngine`. Trading loop consults the universe to skip non-adopted symbols and use per-strategy params for stops, sizing, fees, and regime detection.

**Tech Stack:** Python 3.11, pytest, asyncio, Optuna, dataclasses

**Spec:** `docs/superpowers/specs/2026-03-16-dynamic-universe-design.md`

---

## File Structure

| Action | File | Purpose |
|--------|------|---------|
| Create | `bot/strategy/adopted_universe.py` | AdoptedUniverse class + CHAMPION_DEFAULTS + SymbolConfig/StrategyConfig |
| Create | `tests/test_adopted_universe.py` | Unit tests for AdoptedUniverse |
| Create | `tests/test_backtest_target_strategy.py` | Tests for target_strategy filter |
| Modify | `bot/backtest/engine.py` | Add target_strategy filter + regime params from strategy_params |
| Modify | `bot/learning/walk_forward.py` | Per-strategy optimization + candle prefetch + CHAMPION_DEFAULTS import |
| Modify | `bot/main.py` | Wire AdoptedUniverse, update WF results processing + caller |
| Modify | `bot/trading_loop.py` | Universe gating, per-strategy params, regime refactor, confluence relaxation |
| Modify | `api/routers/analytics.py` | Fix dead method + GET /universe endpoint |
| Modify | `bot/notifications/discord.py` | Handle nested WF results structure |

---

## Chunk 1: AdoptedUniverse

### Task 1: AdoptedUniverse class

**Files:**
- Create: `bot/strategy/adopted_universe.py`
- Create: `tests/test_adopted_universe.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for bot.strategy.adopted_universe."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from bot.strategy.adopted_universe import (
    AdoptedUniverse, SymbolConfig, StrategyConfig,
    CHAMPION_DEFAULTS, ALL_STRATEGIES,
)


def _mock_wf_result(adopted=True, sharpe=1.0, pnl=100.0, params=None):
    """Create a mock WFResult."""
    r = MagicMock()
    r.adopted = adopted
    r.recommended_params = params or dict(CHAMPION_DEFAULTS)
    r.avg_oos_sharpe = sharpe
    r.avg_oos_pnl = pnl
    return r


class TestEmptyUniverse:
    def test_empty_blocks_all(self):
        u = AdoptedUniverse()
        assert u.is_adopted("BTC-EUR") is False
        assert u.get_enabled_strategies("BTC-EUR") == []
        assert u.adopted_symbols() == []
        assert u.count() == 0

    def test_get_strategy_params_fallback(self):
        u = AdoptedUniverse()
        params = u.get_strategy_params("BTC-EUR", "orderflow")
        assert params == CHAMPION_DEFAULTS

    def test_get_regime_params_fallback(self):
        u = AdoptedUniverse()
        rp = u.get_regime_params("BTC-EUR")
        assert rp["quiet_atr_threshold"] == CHAMPION_DEFAULTS["quiet_atr_threshold"]
        assert rp["regime_adx_threshold"] == CHAMPION_DEFAULTS["regime_adx_threshold"]


class TestUpdate:
    def test_mixed_adopted_rejected(self):
        u = AdoptedUniverse()
        results = {
            "BTC-EUR": {
                "orderflow": _mock_wf_result(adopted=True, sharpe=1.5, pnl=3000),
                "range": _mock_wf_result(adopted=False),
                "squeeze": _mock_wf_result(adopted=True, sharpe=0.8, pnl=500),
                "funding_contrarian": _mock_wf_result(adopted=False),
            },
            "DOGE-EUR": {
                "orderflow": _mock_wf_result(adopted=False),
                "range": _mock_wf_result(adopted=False),
                "squeeze": _mock_wf_result(adopted=False),
                "funding_contrarian": _mock_wf_result(adopted=False),
            },
        }
        u.update(results)
        assert u.is_adopted("BTC-EUR") is True
        assert u.is_adopted("DOGE-EUR") is False
        assert set(u.get_enabled_strategies("BTC-EUR")) == {"orderflow", "squeeze"}
        assert u.count() == 1

    def test_per_strategy_params(self):
        of_params = dict(CHAMPION_DEFAULTS, atr_multiplier=3.0, rr_ratio=2.5)
        sq_params = dict(CHAMPION_DEFAULTS, atr_multiplier=4.5, rr_ratio=3.5)
        u = AdoptedUniverse()
        results = {
            "BTC-EUR": {
                "orderflow": _mock_wf_result(adopted=True, params=of_params),
                "squeeze": _mock_wf_result(adopted=True, params=sq_params),
                "range": _mock_wf_result(adopted=False),
                "funding_contrarian": _mock_wf_result(adopted=False),
            },
        }
        u.update(results)
        assert u.get_strategy_params("BTC-EUR", "orderflow")["atr_multiplier"] == 3.0
        assert u.get_strategy_params("BTC-EUR", "squeeze")["atr_multiplier"] == 4.5

    def test_regime_params_from_best_strategy(self):
        params_high = dict(CHAMPION_DEFAULTS, quiet_atr_threshold=1.3, regime_adx_threshold=26)
        params_low = dict(CHAMPION_DEFAULTS, quiet_atr_threshold=0.9, regime_adx_threshold=22)
        u = AdoptedUniverse()
        results = {
            "BTC-EUR": {
                "orderflow": _mock_wf_result(adopted=True, sharpe=2.0, params=params_high),
                "range": _mock_wf_result(adopted=True, sharpe=0.5, params=params_low),
                "squeeze": _mock_wf_result(adopted=False),
                "funding_contrarian": _mock_wf_result(adopted=False),
            },
        }
        u.update(results)
        rp = u.get_regime_params("BTC-EUR")
        assert rp["quiet_atr_threshold"] == 1.3  # from orderflow (higher sharpe)
        assert rp["regime_adx_threshold"] == 26

    def test_zero_adoption_preserves_previous(self):
        u = AdoptedUniverse()
        # First update with adoption
        results1 = {
            "BTC-EUR": {
                "orderflow": _mock_wf_result(adopted=True),
                "range": _mock_wf_result(adopted=False),
                "squeeze": _mock_wf_result(adopted=False),
                "funding_contrarian": _mock_wf_result(adopted=False),
            },
        }
        u.update(results1)
        assert u.count() == 1

        # Second update with zero adoptions
        results2 = {
            "BTC-EUR": {
                "orderflow": _mock_wf_result(adopted=False),
                "range": _mock_wf_result(adopted=False),
                "squeeze": _mock_wf_result(adopted=False),
                "funding_contrarian": _mock_wf_result(adopted=False),
            },
        }
        u.update(results2)
        assert u.count() == 1  # preserved previous

    def test_symbol_zero_strategies_not_adopted(self):
        u = AdoptedUniverse()
        results = {
            "BTC-EUR": {
                "orderflow": _mock_wf_result(adopted=False),
                "range": _mock_wf_result(adopted=False),
                "squeeze": _mock_wf_result(adopted=False),
                "funding_contrarian": _mock_wf_result(adopted=False),
            },
        }
        u.update(results)
        assert u.is_adopted("BTC-EUR") is False


class TestAdoptedSymbols:
    def test_returns_correct_list(self):
        u = AdoptedUniverse()
        results = {
            "BTC-EUR": {
                "orderflow": _mock_wf_result(adopted=True),
                "range": _mock_wf_result(adopted=False),
                "squeeze": _mock_wf_result(adopted=False),
                "funding_contrarian": _mock_wf_result(adopted=False),
            },
            "ETH-EUR": {
                "orderflow": _mock_wf_result(adopted=False),
                "range": _mock_wf_result(adopted=True),
                "squeeze": _mock_wf_result(adopted=False),
                "funding_contrarian": _mock_wf_result(adopted=False),
            },
        }
        u.update(results)
        assert sorted(u.adopted_symbols()) == ["BTC-EUR", "ETH-EUR"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_adopted_universe.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bot.strategy.adopted_universe'`

- [ ] **Step 3: Write implementation**

```python
"""AdoptedUniverse: per-strategy-per-symbol params from walk-forward."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

CHAMPION_DEFAULTS: Dict[str, Any] = {
    "atr_multiplier": 3.5,
    "rr_ratio": 2.5,
    "base_risk_pct": 3.0,
    "min_profit_multiple": 3.0,
    "cooldown_hours": 24,
    "max_hold_hours": 120,
    "quiet_atr_threshold": 1.0,
    "regime_adx_threshold": 24,
}

ALL_STRATEGIES = ["orderflow", "range", "squeeze", "funding_contrarian"]


@dataclass
class StrategyConfig:
    params: Dict[str, Any] = field(default_factory=dict)
    avg_sharpe: float = 0.0
    avg_pnl: float = 0.0


@dataclass
class SymbolConfig:
    strategies: Dict[str, StrategyConfig] = field(default_factory=dict)
    regime_params: Dict[str, float] = field(default_factory=dict)


class AdoptedUniverse:
    """Stores adopted strategy-symbol combos with per-combo parameters."""

    def __init__(self) -> None:
        self._adopted: Dict[str, SymbolConfig] = {}

    def is_adopted(self, symbol: str) -> bool:
        return symbol in self._adopted

    def get_strategy_params(self, symbol: str, strategy: str) -> Dict[str, Any]:
        if symbol in self._adopted and strategy in self._adopted[symbol].strategies:
            return self._adopted[symbol].strategies[strategy].params
        return dict(CHAMPION_DEFAULTS)

    def get_regime_params(self, symbol: str) -> Dict[str, float]:
        if symbol in self._adopted:
            return self._adopted[symbol].regime_params
        return {
            "quiet_atr_threshold": CHAMPION_DEFAULTS["quiet_atr_threshold"],
            "regime_adx_threshold": CHAMPION_DEFAULTS["regime_adx_threshold"],
        }

    def get_enabled_strategies(self, symbol: str) -> List[str]:
        if symbol in self._adopted:
            return list(self._adopted[symbol].strategies.keys())
        return []

    def adopted_symbols(self) -> List[str]:
        return list(self._adopted.keys())

    def count(self) -> int:
        return len(self._adopted)

    def update(self, wf_results: Dict[str, Dict[str, Any]]) -> None:
        """Rebuild universe from walk-forward results.

        Args:
            wf_results: {symbol: {strategy_name: WFResult}}
        """
        new_adopted: Dict[str, SymbolConfig] = {}

        for symbol, strat_results in wf_results.items():
            adopted_strats: Dict[str, StrategyConfig] = {}
            best_sharpe = float("-inf")
            best_regime_params = {
                "quiet_atr_threshold": CHAMPION_DEFAULTS["quiet_atr_threshold"],
                "regime_adx_threshold": CHAMPION_DEFAULTS["regime_adx_threshold"],
            }

            for strat_name, result in strat_results.items():
                if result.adopted and result.recommended_params:
                    adopted_strats[strat_name] = StrategyConfig(
                        params=dict(result.recommended_params),
                        avg_sharpe=result.avg_oos_sharpe,
                        avg_pnl=result.avg_oos_pnl,
                    )
                    if result.avg_oos_sharpe > best_sharpe:
                        best_sharpe = result.avg_oos_sharpe
                        best_regime_params = {
                            "quiet_atr_threshold": result.recommended_params.get(
                                "quiet_atr_threshold", CHAMPION_DEFAULTS["quiet_atr_threshold"]
                            ),
                            "regime_adx_threshold": result.recommended_params.get(
                                "regime_adx_threshold", CHAMPION_DEFAULTS["regime_adx_threshold"]
                            ),
                        }

            if adopted_strats:
                new_adopted[symbol] = SymbolConfig(
                    strategies=adopted_strats,
                    regime_params=best_regime_params,
                )

        if not new_adopted and self._adopted:
            logger.warning(
                "Walk-forward: zero symbols adopted — keeping previous universe (%d symbols)",
                len(self._adopted),
            )
            return

        # Log diff
        old_symbols = set(self._adopted.keys())
        new_symbols = set(new_adopted.keys())
        added = new_symbols - old_symbols
        removed = old_symbols - new_symbols
        total_combos = sum(len(sc.strategies) for sc in new_adopted.values())

        parts = []
        for s in sorted(added):
            strats = ",".join(sorted(new_adopted[s].strategies.keys()))
            parts.append(f"+{s}({strats})")
        for s in sorted(removed):
            parts.append(f"-{s}")

        logger.info(
            "Universe updated: %s — %d symbols / %d strategy-combos adopted",
            " ".join(parts) if parts else "no changes",
            len(new_adopted),
            total_combos,
        )

        self._adopted = new_adopted
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_adopted_universe.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add bot/strategy/adopted_universe.py tests/test_adopted_universe.py
git commit -m "feat: add AdoptedUniverse for per-strategy-per-symbol params"
```

---

## Chunk 2: BacktestEngine target_strategy + regime params

### Task 2: BacktestEngine target_strategy filter

**Files:**
- Modify: `bot/backtest/engine.py:111-164` (init), `bot/backtest/engine.py:269-284` (regime), `bot/backtest/engine.py:298-320` (strategy eval)
- Create: `tests/test_backtest_target_strategy.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for BacktestEngine target_strategy filter."""
from __future__ import annotations

import pytest
from bot.backtest.engine import BacktestEngine


def _make_candles(n=2000):
    """Generate minimal candle data for backtest."""
    import random
    random.seed(42)
    candles = []
    price = 50000.0
    from datetime import datetime, timedelta, timezone
    t = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        change = random.uniform(-0.5, 0.5)
        o = price
        h = price + abs(change) + random.uniform(0, 0.3)
        l = price - abs(change) - random.uniform(0, 0.3)
        c = price + change
        candles.append({
            "timestamp": t + timedelta(minutes=5 * i),
            "open": o, "high": h, "low": l, "close": c,
            "volume": random.uniform(1, 10),
        })
        price = c
    return candles


class TestTargetStrategy:
    def test_target_strategy_filters_trades(self):
        candles = _make_candles(5000)
        # Run with all strategies
        engine_all = BacktestEngine(candles, strategy_params={"base_risk_pct": 3.0})
        result_all = engine_all.run()

        # Run with only orderflow
        engine_of = BacktestEngine(
            candles, strategy_params={"base_risk_pct": 3.0},
            target_strategy="orderflow",
        )
        result_of = engine_of.run()

        # All trades should be from orderflow
        for trade in result_of.trade_log:
            assert trade.strategy == "orderflow", f"Expected orderflow, got {trade.strategy}"

    def test_target_strategy_none_unchanged(self):
        candles = _make_candles(5000)
        engine = BacktestEngine(candles, strategy_params={"base_risk_pct": 3.0})
        result = engine.run()
        strategies_used = set(t.strategy for t in result.trade_log)
        # Should have multiple strategies (or at least not fail)
        assert result is not None

    def test_confluence_disabled_with_target(self):
        """When target_strategy is set, confluence should be skipped."""
        candles = _make_candles(5000)
        engine = BacktestEngine(
            candles, strategy_params={"base_risk_pct": 3.0},
            target_strategy="range",
        )
        result = engine.run()
        # Should not crash — confluence gate is bypassed
        assert result is not None
        # All trades should be from the target strategy only
        for trade in result.trade_log:
            assert trade.strategy == "range", f"Expected range, got {trade.strategy}"


class TestRegimeParamsFromStrategy:
    def test_regime_uses_strategy_params(self):
        candles = _make_candles(5000)
        # Very high quiet threshold = everything is QUIET = no trades
        engine = BacktestEngine(
            candles,
            strategy_params={"quiet_atr_threshold": 999.0, "base_risk_pct": 3.0},
        )
        result = engine.run()
        assert result.total_trades == 0  # all bars classified as QUIET

    def test_regime_default_without_params(self):
        candles = _make_candles(5000)
        engine = BacktestEngine(candles)
        result = engine.run()
        # Should work normally with default thresholds
        assert result is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_backtest_target_strategy.py -v`
Expected: FAIL — `BacktestEngine` doesn't accept `target_strategy` parameter

- [ ] **Step 3: Implement target_strategy filter in BacktestEngine**

In `bot/backtest/engine.py`:

**3a. Add `target_strategy` to `__init__` and store raw strategy_params** (after `strategy_params` param, around line 116):
```python
def __init__(
    self,
    candles,
    initial_capital=10_000.0,
    max_open_positions=10,
    slippage_pct=0.001,
    strategy_params=None,
    target_strategy=None,          # NEW
):
    ...
    self._strategy_params = strategy_params or {}  # NEW: store raw dict for regime lookups
    self._target_strategy = target_strategy  # NEW
```

Note: The engine currently destructures `strategy_params` into individual attributes (`self._atr_multiplier`, etc.) but does NOT store the raw dict. We must add `self._strategy_params = strategy_params or {}` so regime threshold lookups in Step 3b work.

**3b. Read regime thresholds from strategy_params** (replace hardcoded values around lines 269-284):
```python
# Replace hardcoded 1.0, 4.0, 25, 20 with:
_quiet_thresh = self._strategy_params.get("quiet_atr_threshold", 1.0)
_regime_adx = self._strategy_params.get("regime_adx_threshold", 24)
# NOTE: Default changes from 25 (hardcoded in engine) to 24 (CHAMPION_DEFAULTS).
# This is intentional — walk-forward will optimize the actual value.
# Then use _quiet_thresh and _regime_adx in the comparisons:
# atr_pct < _quiet_thresh → QUIET
# atr_pct > 4.0 → VOLATILE (stays hardcoded)
# adx > _regime_adx → TRENDING
# adx < 20 → RANGING (stays hardcoded)
```

**3c. Filter strategies in `run()` method** (in the `run()` method, around line 302, after `strategies = self._router.get_strategies(regime)`):
```python
strategies = self._router.get_strategies(regime)
if self._target_strategy:
    strategies = [s for s in strategies if s.name == self._target_strategy]
    if not strategies:
        continue  # no matching strategy for this regime
```

**3d. Skip confluence when target_strategy is set** (in `_evaluate_precomputed()` method, around where `check_confluence` is called — this is a DIFFERENT method from 3c):
```python
if self._target_strategy:
    # Single strategy isolation — skip confluence
    pass
else:
    confluence = check_confluence(signals)
    ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_backtest_target_strategy.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Run existing backtest tests to verify no regressions**

Run: `python -m pytest tests/test_backtest_engine_overhaul.py tests/test_backtest_ema.py tests/test_range_strategy.py -v`
Expected: All existing tests PASS

- [ ] **Step 6: Commit**

```bash
git add bot/backtest/engine.py tests/test_backtest_target_strategy.py
git commit -m "feat: add target_strategy filter and regime params to BacktestEngine"
```

---

## Chunk 3: Walk-forward per-strategy optimization

### Task 3: Walk-forward CHAMPION_DEFAULTS import + target_strategy threading

**Files:**
- Modify: `bot/learning/walk_forward.py:19-33` (_run_single_backtest), `bot/learning/walk_forward.py:47-56,99-108` (enqueue_trial), `bot/learning/walk_forward.py:77-159` (_run_optuna_window)

- [ ] **Step 1: Import CHAMPION_DEFAULTS and replace inline seeds**

In `bot/learning/walk_forward.py`, add import at top:
```python
from bot.strategy.adopted_universe import CHAMPION_DEFAULTS
```

Replace the two `enqueue_trial` blocks (lines 47-56 and 99-108) with:
```python
study.enqueue_trial(dict(CHAMPION_DEFAULTS))
```

- [ ] **Step 2: Add `target_strategy` param to `_run_single_backtest`**

```python
def _run_single_backtest(candles, params, target_strategy=None):
    """Top-level function so ProcessPoolExecutor can pickle it."""
    import time
    t0 = time.perf_counter()
    from bot.backtest.engine import BacktestEngine
    engine = BacktestEngine(
        candles, strategy_params=params, slippage_pct=0.001,
        target_strategy=target_strategy,
    )
    result = engine.run()
    ...
```

- [ ] **Step 3: Add `target_strategy` param to `_run_optuna_window`**

Change signature from:
```python
def _run_optuna_window(train_candles, test_candles, max_workers):
```
to:
```python
def _run_optuna_window(train_candles, test_candles, max_workers, target_strategy=None):
```

Update the `pool.submit` call inside to pass `target_strategy`:
```python
pool.submit(_run_single_backtest, train_candles, params, target_strategy)
```

Update the test engine creation (line 156):
```python
test_engine = BacktestEngine(
    test_candles, strategy_params=best_params, slippage_pct=0.001,
    target_strategy=target_strategy,
)
```

- [ ] **Step 4: Run existing walk-forward tests**

Run: `python -m pytest tests/test_walk_forward.py tests/test_wf_candidates.py -v`
Expected: All PASS (target_strategy defaults to None, no behavior change)

- [ ] **Step 5: Commit**

```bash
git add bot/learning/walk_forward.py
git commit -m "feat: thread target_strategy through walk-forward + import CHAMPION_DEFAULTS"
```

### Task 4: Walk-forward `run_multi_per_strategy` + candle prefetch

**Files:**
- Modify: `bot/learning/walk_forward.py:247-329` (_execute, run_multi)

- [ ] **Step 1: Add `_execute_from_candles` method**

Add new method to `WalkForwardOptimizer` that takes pre-fetched candles and slices per window:

```python
async def _execute_from_candles(
    self, all_candles, symbol: str = "BTC-EUR", target_strategy: str = None,
) -> WFResult:
    """Execute walk-forward using pre-fetched candles (no API calls)."""
    import asyncio
    import os
    from bot.backtest.engine import BacktestEngine

    total_days = TRAIN_DAYS + TEST_DAYS * MIN_WINDOWS
    windows_spec = self._generate_windows(total_days)

    if len(windows_spec) < MIN_WINDOWS:
        logger.warning("Walk-forward: insufficient windows for %s", symbol)
        return WFResult()

    candles_per_day = len(all_candles) / total_days if total_days > 0 else 288

    loop = asyncio.get_running_loop()
    wf_windows: List[WFWindow] = []
    max_workers = os.cpu_count() or 4

    for i, ws in enumerate(windows_spec):
        # Slice candles by day offset
        train_start = int(ws["train_start_day"] * candles_per_day)
        train_end = int(ws["train_end_day"] * candles_per_day)
        test_start = int(ws["test_start_day"] * candles_per_day)
        test_end = int(ws["test_end_day"] * candles_per_day)

        train_candles = all_candles[train_start:train_end]
        test_candles = all_candles[test_start:test_end]

        if not train_candles or not test_candles:
            continue

        strat_label = f" [{target_strategy}]" if target_strategy else ""
        logger.info(
            "Walk-forward [%s]%s: window %d/%d (%d train, %d test candles, %d Optuna trials across %d cores)...",
            symbol, strat_label, i + 1, len(windows_spec),
            len(train_candles), len(test_candles), OPTUNA_TRIALS, max_workers,
        )

        best_params, test_result = await loop.run_in_executor(
            None, _run_optuna_window, train_candles, test_candles, max_workers, target_strategy,
        )

        logger.info(
            "Walk-forward [%s]%s: window %d/%d done — sharpe=%.2f, pnl=€%.2f",
            symbol, strat_label, i + 1, len(windows_spec),
            test_result.sharpe_ratio, test_result.total_pnl,
        )

        wf_windows.append(WFWindow(
            sharpe=test_result.sharpe_ratio,
            pnl=test_result.total_pnl,
            params=best_params,
            profit_per_fee=getattr(test_result, "profit_per_fee", 0.0),
        ))

    adopted = self._should_adopt(wf_windows)
    avg_sharpe = statistics.mean(w.sharpe for w in wf_windows) if wf_windows else 0
    avg_pnl = statistics.mean(w.pnl for w in wf_windows) if wf_windows else 0
    sharpe_std = statistics.stdev(w.sharpe for w in wf_windows) if len(wf_windows) > 1 else 0

    best_params = None
    if adopted and wf_windows:
        best_window = max(wf_windows, key=lambda w: w.sharpe)
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

- [ ] **Step 2: Add `run_multi_per_strategy` method**

```python
async def run_multi_per_strategy(
    self, candle_fetcher_factory=None, symbols: List[str] = None, strategies: List[str] = None,
) -> Dict[str, Dict[str, WFResult]]:
    """Run walk-forward for each strategy-symbol combo.

    Fetches candles once per symbol, then runs optimization for each strategy.
    Returns: {symbol: {strategy_name: WFResult}}
    """
    from bot.strategy.adopted_universe import ALL_STRATEGIES

    if not symbols:
        symbols = ["BTC-EUR"]
    if not strategies:
        strategies = ALL_STRATEGIES

    self._running = True
    results: Dict[str, Dict[str, WFResult]] = {}

    try:
        for symbol in symbols:
            logger.info("Walk-forward: fetching full candle range for %s", symbol)

            # Fetch all candles once for this symbol
            total_days = TRAIN_DAYS + TEST_DAYS * MIN_WINDOWS
            if candle_fetcher_factory:
                fetcher = candle_fetcher_factory(symbol)
                all_candles = await fetcher(0, total_days)
            else:
                all_candles = []

            if not all_candles:
                logger.warning("Walk-forward: no candles for %s — skipping", symbol)
                continue

            logger.info(
                "Walk-forward: got %d candles for %s — running %d strategies",
                len(all_candles), symbol, len(strategies),
            )

            results[symbol] = {}
            for strategy in strategies:
                result = await self._execute_from_candles(
                    all_candles, symbol, target_strategy=strategy,
                )
                results[symbol][strategy] = result
                self._results_by_symbol[f"{symbol}:{strategy}"] = result
                self._latest_result = result

        return results
    finally:
        self._running = False
```

- [ ] **Step 3: Update `result_for_symbol` to return per-strategy results**

The existing `result_for_symbol(symbol)` returns a single `WFResult`. Update it to support the composite key format and add a new method for the nested structure:

```python
def result_for_symbol(self, symbol: str) -> Optional[WFResult]:
    """Backward-compat: return first result matching this symbol."""
    # Check composite keys first (new format)
    for key, val in self._results_by_symbol.items():
        if key.startswith(f"{symbol}:"):
            return val
    # Fall back to direct key (old format)
    return self._results_by_symbol.get(symbol)

def results_for_symbol(self, symbol: str) -> Dict[str, WFResult]:
    """Return all per-strategy results for a symbol."""
    results = {}
    for key, val in self._results_by_symbol.items():
        if key.startswith(f"{symbol}:"):
            strategy = key.split(":", 1)[1]
            results[strategy] = val
    return results
```

- [ ] **Step 4: Run existing walk-forward tests**

Run: `python -m pytest tests/test_walk_forward.py tests/test_wf_candidates.py -v`
Expected: All PASS (old methods still exist, new ones are additive)

- [ ] **Step 5: Commit**

```bash
git add bot/learning/walk_forward.py
git commit -m "feat: add per-strategy walk-forward with candle prefetch"
```

---

## Chunk 4: Main.py wiring + results processing

### Task 5: Update `_process_walk_forward_results` and `_run_walk_forward`

**Files:**
- Modify: `bot/main.py:280-358` (_process_walk_forward_results), `bot/main.py:458-465` (_run_walk_forward)

- [ ] **Step 1: Import AdoptedUniverse and create singleton in main.py**

Add near the top imports:
```python
from bot.strategy.adopted_universe import AdoptedUniverse
```

Add singleton near other singletons (around line 80):
```python
_universe: Optional[AdoptedUniverse] = None

def _get_universe() -> AdoptedUniverse:
    global _universe
    if _universe is None:
        _universe = AdoptedUniverse()
    return _universe
```

- [ ] **Step 2: Wire universe to trading loop**

In `_get_trading_loop()` (around line 274), after creating the loop:
```python
_trading_loop._universe = _get_universe()
```

- [ ] **Step 3: Rewrite `_process_walk_forward_results`**

Replace the function body to handle `Dict[str, Dict[str, WFResult]]` and call `universe.update()`:

```python
async def _process_walk_forward_results(
    results, get_router_fn, trading_loop, discord, settings, universe,
):
    """Process per-strategy walk-forward results: update universe, log summary."""
    universe.update(results)

    # ── Final analysis summary ──
    logger.info("=" * 70)
    logger.info("WALK-FORWARD ANALYSIS COMPLETE")
    logger.info("=" * 70)

    total_combos = 0
    adopted_combos = 0
    for symbol, strat_results in results.items():
        parts = []
        for strat_name, result in strat_results.items():
            total_combos += 1
            status = "ADOPTED" if result.adopted else "NOT ADOPTED"
            if result.adopted:
                adopted_combos += 1
                parts.append(f"{strat_name}: {status} (sharpe={result.avg_oos_sharpe:.2f}, pnl=€{result.avg_oos_pnl:+.2f})")
            else:
                parts.append(f"{strat_name}: {status}")
        logger.info("  %-12s | %s", symbol, " | ".join(parts))

    logger.info("-" * 70)
    logger.info("  Adopted: %d symbols, %d strategy-combos out of %d",
                universe.count(), adopted_combos, total_combos)
    logger.info("=" * 70)

    # Send report to Discord
    try:
        await discord.send_walk_forward_report(results)
    except Exception as e:
        logger.warning("Failed to send walk-forward report to Discord: %s", e)

    # Auto-enable trading in paper mode after walk-forward completes
    if settings.paper_trading and not is_running():
        logger.info("Walk-forward complete — auto-enabling paper trading")
        await start_bot()
```

- [ ] **Step 4: Update `_run_walk_forward` to use `run_multi_per_strategy`**

```python
async def _run_walk_forward():
    symbols = get_tradeable_symbols() or ["BTC-EUR"]
    logger.info("Walk-forward: running for %d symbols (per-strategy)", len(symbols))
    results = await walk_forward.run_multi_per_strategy(
        _wf_candle_fetcher_factory, symbols,
    )
    await _process_walk_forward_results(
        results, _get_router, trading_loop, discord, settings, _get_universe(),
    )
```

- [ ] **Step 5: Update `_process_walk_forward_results` call sites**

Search for any other calls to `_process_walk_forward_results` and add the `universe` parameter.

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/ -x -q`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add bot/main.py
git commit -m "feat: wire AdoptedUniverse to main + per-strategy WF results"
```

---

## Chunk 5: Trading loop — universe gating + per-strategy params

### Task 6: `run_cycle` universe gating

**Files:**
- Modify: `bot/trading_loop.py:612-885` (run_cycle)

- [ ] **Step 1: Add universe check at start of symbol loop**

At the top of the `for symbol in tradeable_symbols` loop (around line 627), add:
```python
# Skip symbols not in adopted universe
if hasattr(self, '_universe') and self._universe is not None:
    if not self._universe.is_adopted(symbol):
        continue
```

- [ ] **Step 2: Add per-strategy params to stops, sizing, and fee gate**

After the universe check, get per-symbol strategy params for the winning signal. Import at top of file:
```python
from bot.strategy.adopted_universe import CHAMPION_DEFAULTS
```

Where `initial_stops` is called (line 758), pass params:
```python
# Get params from universe for winning strategy
if hasattr(self, '_universe') and self._universe is not None and best_signal:
    strat_params = self._universe.get_strategy_params(symbol, best_signal.strategy_name)
else:
    strat_params = CHAMPION_DEFAULTS

# initial_stops — pass per-strategy params
stop_loss, take_profit = initial_stops(
    entry_price, df_1h, direction=direction,
    atr_multiplier=strat_params["atr_multiplier"],
    rr_ratio=strat_params["rr_ratio"],
)
```

Where `fixed_fractional_size` is called (line 774):
```python
size_eur = fixed_fractional_size(
    equity=equity,
    base_risk_pct=strat_params["base_risk_pct"],
    stop_distance_pct=stop_distance_pct,
    atr_pct=atr_pct,
    median_atr_pct=median_atr_pct,
)
```

Where `check_fee_gate` is called:
```python
fee_result = check_fee_gate(
    position_size=size_eur,
    tp_distance_pct=tp_distance_pct,
    min_profit_multiple=strat_params["min_profit_multiple"],
)
```

- [ ] **Step 3: Filter strategies to enabled list**

Where strategies are retrieved from router (around line 686):
```python
strategies = router.get_strategies(regime)
if hasattr(self, '_universe') and self._universe is not None:
    enabled = self._universe.get_enabled_strategies(symbol)
    if enabled:
        strategies = [s for s in strategies if s.name in enabled]
```

- [ ] **Step 4: Replace global cooldown with per-strategy cooldown**

Where cooldown is checked (lines 644-653), replace `self._cooldown_seconds` with per-strategy value.

**Note:** The spec says to use the winning strategy's cooldown, but cooldown is checked BEFORE signal evaluation (to avoid unnecessary work). Using `min()` across enabled strategies is a justified deviation — it's slightly more permissive but avoids restructuring the control flow. The winning strategy's params are used for stops/sizing/fees after signal evaluation.

```python
if hasattr(self, '_universe') and self._universe is not None and self._universe.is_adopted(symbol):
    # Use the shortest cooldown among enabled strategies for this symbol
    enabled = self._universe.get_enabled_strategies(symbol)
    cooldown_s = min(
        self._universe.get_strategy_params(symbol, s).get("cooldown_hours", CHAMPION_DEFAULTS["cooldown_hours"]) * 3600
        for s in enabled
    ) if enabled else CHAMPION_DEFAULTS["cooldown_hours"] * 3600
else:
    cooldown_s = CHAMPION_DEFAULTS["cooldown_hours"] * 3600

if time.monotonic() - self._last_trade_closed.get(symbol, 0) < cooldown_s:
    continue
```

- [ ] **Step 5: Commit**

```bash
git add bot/trading_loop.py
git commit -m "feat: universe gating + per-strategy params in run_cycle"
```

### Task 7: `_evaluate_1h_signals` universe gating + confluence relaxation

**Files:**
- Modify: `bot/trading_loop.py:142-454` (_evaluate_1h_signals)

- [ ] **Step 1: Add universe check at start of method**

Near the top of `_evaluate_1h_signals` (around line 150):
```python
if hasattr(self, '_universe') and self._universe is not None:
    if not self._universe.is_adopted(symbol):
        return
```

- [ ] **Step 2: Filter strategies to enabled list**

Where strategies are collected for evaluation (around lines 283-300):
```python
if hasattr(self, '_universe') and self._universe is not None:
    enabled = self._universe.get_enabled_strategies(symbol)
    if enabled:
        # Only evaluate enabled strategies
        ...
```

- [ ] **Step 3: Relax confluence for single-strategy symbols**

At the confluence check (line 318):
```python
confluence = check_confluence(signals)
# Relax confluence when only 1 strategy is enabled
if hasattr(self, '_universe') and self._universe is not None:
    enabled = self._universe.get_enabled_strategies(symbol)
    if len(enabled) <= 1 and signals:
        # Single strategy — use its signal directly without confluence
        best = max(signals, key=lambda s: s.get("strength", 0))
        # ... proceed with this signal
    elif not confluence.triggered:
        return
else:
    if not confluence.triggered:
        return
```

- [ ] **Step 4: Pass per-strategy params to stops, sizing, fee gate**

Same pattern as Task 6: get `strat_params` from universe for the winning strategy (falling back to `CHAMPION_DEFAULTS`), pass to `initial_stops()` (line 332), `fixed_fractional_size()` (line 345), `check_fee_gate()` (line 356).

- [ ] **Step 5: Add per-strategy cooldown**

Same pattern as Task 6 Step 4: replace cooldown check with per-strategy value from universe. Where cooldown is checked in `_evaluate_1h_signals` (around lines 154-157):
```python
if hasattr(self, '_universe') and self._universe is not None and self._universe.is_adopted(symbol):
    enabled = self._universe.get_enabled_strategies(symbol)
    cooldown_s = min(
        self._universe.get_strategy_params(symbol, s).get("cooldown_hours", CHAMPION_DEFAULTS["cooldown_hours"]) * 3600
        for s in enabled
    ) if enabled else CHAMPION_DEFAULTS["cooldown_hours"] * 3600
else:
    cooldown_s = CHAMPION_DEFAULTS["cooldown_hours"] * 3600
```

- [ ] **Step 6: Commit**

```bash
git add bot/trading_loop.py
git commit -m "feat: universe gating + confluence relaxation in _evaluate_1h_signals"
```

### Task 8: Regime detection refactor in trading loop

**Files:**
- Modify: `bot/trading_loop.py:174-191,665-684` (inline regime detection)

- [ ] **Step 1: Replace inline regime detection in `_evaluate_1h_signals`**

Replace lines 174-191 with:

**NOTE:** Before writing this code, inspect `detect_regime()` in `bot/strategy/router.py` to check whether it computes `adx`/`atr` columns internally or expects them pre-computed on the DataFrame. If `detect_regime()` already computes them, skip the `compute_atr`/`compute_adx` lines below. If it reads from existing columns, keep them.

```python
from bot.strategy.router import detect_regime
# Compute adx/atr columns on 4h DataFrame IF detect_regime() expects them pre-computed
# (verify by reading detect_regime() — it may compute these internally)
df_4h["atr"] = compute_atr(df_4h, period=14)
df_4h["adx"] = compute_adx(df_4h, period=14)

# Use per-symbol regime params from universe
if hasattr(self, '_universe') and self._universe is not None and self._universe.is_adopted(symbol):
    rp = self._universe.get_regime_params(symbol)
    regime = detect_regime(
        df_4h,
        quiet_atr_threshold=rp["quiet_atr_threshold"],
        trending_adx_threshold=rp["regime_adx_threshold"],
    )
else:
    regime = detect_regime(df_4h)
```

- [ ] **Step 2: Replace inline regime detection in `run_cycle`**

Same refactor for lines 665-684.

- [ ] **Step 3: Run all tests**

Run: `python -m pytest tests/ -x -q`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add bot/trading_loop.py
git commit -m "refactor: use detect_regime() with per-symbol params in trading loop"
```

---

## Chunk 6: API + Discord + final integration

### Task 9: Analytics API — fix dead method + GET /universe

**Files:**
- Modify: `api/routers/analytics.py:197-233`

- [ ] **Step 1: Fix dead `update_hybrid_params` call and update POST endpoint**

In `POST /walk-forward/run` (lines 197-233):
1. Remove the dead `router.update_hybrid_params(...)` call (lines 225-229)
2. Rewrite the endpoint to call `run_multi_per_strategy()` and pass results through `universe.update()`:

```python
@router.post("/walk-forward/run")
async def run_walk_forward():
    """Manually trigger walk-forward optimization."""
    from bot.main import walk_forward, _wf_candle_fetcher_factory, _get_universe, get_tradeable_symbols
    symbols = get_tradeable_symbols() or ["BTC-EUR"]
    results = await walk_forward.run_multi_per_strategy(
        _wf_candle_fetcher_factory, symbols,
    )
    universe = _get_universe()
    universe.update(results)
    return {"status": "ok", "adopted_count": universe.count()}
```

- [ ] **Step 2: Add GET /universe endpoint**

```python
@router.get("/universe")
async def get_universe():
    """Return the current adopted universe."""
    from bot.main import _get_universe
    universe = _get_universe()
    symbols = {}
    total_combos = 0
    for sym in universe.adopted_symbols():
        strategies = {}
        for strat in universe.get_enabled_strategies(sym):
            sp = universe.get_strategy_params(sym, strat)
            # Access StrategyConfig for sharpe/pnl via the internal _adopted dict
            sc = universe._adopted[sym].strategies[strat]
            strategies[strat] = {
                "params": sp,
                "avg_sharpe": sc.avg_sharpe,
                "avg_pnl": sc.avg_pnl,
            }
            total_combos += 1
        symbols[sym] = {
            "regime_params": universe.get_regime_params(sym),
            "strategies": strategies,
        }
    return {
        "adopted_count": universe.count(),
        "strategy_combo_count": total_combos,
        "symbols": symbols,
    }
```

- [ ] **Step 3: Commit**

```bash
git add api/routers/analytics.py
git commit -m "fix: remove dead update_hybrid_params + add GET /universe endpoint"
```

### Task 10: Discord notifier update

**Files:**
- Modify: `bot/notifications/discord.py:133-183`

- [ ] **Step 1: Update `send_walk_forward_report` for nested structure**

The method currently expects `Dict[str, WFResult]`. Update to handle `Dict[str, Dict[str, WFResult]]`:

```python
async def send_walk_forward_report(self, results: dict) -> None:
    """Send walk-forward analysis summary to Discord."""
    lines = ["**Walk-Forward Analysis Complete**\n"]

    for symbol, strat_results in results.items():
        # Handle both old flat format and new nested format
        if hasattr(strat_results, "adopted"):
            # Old flat format: strat_results is a WFResult
            status = "ADOPTED" if strat_results.adopted else "NOT ADOPTED"
            lines.append(f"**{symbol}**: {status} (sharpe={strat_results.avg_oos_sharpe:.2f})")
        else:
            # New nested format: strat_results is Dict[str, WFResult]
            parts = []
            for strat_name, result in strat_results.items():
                if result.adopted:
                    parts.append(f"{strat_name} ✓")
                else:
                    parts.append(f"{strat_name} ✗")
            lines.append(f"**{symbol}**: {' | '.join(parts)}")

    ...  # rest of method unchanged (send to Discord)
```

- [ ] **Step 2: Commit**

```bash
git add bot/notifications/discord.py
git commit -m "feat: update discord notifier for per-strategy walk-forward results"
```

### Task 11: Full test suite + Docker rebuild

**Files:** None new

- [ ] **Step 1: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 2: Docker rebuild and start**

```bash
docker compose up -d --build bot
```

- [ ] **Step 3: Verify bot starts cleanly**

```bash
sleep 20 && docker compose logs --tail=30 bot
```
Expected: WebSocket connected, candles loaded, no errors. Universe starts empty (no trades until walk-forward completes).

- [ ] **Step 4: Commit any test fixes**

Stage only the specific files that were fixed (do NOT use `git add -A`):
```bash
git add <specific-files-that-changed>
git commit -m "fix: test adjustments for dynamic universe integration"
```
