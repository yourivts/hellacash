# Higher Timeframe Overhaul Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the trading engine for 1h signal generation, maker-first orders, and fee-aware entry gating to achieve consistent profitability on a sub-EUR 1,000 account.

**Architecture:** Dual-timeframe system — entry signals evaluated on 1h candle closes, stop/exit monitoring on 5m candles. Maker limit orders for entries and take-profits cut fees by 40%. A fee-aware gate rejects trades where expected profit doesn't exceed 3x total fees. Four redesigned strategies (orderflow, funding contrarian, range, squeeze) replace the current six.

**Tech Stack:** Python 3.11+, numpy, pandas, Optuna, pytest, Bitvavo REST/WebSocket API

**Spec:** `docs/superpowers/specs/2026-03-16-higher-timeframe-overhaul-design.md`

---

## File Structure

### New Files
| File | Responsibility |
|------|---------------|
| `bot/strategy/squeeze.py` | Volatility Squeeze strategy (BB compression → expansion) |
| `bot/strategy/confluence.py` | Confluence Gate meta-strategy (2+ strategies agree → bigger size) |
| `bot/execution/limit_order_manager.py` | Limit order lifecycle (place, monitor, timeout, cancel) |
| `bot/execution/__init__.py` | Package init |
| `bot/risk/fee_gate.py` | Fee-aware entry gate (reject if profit < 3x fees) |
| `tests/test_fee_gate.py` | Tests for fee-aware gate |
| `tests/test_squeeze_strategy.py` | Tests for Volatility Squeeze |
| `tests/test_confluence.py` | Tests for Confluence Gate |
| `tests/test_limit_order_manager.py` | Tests for limit order lifecycle |
| `tests/test_fixed_fractional.py` | Tests for new position sizing |

### Modified Files
| File | Changes |
|------|---------|
| `bot/config.py` | New params: base_risk_pct, min_profit_multiple, quiet_atr_threshold, regime_adx_threshold, weekend_filter_enabled; defaults: max_open_positions=10, max_daily_loss_eur=50 |
| `bot/risk/position_sizer.py` | Replace Kelly with fixed fractional + volatility scaling |
| `bot/risk/stop_loss.py` | Add activation_threshold param to trail_stop(), fee-adjusted TP |
| `bot/risk/engine.py` | Remove drawdown halt, add correlation guard, integrate fee gate |
| `bot/risk/drawdown_guard.py` | Remove hard halt, update daily loss default to 50 |
| `bot/strategy/router.py` | 4h regime detection with QUIET/NEUTRAL, new strategy roster |
| `bot/strategy/orderflow.py` | Redesign for 1h candle signals |
| `bot/strategy/funding_contrarian.py` | Loosen thresholds for 1h |
| `bot/strategy/range_trading.py` | Redesign for 1h bands, 72h max hold |
| `bot/backtest/engine.py` | SIGNAL_EVERY=12, maker fees, fill rate model, remove trend/breakout, new metrics |
| `bot/trading_loop.py` | 1h signal eval, 5m stop check, limit order integration |
| `bot/learning/walk_forward.py` | New param space, 180d/30d windows, per-strategy optimization |
| `bot/data_loader.py` | Add build_1h_candle() to CandleCache |
| `bot/main.py` | Wire new components, update scheduler cycles |
| `tests/test_position_sizer.py` | Update for fixed fractional |
| `tests/test_stop_loss.py` | Update for activation_threshold |
| `tests/test_strategy_router.py` | Update for 4h regime + QUIET/NEUTRAL |
| `tests/test_risk_engine.py` | Update for correlation guard, fee gate |
| `tests/test_drawdown_guard.py` | Update for removed hard halt |

---

## Chunk 1: Foundation Layer

### Task 1: Update Config Defaults

**Files:**
- Modify: `bot/config.py`
- Modify: `tests/test_config_validation.py`

- [ ] **Step 1: Update config.py with new parameters and defaults**

```python
# In bot/config.py, add these fields to the Settings class:

# Position sizing (replaces Kelly)
base_risk_pct: float = 3.0          # % of equity risked per trade
confluence_risk_pct: float = 4.5    # % when confluence gate triggers

# Fee-aware gate
min_profit_multiple: float = 3.0    # reject if expected_profit < N * total_fees

# Regime detection
quiet_atr_threshold: float = 1.0    # ATR% below this = QUIET (no trading)
regime_adx_threshold: float = 25.0  # ADX above this = TRENDING

# Weekend filter
weekend_filter_enabled: bool = False

# Change existing defaults:
max_open_positions: int = 10        # was 5
max_daily_loss_eur: float = 50.0    # was 200
```

- [ ] **Step 2: Update test_config_validation.py**

Add test verifying new defaults load correctly. Update any tests that assert old defaults (max_open_positions=5, max_daily_loss_eur=200).

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_config_validation.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add bot/config.py tests/test_config_validation.py
git commit -m "feat: update config defaults for higher timeframe overhaul"
```

---

### Task 2: Fixed Fractional Position Sizer

**Files:**
- Modify: `bot/risk/position_sizer.py`
- Create: `tests/test_fixed_fractional.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_fixed_fractional.py
from bot.risk.position_sizer import fixed_fractional_size

class TestFixedFractional:
    def test_basic_sizing(self):
        """3% risk on 1000 EUR with 2% stop, capped at 30% of equity."""
        size = fixed_fractional_size(
            equity=1000.0,
            base_risk_pct=3.0,
            stop_distance_pct=2.0,
            atr_pct=1.5,
            median_atr_pct=1.5,  # vol_scale = 1.0
        )
        # risk_eur = 1000 * 0.03 * 1.0 = 30
        # position = 30 / 0.02 = 1500 -- but capped at 30% of equity = 300
        assert abs(size - 300.0) < 0.01

    def test_volatility_scaling_up(self):
        """Higher ATR = bigger position (vol_scale > 1)."""
        size = fixed_fractional_size(
            equity=1000.0,
            base_risk_pct=3.0,
            stop_distance_pct=4.0,
            atr_pct=3.0,
            median_atr_pct=1.5,  # vol_scale = 2.0
        )
        # risk_eur = 1000 * 0.03 * 2.0 = 60
        # position = 60 / 0.04 = 1500, capped at 300
        assert abs(size - 300.0) < 0.01

    def test_volatility_scaling_down(self):
        """Lower ATR = smaller position (vol_scale < 1)."""
        size = fixed_fractional_size(
            equity=1000.0,
            base_risk_pct=3.0,
            stop_distance_pct=1.0,
            atr_pct=0.5,
            median_atr_pct=1.5,  # vol_scale = 0.5 (clamped from 0.33)
        )
        # risk_eur = 1000 * 0.03 * 0.5 = 15
        # position = 15 / 0.01 = 1500, capped at 300
        assert abs(size - 300.0) < 0.01

    def test_minimum_position(self):
        """Never go below EUR 10 (Bitvavo minimum)."""
        size = fixed_fractional_size(
            equity=100.0,
            base_risk_pct=1.0,
            stop_distance_pct=20.0,
            atr_pct=1.0,
            median_atr_pct=1.0,
        )
        # risk_eur = 100 * 0.01 * 1.0 = 1
        # position = 1 / 0.20 = 5, below minimum
        assert abs(size - 10.0) < 0.01

    def test_max_cap_30_percent(self):
        """Position capped at 30% of equity."""
        size = fixed_fractional_size(
            equity=500.0,
            base_risk_pct=5.0,
            stop_distance_pct=0.5,
            atr_pct=1.5,
            median_atr_pct=1.5,
        )
        # risk_eur = 500 * 0.05 * 1.0 = 25
        # position = 25 / 0.005 = 5000, capped at 150
        assert abs(size - 150.0) < 0.01

    def test_confluence_bonus(self):
        """Confluence flag increases risk to 4.5%."""
        size = fixed_fractional_size(
            equity=1000.0,
            base_risk_pct=4.5,
            stop_distance_pct=3.0,
            atr_pct=1.5,
            median_atr_pct=1.5,
        )
        # risk_eur = 1000 * 0.045 * 1.0 = 45
        # position = 45 / 0.03 = 1500, capped at 300
        assert abs(size - 300.0) < 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_fixed_fractional.py -v`
Expected: FAIL (function not found)

- [ ] **Step 3: Implement fixed_fractional_size**

Replace contents of `bot/risk/position_sizer.py`:

```python
"""Position sizing: fixed fractional with volatility scaling."""
from __future__ import annotations

MIN_POSITION_EUR = 10.0
MAX_POSITION_PCT = 0.30  # 30% of equity


def fixed_fractional_size(
    equity: float,
    base_risk_pct: float,
    stop_distance_pct: float,
    atr_pct: float,
    median_atr_pct: float,
) -> float:
    """Compute position size using fixed fractional with volatility scaling.

    Args:
        equity: Current portfolio value in EUR.
        base_risk_pct: Percentage of equity to risk per trade (e.g., 3.0).
        stop_distance_pct: Stop distance as % of price (e.g., 2.0 for 2%).
        atr_pct: Current ATR as % of price.
        median_atr_pct: Median ATR% over lookback period.

    Returns:
        Position size in EUR.
    """
    if equity <= 0 or stop_distance_pct <= 0 or median_atr_pct <= 0:
        return MIN_POSITION_EUR

    # Volatility scaling: bigger positions in volatile markets
    vol_scale = max(0.5, min(2.0, atr_pct / median_atr_pct))

    risk_eur = equity * (base_risk_pct / 100.0) * vol_scale
    position_size = risk_eur / (stop_distance_pct / 100.0)

    # Hard caps
    position_size = min(position_size, equity * MAX_POSITION_PCT)
    position_size = max(position_size, MIN_POSITION_EUR)

    return round(position_size, 2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_fixed_fractional.py -v`
Expected: PASS

- [ ] **Step 5: Update existing test files for Kelly removal**

`tests/test_position_sizer.py` and `tests/test_kelly_clamp.py` test the old Kelly sizing. Either:
- Delete both files if they only test `kelly_size()`, OR
- Rewrite them to test `fixed_fractional_size()` instead (but `test_fixed_fractional.py` already covers this, so deletion is preferred)

```bash
# Check if files exist and remove if Kelly-only:
git rm tests/test_position_sizer.py tests/test_kelly_clamp.py 2>/dev/null || true
```

- [ ] **Step 6: Run all tests to verify no references to old kelly_size remain**

Run: `python -m pytest tests/ -v -k "position_sizer or kelly"` — should only run `test_fixed_fractional.py` tests.

- [ ] **Step 7: Commit**

```bash
git add bot/risk/position_sizer.py tests/test_fixed_fractional.py
git rm --cached tests/test_position_sizer.py tests/test_kelly_clamp.py 2>/dev/null || true
git commit -m "feat: replace Kelly with fixed fractional position sizing"
```

---

### Task 3: Stop Loss Updates (Activation Threshold + Fee-Adjusted TP)

**Files:**
- Modify: `bot/risk/stop_loss.py`
- Modify: `tests/test_stop_loss.py`

- [ ] **Step 1: Write failing tests for activation threshold**

Add to `tests/test_stop_loss.py`:

```python
class TestTrailStopActivation:
    def test_no_trail_before_activation(self):
        """Trail stop should NOT move until price moves 1.5x stop distance in profit."""
        from bot.risk.stop_loss import trail_stop
        # LONG: entry=100, stop=97 (3% stop distance)
        # Price at 103 = 3% profit = 1.0x stop distance (below 1.5x)
        new_stop = trail_stop(
            current_price=103.0,
            highest_price=103.0,
            current_stop=97.0,
            atr_value=1.0,
            direction="LONG",
            activation_multiplier=3.0,
            activation_threshold=1.5,
            entry_price=100.0,
        )
        assert new_stop == 97.0  # unchanged

    def test_trail_after_activation(self):
        """Trail stop SHOULD move after 1.5x stop distance in profit."""
        from bot.risk.stop_loss import trail_stop
        # LONG: entry=100, stop=97 (3% stop distance)
        # Price at 104.5 = 4.5% profit = 1.5x stop distance (activated!)
        new_stop = trail_stop(
            current_price=104.5,
            highest_price=104.5,
            current_stop=97.0,
            atr_value=1.0,
            direction="LONG",
            activation_multiplier=2.0,  # trail at 2x ATR
            activation_threshold=1.5,
            entry_price=100.0,
        )
        # New trail = 104.5 - 2.0 = 102.5, which is > 97
        assert new_stop == 102.5


class TestFeeAdjustedTP:
    def test_tp_includes_fees(self):
        """TP distance should include fee compensation."""
        from bot.risk.stop_loss import initial_stops
        entry = 100.0
        atr = 2.0
        sl, tp = initial_stops(
            entry_price=entry,
            atr_value=atr,
            direction="LONG",
            atr_multiplier=3.5,
            rr_ratio=2.5,
            total_fee_pct=0.40,
        )
        # stop = 100 - 3.5*2 = 93.0, risk = 7.0
        # tp = 100 + 2.5*7.0 + 0.40/100*100 = 100 + 17.5 + 0.4 = 117.9
        assert abs(tp - 117.9) < 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_stop_loss.py::TestTrailStopActivation -v`
Expected: FAIL

- [ ] **Step 3: Update bot/risk/stop_loss.py**

**IMPORTANT:** Preserve existing parameter order for backward compatibility. Add new params at the end with defaults. Keep `initial_stops()` returning a tuple (existing callers destructure it as `sl, tp = initial_stops(...)`).

Read the existing file first to see the current signatures:
- `trail_stop(current_price, highest_price, current_stop, atr_value, direction, activation_multiplier)` — add `entry_price=0.0` at end
- `initial_stops(entry_price, df, atr_multiplier, rr_ratio, direction)` — add `total_fee_pct=0.0` at end, keep tuple return

```python
def trail_stop(
    current_price: float,
    highest_price: float,
    current_stop: float,
    atr_value: float,
    direction: str = "LONG",
    activation_multiplier: float = 3.0,
    activation_threshold: float = 1.5,
    entry_price: float = 0.0,
) -> float:
    """Trailing stop with activation threshold.

    Preserves existing parameter order. New params added at end.
    Stop only starts trailing after price moves activation_threshold * stop_distance
    in profit direction. Before that, original stop is maintained.
    """
    if entry_price > 0 and activation_threshold > 0:
        stop_distance = abs(entry_price - current_stop)
        if direction == "LONG":
            profit = highest_price - entry_price
        else:
            profit = entry_price - highest_price

        if stop_distance > 0 and profit < stop_distance * activation_threshold:
            return current_stop

    trail_dist = atr_value * activation_multiplier
    if direction == "LONG":
        new_stop = highest_price - trail_dist
        return max(current_stop, new_stop)
    else:
        new_stop = highest_price + trail_dist
        return min(current_stop, new_stop) if current_stop > 0 else new_stop
```

For `initial_stops`, add `total_fee_pct` param but keep tuple return type. The existing signature accepts a DataFrame `df` for ATR computation — keep that path working but also accept a pre-computed `atr_value` kwarg:

```python
def initial_stops(
    entry_price: float,
    df=None,
    atr_multiplier: float = 3.0,
    rr_ratio: float = 2.0,
    direction: str = "LONG",
    total_fee_pct: float = 0.0,
    atr_value: float = 0.0,
) -> tuple:
    """Calculate initial stop-loss and take-profit levels.

    Returns (stop_loss, take_profit) tuple.
    If atr_value is provided, uses it directly. Otherwise computes from df.
    total_fee_pct: added to TP distance so R:R is net of fees.
    """
    if atr_value <= 0 and df is not None:
        from bot.indicators.volatility import atr as compute_atr
        atr_series = compute_atr(df)
        atr_value = float(atr_series.iloc[-1])

    if direction == "LONG":
        stop_loss = entry_price - atr_multiplier * atr_value
        risk = entry_price - stop_loss
        fee_compensation = entry_price * (total_fee_pct / 100.0)
        take_profit = entry_price + rr_ratio * risk + fee_compensation
    else:
        stop_loss = entry_price + atr_multiplier * atr_value
        risk = stop_loss - entry_price
        fee_compensation = entry_price * (total_fee_pct / 100.0)
        take_profit = entry_price - rr_ratio * risk - fee_compensation

    return round(stop_loss, 8), round(take_profit, 8)
```

**Existing callers remain valid** — `sl, tp = initial_stops(price, df, ...)` still works. New callers can use `sl, tp = initial_stops(price, atr_value=atr, total_fee_pct=0.40, ...)`.

- [ ] **Step 4: Run all stop_loss tests**

Run: `python -m pytest tests/test_stop_loss.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add bot/risk/stop_loss.py tests/test_stop_loss.py
git commit -m "feat: add trailing stop activation threshold and fee-adjusted TP"
```

---

### Task 4: Fee-Aware Entry Gate

**Files:**
- Create: `bot/risk/fee_gate.py`
- Create: `tests/test_fee_gate.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_fee_gate.py
from bot.risk.fee_gate import check_fee_gate, FeeGateResult


class TestFeeGate:
    def test_passes_when_profit_exceeds_fees(self):
        """Trade with 3x fee coverage should pass."""
        result = check_fee_gate(
            position_size=100.0,
            tp_distance_pct=2.0,
            maker_fee_pct=0.15,
            taker_fee_pct=0.25,
            min_profit_multiple=3.0,
            is_short=False,
            expected_hold_hours=0,
        )
        # expected_profit = 100 * 0.02 = 2.0
        # total_fee = 100*0.0015 + 100*0.0025 = 0.15 + 0.25 = 0.40
        # ratio = 2.0 / 0.40 = 5.0 >= 3.0
        assert result.approved is True
        assert result.fee_ratio >= 3.0

    def test_rejects_when_profit_below_threshold(self):
        """Trade with low TP distance should be rejected."""
        result = check_fee_gate(
            position_size=50.0,
            tp_distance_pct=0.5,
            maker_fee_pct=0.15,
            taker_fee_pct=0.25,
            min_profit_multiple=3.0,
            is_short=False,
            expected_hold_hours=0,
        )
        # expected_profit = 50 * 0.005 = 0.25
        # total_fee = 50*0.0015 + 50*0.0025 = 0.075 + 0.125 = 0.20
        # ratio = 0.25 / 0.20 = 1.25 < 3.0
        assert result.approved is False

    def test_short_includes_borrow_fee(self):
        """Short trades include borrowing cost in fee calculation."""
        result = check_fee_gate(
            position_size=100.0,
            tp_distance_pct=2.0,
            maker_fee_pct=0.15,
            taker_fee_pct=0.25,
            min_profit_multiple=3.0,
            is_short=True,
            expected_hold_hours=48,
            annual_borrow_rate=0.03,
        )
        # borrow_fee = 100 * (0.03/8760) * 48 = 0.0164
        # total_fee = 0.40 + 0.0164 = 0.4164
        # ratio = 2.0 / 0.4164 = 4.80
        assert result.approved is True
        assert result.total_fees > 0.40  # includes borrow

    def test_zero_position_rejects(self):
        result = check_fee_gate(
            position_size=0.0,
            tp_distance_pct=2.0,
            maker_fee_pct=0.15,
            taker_fee_pct=0.25,
            min_profit_multiple=3.0,
        )
        assert result.approved is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_fee_gate.py -v`
Expected: FAIL

- [ ] **Step 3: Implement fee gate**

```python
# bot/risk/fee_gate.py
"""Fee-aware entry gate: rejects trades where expected profit < N * total fees."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FeeGateResult:
    approved: bool
    fee_ratio: float  # expected_profit / total_fees
    total_fees: float
    expected_profit: float
    reason: str = ""


def check_fee_gate(
    position_size: float,
    tp_distance_pct: float,
    maker_fee_pct: float = 0.15,
    taker_fee_pct: float = 0.25,
    min_profit_multiple: float = 3.0,
    is_short: bool = False,
    expected_hold_hours: float = 0,
    annual_borrow_rate: float = 0.03,
) -> FeeGateResult:
    """Check if a trade's expected profit justifies its fee cost.

    Args:
        position_size: Trade size in EUR.
        tp_distance_pct: Take-profit distance as % (e.g., 2.0 for 2%).
        maker_fee_pct: Maker fee as % (entry).
        taker_fee_pct: Taker fee as % (worst-case exit via SL).
        min_profit_multiple: Minimum ratio of profit to fees.
        is_short: Whether this is a short trade (adds borrow fees).
        expected_hold_hours: Expected hold time for borrow calculation.
        annual_borrow_rate: Annual borrowing rate for shorts.
    """
    if position_size <= 0 or tp_distance_pct <= 0:
        return FeeGateResult(
            approved=False, fee_ratio=0.0, total_fees=0.0,
            expected_profit=0.0, reason="invalid position or TP"
        )

    entry_fee = position_size * (maker_fee_pct / 100.0)
    exit_fee = position_size * (taker_fee_pct / 100.0)
    total_fees = entry_fee + exit_fee

    if is_short and expected_hold_hours > 0:
        hourly_rate = annual_borrow_rate / 8760.0
        borrow_fee = position_size * hourly_rate * expected_hold_hours
        total_fees += borrow_fee

    expected_profit = position_size * (tp_distance_pct / 100.0)
    fee_ratio = expected_profit / total_fees if total_fees > 0 else 999.0

    approved = fee_ratio >= min_profit_multiple
    reason = "" if approved else f"fee_ratio {fee_ratio:.2f} < {min_profit_multiple}"

    return FeeGateResult(
        approved=approved,
        fee_ratio=round(fee_ratio, 2),
        total_fees=round(total_fees, 4),
        expected_profit=round(expected_profit, 4),
        reason=reason,
    )
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_fee_gate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add bot/risk/fee_gate.py tests/test_fee_gate.py
git commit -m "feat: add fee-aware entry gate"
```

---

## Chunk 2: Strategy Layer

### Task 5: Regime Detection on 4h + Router Update

**Files:**
- Modify: `bot/strategy/router.py`
- Modify: `tests/test_strategy_router.py`

- [ ] **Step 1: Write failing tests for new regime detection**

Replace `tests/test_strategy_router.py` contents:

```python
# tests/test_strategy_router.py
import pandas as pd
import numpy as np
from bot.strategy.router import detect_regime, Regime, StrategyRouter


def _make_df(adx: float, atr_pct: float, rows: int = 50) -> pd.DataFrame:
    """Create a minimal 4h DataFrame with given ADX and ATR%."""
    close = np.full(rows, 100.0)
    return pd.DataFrame({
        "close": close,
        "adx": np.full(rows, adx),
        "atr": np.full(rows, atr_pct),  # ATR as absolute; atr_pct computed internally
        "high": close * (1 + atr_pct / 100),
        "low": close * (1 - atr_pct / 100),
    })


class TestRegimeDetection:
    def test_quiet_regime(self):
        """ATR% < 1.0 → QUIET regardless of ADX."""
        df = _make_df(adx=30.0, atr_pct=0.5)
        assert detect_regime(df) == Regime.QUIET

    def test_volatile_regime(self):
        """ATR% > 4.0 → VOLATILE (checked before ADX)."""
        df = _make_df(adx=30.0, atr_pct=5.0)
        assert detect_regime(df) == Regime.VOLATILE

    def test_trending_regime(self):
        """ADX > 25, ATR 1.0-4.0 → TRENDING."""
        df = _make_df(adx=30.0, atr_pct=2.0)
        assert detect_regime(df) == Regime.TRENDING

    def test_ranging_regime(self):
        """ADX < 20, ATR 1.0-4.0 → RANGING."""
        df = _make_df(adx=15.0, atr_pct=2.0)
        assert detect_regime(df) == Regime.RANGING

    def test_neutral_regime(self):
        """ADX 20-25, ATR 1.0-4.0 → NEUTRAL."""
        df = _make_df(adx=22.0, atr_pct=2.0)
        assert detect_regime(df) == Regime.NEUTRAL


class TestStrategyRouter:
    def test_quiet_returns_empty(self):
        """QUIET regime returns no strategies."""
        router = StrategyRouter()
        strats = router.get_strategies(Regime.QUIET)
        assert len(strats) == 0

    def test_trending_strategies(self):
        """TRENDING returns orderflow, funding, squeeze."""
        router = StrategyRouter()
        strats = router.get_strategies(Regime.TRENDING)
        names = {s.name for s in strats}
        assert names == {"orderflow", "funding_contrarian", "squeeze"}

    def test_ranging_strategies(self):
        """RANGING returns range, orderflow, funding."""
        router = StrategyRouter()
        strats = router.get_strategies(Regime.RANGING)
        names = {s.name for s in strats}
        assert names == {"range", "orderflow", "funding_contrarian"}

    def test_volatile_strategies(self):
        """VOLATILE returns funding, squeeze."""
        router = StrategyRouter()
        strats = router.get_strategies(Regime.VOLATILE)
        names = {s.name for s in strats}
        assert names == {"funding_contrarian", "squeeze"}

    def test_neutral_strategies(self):
        """NEUTRAL returns funding, orderflow."""
        router = StrategyRouter()
        strats = router.get_strategies(Regime.NEUTRAL)
        names = {s.name for s in strats}
        assert names == {"funding_contrarian", "orderflow"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_strategy_router.py -v`
Expected: FAIL

- [ ] **Step 3: Rewrite bot/strategy/router.py**

```python
"""Strategy router: regime detection on 4h data, strategy dispatch."""
from __future__ import annotations

from enum import Enum
from typing import List

import pandas as pd

from bot.strategy.base import BaseStrategy
from bot.strategy.orderflow import OrderFlowStrategy
from bot.strategy.funding_contrarian import FundingContrarianStrategy
from bot.strategy.range_trading import RangeStrategy
from bot.strategy.squeeze import SqueezeStrategy


class Regime(str, Enum):
    QUIET = "quiet"
    VOLATILE = "volatile"
    TRENDING = "trending"
    RANGING = "ranging"
    NEUTRAL = "neutral"


def detect_regime(
    df_4h: pd.DataFrame,
    quiet_atr_threshold: float = 1.0,
    volatile_atr_threshold: float = 4.0,
    trending_adx_threshold: float = 25.0,
    ranging_adx_threshold: float = 20.0,
) -> Regime:
    """Detect market regime from 4h candle data.

    Evaluation order (first match wins):
    1. QUIET:    ATR% < quiet_atr_threshold
    2. VOLATILE: ATR% > volatile_atr_threshold
    3. TRENDING: ADX > trending_adx_threshold
    4. RANGING:  ADX < ranging_adx_threshold
    5. NEUTRAL:  everything else (ADX 20-25, ATR 1.0-4.0)
    """
    if df_4h is None or len(df_4h) < 2:
        return Regime.NEUTRAL

    last = df_4h.iloc[-1]
    adx = float(last.get("adx", 20.0))

    # Compute ATR% from the last row
    close = float(last.get("close", 1.0))
    atr = float(last.get("atr", 0.0))
    atr_pct = (atr / close * 100.0) if close > 0 else 0.0

    # Priority-ordered evaluation
    if atr_pct < quiet_atr_threshold:
        return Regime.QUIET
    if atr_pct > volatile_atr_threshold:
        return Regime.VOLATILE
    if adx > trending_adx_threshold:
        return Regime.TRENDING
    if adx < ranging_adx_threshold:
        return Regime.RANGING
    return Regime.NEUTRAL


class StrategyRouter:
    """Routes to applicable strategies based on market regime."""

    def __init__(self) -> None:
        self._orderflow = OrderFlowStrategy()
        self._funding = FundingContrarianStrategy()
        self._range = RangeStrategy()
        self._squeeze = SqueezeStrategy()

    def get_strategies(self, regime: Regime) -> List[BaseStrategy]:
        """Return strategies applicable for the given regime."""
        if regime == Regime.QUIET:
            return []
        elif regime == Regime.VOLATILE:
            return [self._funding, self._squeeze]
        elif regime == Regime.TRENDING:
            return [self._orderflow, self._funding, self._squeeze]
        elif regime == Regime.RANGING:
            return [self._range, self._orderflow, self._funding]
        else:  # NEUTRAL
            return [self._funding, self._orderflow]

    def update_params(self, **kwargs) -> None:
        """Update strategy parameters (called by walk-forward)."""
        for strat in [self._orderflow, self._funding, self._range, self._squeeze]:
            for key, val in kwargs.items():
                if hasattr(strat, key):
                    setattr(strat, key, val)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_strategy_router.py -v`
Expected: PASS (may fail if squeeze.py doesn't exist yet — create stub first)

- [ ] **Step 5: Create squeeze stub if needed**

If tests fail because `bot/strategy/squeeze.py` doesn't exist, create a minimal stub:

```python
# bot/strategy/squeeze.py
"""Volatility Squeeze strategy stub — full implementation in Task 9."""
from bot.strategy.base import BaseStrategy, MarketContext, Signal


class SqueezeStrategy(BaseStrategy):
    name = "squeeze"

    def generate_signal(self, ctx: MarketContext) -> Signal:
        return Signal(
            symbol=ctx.symbol, direction="NEUTRAL", strength=0.0,
            strategy_name=self.name, technical_score=0.0,
        )
```

- [ ] **Step 6: Run tests again and commit**

Run: `python -m pytest tests/test_strategy_router.py -v`
Expected: PASS

```bash
git add bot/strategy/router.py bot/strategy/squeeze.py tests/test_strategy_router.py
git commit -m "feat: 4h regime detection with QUIET/NEUTRAL, updated strategy roster"
```

---

### Task 6: Orderflow Strategy Redesign (1h signals)

**Files:**
- Modify: `bot/strategy/orderflow.py`
- Modify: existing tests or add new ones

- [ ] **Step 1: Write failing tests for 1h orderflow signals**

```python
# tests/test_orderflow_1h.py
from bot.strategy.orderflow import OrderFlowStrategy
from bot.strategy.base import MarketContext, Signal


def _make_ctx(
    body_ratio: float = 0.2,
    wick_lower_ratio: float = 0.5,
    wick_upper_ratio: float = 0.1,
    volume_surge: float = 2.0,
    rsi: float = 50.0,
    cmf: float = 0.1,
    obv_divergence: float = 1.0,
) -> dict:
    """Return kwargs for the strategy's evaluate_1h method."""
    return dict(
        body_ratio=body_ratio,
        wick_lower_ratio=wick_lower_ratio,
        wick_upper_ratio=wick_upper_ratio,
        volume_surge=volume_surge,
        rsi_1h=rsi,
        cmf=cmf,
        obv_divergence=obv_divergence,
    )


class TestOrderflow1h:
    def test_buying_absorption_long(self):
        """Small body + long lower wick + high volume + RSI mid = LONG."""
        strat = OrderFlowStrategy()
        direction, strength = strat.evaluate_1h(**_make_ctx(
            body_ratio=0.15, wick_lower_ratio=0.55, volume_surge=1.8,
            rsi=45.0, cmf=0.15, obv_divergence=1.0,
        ))
        assert direction == "LONG"
        assert strength > 0.0

    def test_selling_absorption_short(self):
        """Small body + long upper wick + high volume = SHORT."""
        strat = OrderFlowStrategy()
        direction, strength = strat.evaluate_1h(**_make_ctx(
            body_ratio=0.15, wick_upper_ratio=0.55, wick_lower_ratio=0.1,
            volume_surge=1.8, rsi=55.0, cmf=-0.15, obv_divergence=-1.0,
        ))
        assert direction == "SHORT"
        assert strength > 0.0

    def test_rejects_large_body(self):
        """Body > 30% of range = no absorption pattern."""
        strat = OrderFlowStrategy()
        direction, _ = strat.evaluate_1h(**_make_ctx(body_ratio=0.40))
        assert direction == "NEUTRAL"

    def test_rejects_low_volume(self):
        """Volume surge < 1.5 = no signal."""
        strat = OrderFlowStrategy()
        direction, _ = strat.evaluate_1h(**_make_ctx(volume_surge=1.2))
        assert direction == "NEUTRAL"

    def test_rejects_extreme_rsi(self):
        """RSI outside 30-70 = absorption less reliable."""
        strat = OrderFlowStrategy()
        direction, _ = strat.evaluate_1h(**_make_ctx(rsi=75.0))
        assert direction == "NEUTRAL"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_orderflow_1h.py -v`
Expected: FAIL

- [ ] **Step 3: Rewrite bot/strategy/orderflow.py**

```python
"""Order Flow Absorption strategy — redesigned for 1h candles."""
from __future__ import annotations

from bot.strategy.base import BaseStrategy, MarketContext, Signal


class OrderFlowStrategy(BaseStrategy):
    name = "orderflow"

    # Thresholds for 1h candle absorption detection
    MAX_BODY_RATIO = 0.30
    MIN_WICK_RATIO = 0.40
    MIN_VOLUME_SURGE = 1.5
    RSI_LOW = 30.0
    RSI_HIGH = 70.0

    def evaluate_1h(
        self,
        body_ratio: float,
        wick_lower_ratio: float,
        wick_upper_ratio: float,
        volume_surge: float,
        rsi_1h: float,
        cmf: float,
        obv_divergence: float,
    ) -> tuple[str, float]:
        """Evaluate absorption pattern on 1h candle data.

        Returns (direction, strength) tuple.
        """
        # Gate checks
        if body_ratio > self.MAX_BODY_RATIO:
            return "NEUTRAL", 0.0
        if volume_surge < self.MIN_VOLUME_SURGE:
            return "NEUTRAL", 0.0
        if rsi_1h < self.RSI_LOW or rsi_1h > self.RSI_HIGH:
            return "NEUTRAL", 0.0

        # Detect absorption direction
        buying_absorption = wick_lower_ratio >= self.MIN_WICK_RATIO
        selling_absorption = wick_upper_ratio >= self.MIN_WICK_RATIO

        if not buying_absorption and not selling_absorption:
            return "NEUTRAL", 0.0

        # Score confirmations
        confirming = 0
        if buying_absorption:
            direction = "LONG"
            if cmf > 0:
                confirming += 1
            if obv_divergence > 0:
                confirming += 1
        else:
            direction = "SHORT"
            if cmf < 0:
                confirming += 1
            if obv_divergence < 0:
                confirming += 1

        strength = min(0.4 + confirming * 0.3, 1.0)
        return direction, strength

    def generate_signal(self, ctx: MarketContext) -> Signal:
        """Generate signal from MarketContext (live trading path)."""
        # In live mode, use orderbook_imbalance directly
        book_score = ctx.orderbook_imbalance
        direction = "NEUTRAL"
        strength = 0.0

        if abs(book_score) >= 0.3:
            direction = "LONG" if book_score > 0 else "SHORT"
            strength = min(abs(book_score), 1.0)

        return Signal(
            symbol=ctx.symbol,
            direction=direction,
            strength=strength,
            strategy_name=self.name,
            technical_score=book_score,
            indicator_snapshot={"confirming_count": 2 if strength > 0 else 0},
        )
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_orderflow_1h.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add bot/strategy/orderflow.py tests/test_orderflow_1h.py
git commit -m "feat: redesign orderflow strategy for 1h candle signals"
```

---

### Task 7: Funding Contrarian Strategy Redesign (1h signals)

**Files:**
- Modify: `bot/strategy/funding_contrarian.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_funding_1h.py
from bot.strategy.funding_contrarian import FundingContrarianStrategy


class TestFunding1h:
    def test_overbought_short(self):
        """1h RSI > 72 + MACD fading + 4h RSI > 60 = SHORT."""
        strat = FundingContrarianStrategy()
        direction, strength = strat.evaluate_1h(
            rsi_1h=75.0, rsi_4h=65.0,
            macd_hist=0.5, macd_hist_prev=1.2,  # fading
        )
        assert direction == "SHORT"
        assert strength > 0.0

    def test_oversold_long(self):
        """1h RSI < 28 + MACD recovering + 4h RSI < 40 = LONG."""
        strat = FundingContrarianStrategy()
        direction, strength = strat.evaluate_1h(
            rsi_1h=22.0, rsi_4h=35.0,
            macd_hist=-0.5, macd_hist_prev=-1.2,  # recovering
        )
        assert direction == "LONG"
        assert strength > 0.0

    def test_neutral_when_not_extreme(self):
        """RSI in normal range = NEUTRAL."""
        strat = FundingContrarianStrategy()
        direction, _ = strat.evaluate_1h(
            rsi_1h=50.0, rsi_4h=50.0,
            macd_hist=0.1, macd_hist_prev=0.1,
        )
        assert direction == "NEUTRAL"

    def test_no_signal_without_4h_confirmation(self):
        """1h RSI extreme but 4h RSI not confirming = NEUTRAL."""
        strat = FundingContrarianStrategy()
        direction, _ = strat.evaluate_1h(
            rsi_1h=75.0, rsi_4h=45.0,  # 4h not overbought
            macd_hist=0.5, macd_hist_prev=1.2,
        )
        assert direction == "NEUTRAL"

    def test_no_signal_without_macd_fading(self):
        """RSI extreme but MACD still strengthening = NEUTRAL."""
        strat = FundingContrarianStrategy()
        direction, _ = strat.evaluate_1h(
            rsi_1h=75.0, rsi_4h=65.0,
            macd_hist=1.5, macd_hist_prev=1.0,  # strengthening, not fading
        )
        assert direction == "NEUTRAL"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_funding_1h.py -v`
Expected: FAIL

- [ ] **Step 3: Rewrite bot/strategy/funding_contrarian.py**

```python
"""Funding Rate Contrarian strategy — redesigned for 1h candles."""
from __future__ import annotations

from bot.strategy.base import BaseStrategy, MarketContext, Signal


class FundingContrarianStrategy(BaseStrategy):
    name = "funding_contrarian"

    RSI_OVERBOUGHT = 72.0
    RSI_OVERSOLD = 28.0
    RSI_4H_OVERBOUGHT = 60.0
    RSI_4H_OVERSOLD = 40.0

    def evaluate_1h(
        self,
        rsi_1h: float,
        rsi_4h: float,
        macd_hist: float,
        macd_hist_prev: float,
    ) -> tuple[str, float]:
        """Evaluate funding contrarian signal on 1h data (backtest proxy).

        Requires: RSI extreme on 1h + 4h confirmation + MACD momentum fading.
        """
        # Check for overbought (SHORT signal)
        if rsi_1h > self.RSI_OVERBOUGHT:
            if rsi_4h <= self.RSI_4H_OVERBOUGHT:
                return "NEUTRAL", 0.0  # 4h not confirming
            if macd_hist >= macd_hist_prev:
                return "NEUTRAL", 0.0  # MACD still strengthening
            # All conditions met: SHORT
            strength = min((rsi_1h - self.RSI_OVERBOUGHT) / 20.0 + 0.5, 1.0)
            return "SHORT", strength

        # Check for oversold (LONG signal)
        if rsi_1h < self.RSI_OVERSOLD:
            if rsi_4h >= self.RSI_4H_OVERSOLD:
                return "NEUTRAL", 0.0  # 4h not confirming
            if macd_hist <= macd_hist_prev:
                return "NEUTRAL", 0.0  # MACD still weakening
            # All conditions met: LONG
            strength = min((self.RSI_OVERSOLD - rsi_1h) / 20.0 + 0.5, 1.0)
            return "LONG", strength

        return "NEUTRAL", 0.0

    def generate_signal(self, ctx: MarketContext) -> Signal:
        """Generate signal from MarketContext (live trading path)."""
        onchain = ctx.onchain_score
        direction = "NEUTRAL"
        strength = 0.0

        if onchain > 0.3:
            direction = "LONG"
            strength = min(abs(onchain), 1.0)
        elif onchain < -0.3:
            direction = "SHORT"
            strength = min(abs(onchain), 1.0)

        return Signal(
            symbol=ctx.symbol,
            direction=direction,
            strength=strength,
            strategy_name=self.name,
            technical_score=onchain,
            indicator_snapshot={"confirming_count": 2 if strength > 0 else 0},
        )
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_funding_1h.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add bot/strategy/funding_contrarian.py tests/test_funding_1h.py
git commit -m "feat: redesign funding contrarian for 1h signals with loosened thresholds"
```

---

### Task 8: Range Strategy Redesign (1h signals)

**Files:**
- Modify: `bot/strategy/range_trading.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_range_1h.py
from bot.strategy.range_trading import RangeStrategy


class TestRange1h:
    def test_long_at_lower_band(self):
        """Price near lower BB + RSI < 35 + ADX < 22 + BB in range = LONG."""
        strat = RangeStrategy()
        direction, strength = strat.evaluate_1h(
            price=98.5, bb_lower=98.0, bb_upper=102.0, bb_mid=100.0,
            bb_bandwidth=0.04, adx_4h=18.0, rsi_1h=30.0,
        )
        assert direction == "LONG"
        assert strength > 0.0

    def test_short_at_upper_band(self):
        """Price near upper BB + RSI > 65 + range confirmed = SHORT."""
        strat = RangeStrategy()
        direction, strength = strat.evaluate_1h(
            price=101.8, bb_lower=98.0, bb_upper=102.0, bb_mid=100.0,
            bb_bandwidth=0.04, adx_4h=18.0, rsi_1h=70.0,
        )
        assert direction == "SHORT"
        assert strength > 0.0

    def test_neutral_when_trending(self):
        """ADX > 22 on 4h = not a range, NEUTRAL."""
        strat = RangeStrategy()
        direction, _ = strat.evaluate_1h(
            price=98.5, bb_lower=98.0, bb_upper=102.0, bb_mid=100.0,
            bb_bandwidth=0.04, adx_4h=28.0, rsi_1h=30.0,
        )
        assert direction == "NEUTRAL"

    def test_neutral_when_bandwidth_too_wide(self):
        """BB bandwidth > 8% = not a tight range."""
        strat = RangeStrategy()
        direction, _ = strat.evaluate_1h(
            price=98.5, bb_lower=90.0, bb_upper=110.0, bb_mid=100.0,
            bb_bandwidth=0.10, adx_4h=18.0, rsi_1h=30.0,
        )
        assert direction == "NEUTRAL"

    def test_neutral_when_bandwidth_too_narrow(self):
        """BB bandwidth < 1.5% = potential squeeze territory, not range."""
        strat = RangeStrategy()
        direction, _ = strat.evaluate_1h(
            price=99.8, bb_lower=99.5, bb_upper=100.5, bb_mid=100.0,
            bb_bandwidth=0.01, adx_4h=18.0, rsi_1h=30.0,
        )
        assert direction == "NEUTRAL"

    def test_neutral_when_price_not_near_band(self):
        """Price not within 0.5% of band = no entry."""
        strat = RangeStrategy()
        direction, _ = strat.evaluate_1h(
            price=99.0, bb_lower=97.0, bb_upper=103.0, bb_mid=100.0,
            bb_bandwidth=0.06, adx_4h=18.0, rsi_1h=30.0,
        )
        assert direction == "NEUTRAL"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_range_1h.py -v`
Expected: FAIL

- [ ] **Step 3: Rewrite bot/strategy/range_trading.py**

```python
"""Range Mean-Reversion strategy — redesigned for 1h candles."""
from __future__ import annotations

from bot.strategy.base import BaseStrategy, MarketContext, Signal


class RangeStrategy(BaseStrategy):
    name = "range"

    # 1h thresholds
    MIN_BANDWIDTH = 0.015   # 1.5%
    MAX_BANDWIDTH = 0.08    # 8%
    MAX_ADX_4H = 22.0
    BAND_PROXIMITY_PCT = 0.5  # within 0.5% of band
    RSI_OVERSOLD = 35.0
    RSI_OVERBOUGHT = 65.0

    def evaluate_1h(
        self,
        price: float,
        bb_lower: float,
        bb_upper: float,
        bb_mid: float,
        bb_bandwidth: float,
        adx_4h: float,
        rsi_1h: float,
    ) -> tuple[str, float]:
        """Evaluate range mean-reversion on 1h data.

        Requires: BB bandwidth in range, ADX < 22 on 4h, price near band, RSI extreme.
        """
        # Range confirmation gates
        if bb_bandwidth < self.MIN_BANDWIDTH or bb_bandwidth > self.MAX_BANDWIDTH:
            return "NEUTRAL", 0.0
        if adx_4h > self.MAX_ADX_4H:
            return "NEUTRAL", 0.0

        # Check proximity to bands
        band_range = bb_upper - bb_lower
        if band_range <= 0:
            return "NEUTRAL", 0.0

        proximity_threshold = price * (self.BAND_PROXIMITY_PCT / 100.0)

        near_lower = (price - bb_lower) <= proximity_threshold
        near_upper = (bb_upper - price) <= proximity_threshold

        # LONG at lower band
        if near_lower and rsi_1h < self.RSI_OVERSOLD:
            strength = min(0.5 + (self.RSI_OVERSOLD - rsi_1h) / 30.0, 1.0)
            return "LONG", strength

        # SHORT at upper band
        if near_upper and rsi_1h > self.RSI_OVERBOUGHT:
            strength = min(0.5 + (rsi_1h - self.RSI_OVERBOUGHT) / 30.0, 1.0)
            return "SHORT", strength

        return "NEUTRAL", 0.0

    def generate_signal(self, ctx: MarketContext) -> Signal:
        """Generate signal from MarketContext (live trading path)."""
        return Signal(
            symbol=ctx.symbol,
            direction="NEUTRAL",
            strength=0.0,
            strategy_name=self.name,
            technical_score=0.0,
        )
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_range_1h.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add bot/strategy/range_trading.py tests/test_range_1h.py
git commit -m "feat: redesign range strategy for 1h BB bands with wider thresholds"
```

---

### Task 9: Volatility Squeeze Strategy (new)

**Files:**
- Modify: `bot/strategy/squeeze.py` (replace stub from Task 5)
- Create: `tests/test_squeeze_strategy.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_squeeze_strategy.py
from bot.strategy.squeeze import SqueezeStrategy


class TestSqueeze:
    def test_long_breakout(self):
        """BB expands from <2% to >3%, price above upper BB, volume surge = LONG."""
        strat = SqueezeStrategy()
        direction, strength = strat.evaluate_1h(
            bb_bandwidth_prev=0.018,  # was compressed
            bb_bandwidth=0.035,       # now expanded
            price=102.0,
            bb_upper=101.5,
            bb_lower=98.5,
            volume_surge=1.8,
            ema50_slope=0.1,  # positive = uptrend on 4h
        )
        assert direction == "LONG"
        assert strength > 0.0

    def test_short_breakout(self):
        """Squeeze expansion, price below lower BB = SHORT."""
        strat = SqueezeStrategy()
        direction, strength = strat.evaluate_1h(
            bb_bandwidth_prev=0.015,
            bb_bandwidth=0.04,
            price=97.0,
            bb_upper=101.0,
            bb_lower=99.0,
            volume_surge=2.0,
            ema50_slope=-0.1,  # negative = downtrend
        )
        assert direction == "SHORT"
        assert strength > 0.0

    def test_no_squeeze_without_compression(self):
        """Previous bandwidth not compressed = no squeeze."""
        strat = SqueezeStrategy()
        direction, _ = strat.evaluate_1h(
            bb_bandwidth_prev=0.03,  # was already wide
            bb_bandwidth=0.035,
            price=102.0, bb_upper=101.5, bb_lower=98.5,
            volume_surge=1.8, ema50_slope=0.1,
        )
        assert direction == "NEUTRAL"

    def test_no_squeeze_without_expansion(self):
        """Bandwidth still compressed = no expansion yet."""
        strat = SqueezeStrategy()
        direction, _ = strat.evaluate_1h(
            bb_bandwidth_prev=0.015,
            bb_bandwidth=0.025,  # still below 3%
            price=102.0, bb_upper=101.5, bb_lower=98.5,
            volume_surge=1.8, ema50_slope=0.1,
        )
        assert direction == "NEUTRAL"

    def test_no_squeeze_without_volume(self):
        """Squeeze without volume confirmation = no signal."""
        strat = SqueezeStrategy()
        direction, _ = strat.evaluate_1h(
            bb_bandwidth_prev=0.015,
            bb_bandwidth=0.04,
            price=102.0, bb_upper=101.5, bb_lower=98.5,
            volume_surge=1.2,  # below 1.5
            ema50_slope=0.1,
        )
        assert direction == "NEUTRAL"

    def test_no_squeeze_against_trend(self):
        """Price breaks up but 4h EMA slope is down = no signal."""
        strat = SqueezeStrategy()
        direction, _ = strat.evaluate_1h(
            bb_bandwidth_prev=0.015,
            bb_bandwidth=0.04,
            price=102.0, bb_upper=101.5, bb_lower=98.5,
            volume_surge=2.0,
            ema50_slope=-0.1,  # against breakout direction
        )
        assert direction == "NEUTRAL"

    def test_tp_is_2x_squeeze_width(self):
        """TP target = 2x the BB width at squeeze point."""
        strat = SqueezeStrategy()
        _, strength = strat.evaluate_1h(
            bb_bandwidth_prev=0.018,
            bb_bandwidth=0.04,
            price=102.0, bb_upper=101.5, bb_lower=98.5,
            volume_surge=2.0, ema50_slope=0.1,
        )
        # squeeze_width = bb_upper - bb_lower = 3.0
        # tp_distance = 2 * 3.0 = 6.0
        assert strat.last_tp_distance == 6.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_squeeze_strategy.py -v`
Expected: FAIL

- [ ] **Step 3: Implement squeeze strategy**

```python
# bot/strategy/squeeze.py
"""Volatility Squeeze strategy — BB compression followed by expansion."""
from __future__ import annotations

from bot.strategy.base import BaseStrategy, MarketContext, Signal


class SqueezeStrategy(BaseStrategy):
    name = "squeeze"

    COMPRESSION_THRESHOLD = 0.02   # BB bandwidth < 2% = compressed
    EXPANSION_THRESHOLD = 0.03     # BB bandwidth > 3% = expanded
    MIN_VOLUME_SURGE = 1.5

    def __init__(self):
        self.last_tp_distance: float = 0.0

    def evaluate_1h(
        self,
        bb_bandwidth_prev: float,
        bb_bandwidth: float,
        price: float,
        bb_upper: float,
        bb_lower: float,
        volume_surge: float,
        ema50_slope: float,
    ) -> tuple[str, float]:
        """Evaluate volatility squeeze on 1h data.

        Requires: previous candle bandwidth < 2%, current > 3%,
        price breaks a band, volume surge, 4h trend agreement.
        """
        self.last_tp_distance = 0.0

        # Check squeeze: compression → expansion
        if bb_bandwidth_prev >= self.COMPRESSION_THRESHOLD:
            return "NEUTRAL", 0.0  # wasn't compressed
        if bb_bandwidth < self.EXPANSION_THRESHOLD:
            return "NEUTRAL", 0.0  # hasn't expanded yet

        # Volume confirmation
        if volume_surge < self.MIN_VOLUME_SURGE:
            return "NEUTRAL", 0.0

        # Direction: which band did price break?
        squeeze_width = bb_upper - bb_lower
        if squeeze_width <= 0:
            return "NEUTRAL", 0.0

        if price > bb_upper:
            direction = "LONG"
            if ema50_slope <= 0:
                return "NEUTRAL", 0.0  # against 4h trend
        elif price < bb_lower:
            direction = "SHORT"
            if ema50_slope >= 0:
                return "NEUTRAL", 0.0  # against 4h trend
        else:
            return "NEUTRAL", 0.0  # price inside bands

        # TP = 2x the squeeze width (measured move)
        self.last_tp_distance = squeeze_width * 2.0

        strength = min(0.6 + volume_surge * 0.1, 1.0)
        return direction, strength

    def generate_signal(self, ctx: MarketContext) -> Signal:
        """Generate signal from MarketContext (live trading path)."""
        return Signal(
            symbol=ctx.symbol,
            direction="NEUTRAL",
            strength=0.0,
            strategy_name=self.name,
            technical_score=0.0,
        )
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_squeeze_strategy.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add bot/strategy/squeeze.py tests/test_squeeze_strategy.py
git commit -m "feat: add volatility squeeze strategy (BB compression -> expansion)"
```

---

### Task 10: Confluence Gate Meta-Strategy

**Files:**
- Create: `bot/strategy/confluence.py`
- Create: `tests/test_confluence.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_confluence.py
from bot.strategy.confluence import check_confluence, ConfluenceResult


class TestConfluence:
    def test_two_agree_triggers_confluence(self):
        """Two strategies agreeing on direction = confluence."""
        signals = [
            {"strategy": "orderflow", "direction": "LONG", "strength": 0.7},
            {"strategy": "funding_contrarian", "direction": "LONG", "strength": 0.6},
        ]
        result = check_confluence(signals)
        assert result.triggered is True
        assert result.direction == "LONG"
        assert result.agreeing_strategies == ["orderflow", "funding_contrarian"]

    def test_disagreement_no_confluence(self):
        """Strategies disagree = no confluence."""
        signals = [
            {"strategy": "orderflow", "direction": "LONG", "strength": 0.7},
            {"strategy": "funding_contrarian", "direction": "SHORT", "strength": 0.6},
        ]
        result = check_confluence(signals)
        assert result.triggered is False

    def test_single_signal_no_confluence(self):
        """Only one signal = no confluence possible."""
        signals = [
            {"strategy": "orderflow", "direction": "LONG", "strength": 0.7},
        ]
        result = check_confluence(signals)
        assert result.triggered is False

    def test_neutral_signals_ignored(self):
        """NEUTRAL signals don't count toward confluence."""
        signals = [
            {"strategy": "orderflow", "direction": "LONG", "strength": 0.7},
            {"strategy": "range", "direction": "NEUTRAL", "strength": 0.0},
            {"strategy": "funding_contrarian", "direction": "NEUTRAL", "strength": 0.0},
        ]
        result = check_confluence(signals)
        assert result.triggered is False

    def test_best_strength_used(self):
        """Confluence strength = max of agreeing signals."""
        signals = [
            {"strategy": "orderflow", "direction": "SHORT", "strength": 0.5},
            {"strategy": "squeeze", "direction": "SHORT", "strength": 0.9},
        ]
        result = check_confluence(signals)
        assert result.triggered is True
        assert result.strength == 0.9

    def test_empty_signals(self):
        result = check_confluence([])
        assert result.triggered is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_confluence.py -v`
Expected: FAIL

- [ ] **Step 3: Implement confluence gate**

```python
# bot/strategy/confluence.py
"""Confluence Gate: detects when 2+ strategies agree on direction."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ConfluenceResult:
    triggered: bool = False
    direction: str = "NEUTRAL"
    strength: float = 0.0
    agreeing_strategies: List[str] = field(default_factory=list)


def check_confluence(signals: List[Dict]) -> ConfluenceResult:
    """Check if 2+ strategy signals agree on direction.

    Args:
        signals: List of dicts with keys: strategy, direction, strength.

    Returns:
        ConfluenceResult with triggered=True if 2+ agree.
    """
    if not signals:
        return ConfluenceResult()

    # Filter out NEUTRAL signals
    directional = [s for s in signals if s.get("direction") not in ("NEUTRAL", None)]
    if len(directional) < 2:
        return ConfluenceResult()

    # Count directions
    direction_counts = Counter(s["direction"] for s in directional)
    most_common_dir, count = direction_counts.most_common(1)[0]

    if count < 2:
        return ConfluenceResult()

    agreeing = [s for s in directional if s["direction"] == most_common_dir]
    best_strength = max(s.get("strength", 0.0) for s in agreeing)

    return ConfluenceResult(
        triggered=True,
        direction=most_common_dir,
        strength=best_strength,
        agreeing_strategies=[s["strategy"] for s in agreeing],
    )
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_confluence.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add bot/strategy/confluence.py tests/test_confluence.py
git commit -m "feat: add confluence gate meta-strategy"
```

---

## Chunk 3: Execution & Engine

### Task 11: Limit Order Manager

**Files:**
- Create: `bot/execution/__init__.py`
- Create: `bot/execution/limit_order_manager.py`
- Create: `tests/test_limit_order_manager.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_limit_order_manager.py
import time
from unittest.mock import MagicMock, AsyncMock
from bot.execution.limit_order_manager import LimitOrderManager, PendingOrder


class TestLimitOrderManager:
    def test_create_pending_order(self):
        """Creating a pending order stores it correctly."""
        mgr = LimitOrderManager(client=MagicMock())
        order = mgr.create_pending(
            symbol="BTC-EUR",
            direction="LONG",
            limit_price=50000.0,
            size_eur=100.0,
            stop_loss=48000.0,
            take_profit=53000.0,
            strategy="orderflow",
        )
        assert order.symbol == "BTC-EUR"
        assert order.direction == "LONG"
        assert order.status == "pending"
        assert len(mgr.pending_orders) == 1

    def test_check_timeout_cancels(self):
        """Orders older than timeout_seconds should be cancelled."""
        mgr = LimitOrderManager(client=MagicMock(), timeout_seconds=0)
        order = mgr.create_pending(
            symbol="BTC-EUR", direction="LONG", limit_price=50000.0,
            size_eur=100.0, stop_loss=48000.0, take_profit=53000.0,
            strategy="orderflow",
        )
        # Simulate time passing
        order.created_at = time.time() - 1
        timed_out = mgr.check_timeouts()
        assert len(timed_out) == 1
        assert timed_out[0].status == "timed_out"
        assert len(mgr.pending_orders) == 0

    def test_price_moved_away_cancels(self):
        """If price moves >0.3% from limit, cancel."""
        mgr = LimitOrderManager(client=MagicMock(), max_price_deviation_pct=0.3)
        order = mgr.create_pending(
            symbol="BTC-EUR", direction="LONG", limit_price=50000.0,
            size_eur=100.0, stop_loss=48000.0, take_profit=53000.0,
            strategy="orderflow",
        )
        # Price moved 0.5% away
        cancelled = mgr.check_price_deviation(
            symbol="BTC-EUR", current_price=50250.0
        )
        assert len(cancelled) == 1

    def test_price_within_range_keeps_order(self):
        """If price within 0.3%, order stays."""
        mgr = LimitOrderManager(client=MagicMock(), max_price_deviation_pct=0.3)
        mgr.create_pending(
            symbol="BTC-EUR", direction="LONG", limit_price=50000.0,
            size_eur=100.0, stop_loss=48000.0, take_profit=53000.0,
            strategy="orderflow",
        )
        cancelled = mgr.check_price_deviation(
            symbol="BTC-EUR", current_price=50100.0  # 0.2%
        )
        assert len(cancelled) == 0
        assert len(mgr.pending_orders) == 1

    def test_mark_filled(self):
        """Marking an order as filled removes it from pending."""
        mgr = LimitOrderManager(client=MagicMock())
        order = mgr.create_pending(
            symbol="BTC-EUR", direction="LONG", limit_price=50000.0,
            size_eur=100.0, stop_loss=48000.0, take_profit=53000.0,
            strategy="orderflow",
        )
        mgr.mark_filled(order.order_id, fill_price=50001.0)
        assert order.status == "filled"
        assert len(mgr.pending_orders) == 0
        assert len(mgr.filled_orders) == 1

    def test_partial_fill(self):
        """Partial fill keeps order pending with reduced remaining size."""
        mgr = LimitOrderManager(client=MagicMock())
        order = mgr.create_pending(
            symbol="BTC-EUR", direction="LONG", limit_price=50000.0,
            size_eur=100.0, stop_loss=48000.0, take_profit=53000.0,
            strategy="orderflow",
        )
        result = mgr.mark_partial_fill(order.order_id, filled_eur=60.0, fill_price=50001.0)
        assert result is not None
        assert result.filled_eur == 60.0
        assert order.size_eur == 40.0  # remaining
        assert order.status == "pending"  # still pending for remainder
        assert len(mgr.filled_orders) == 1  # partial position opened
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_limit_order_manager.py -v`
Expected: FAIL

- [ ] **Step 3: Create bot/execution/__init__.py**

```python
# bot/execution/__init__.py
```

- [ ] **Step 4: Implement limit order manager**

```python
# bot/execution/limit_order_manager.py
"""Limit order lifecycle management: place, monitor, timeout, cancel."""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PendingOrder:
    order_id: str
    symbol: str
    direction: str
    limit_price: float
    size_eur: float
    stop_loss: float
    take_profit: float
    strategy: str
    status: str = "pending"  # pending, filled, timed_out, cancelled
    created_at: float = field(default_factory=time.time)
    exchange_order_id: Optional[str] = None
    fill_price: Optional[float] = None


class LimitOrderManager:
    """Manages the lifecycle of limit orders."""

    def __init__(
        self,
        client,
        timeout_seconds: int = 600,  # 10 min (2 x 5m candles)
        max_price_deviation_pct: float = 0.3,
    ):
        self._client = client
        self._timeout = timeout_seconds
        self._max_deviation = max_price_deviation_pct
        self.pending_orders: Dict[str, PendingOrder] = {}
        self.filled_orders: List[PendingOrder] = []

    def create_pending(
        self,
        symbol: str,
        direction: str,
        limit_price: float,
        size_eur: float,
        stop_loss: float,
        take_profit: float,
        strategy: str,
    ) -> PendingOrder:
        """Create and track a new pending limit order."""
        order = PendingOrder(
            order_id=str(uuid.uuid4())[:8],
            symbol=symbol,
            direction=direction,
            limit_price=limit_price,
            size_eur=size_eur,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy=strategy,
        )
        self.pending_orders[order.order_id] = order
        logger.info(
            "Pending order %s: %s %s @ %.2f (SL=%.2f, TP=%.2f)",
            order.order_id, direction, symbol, limit_price, stop_loss, take_profit,
        )
        return order

    def check_timeouts(self) -> List[PendingOrder]:
        """Cancel orders that exceeded the timeout."""
        now = time.time()
        timed_out = []
        for oid in list(self.pending_orders):
            order = self.pending_orders[oid]
            if now - order.created_at >= self._timeout:
                order.status = "timed_out"
                del self.pending_orders[oid]
                timed_out.append(order)
                logger.info("Order %s timed out for %s", oid, order.symbol)
        return timed_out

    def check_price_deviation(
        self, symbol: str, current_price: float
    ) -> List[PendingOrder]:
        """Cancel orders where price moved too far from the limit."""
        cancelled = []
        for oid in list(self.pending_orders):
            order = self.pending_orders[oid]
            if order.symbol != symbol:
                continue
            deviation_pct = abs(current_price - order.limit_price) / order.limit_price * 100
            if deviation_pct > self._max_deviation:
                order.status = "cancelled"
                del self.pending_orders[oid]
                cancelled.append(order)
                logger.info(
                    "Order %s cancelled: price moved %.2f%% from limit",
                    oid, deviation_pct,
                )
        return cancelled

    def mark_filled(self, order_id: str, fill_price: float) -> Optional[PendingOrder]:
        """Mark an order as filled."""
        if order_id not in self.pending_orders:
            return None
        order = self.pending_orders.pop(order_id)
        order.status = "filled"
        order.fill_price = fill_price
        self.filled_orders.append(order)
        logger.info("Order %s filled at %.2f for %s", order_id, fill_price, order.symbol)
        return order

    def mark_partial_fill(
        self, order_id: str, filled_eur: float, fill_price: float
    ) -> Optional[PendingOrder]:
        """Handle a partial fill: open position for filled portion, keep remainder pending."""
        if order_id not in self.pending_orders:
            return None
        order = self.pending_orders[order_id]
        if filled_eur >= order.size_eur:
            return self.mark_filled(order_id, fill_price)

        # Create a filled record for the partial
        partial = PendingOrder(
            order_id=f"{order_id}-p",
            symbol=order.symbol,
            direction=order.direction,
            limit_price=order.limit_price,
            size_eur=order.size_eur,  # original size for reference
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            strategy=order.strategy,
            status="filled",
            fill_price=fill_price,
        )
        partial.filled_eur = filled_eur
        self.filled_orders.append(partial)

        # Reduce remaining size on the pending order
        order.size_eur -= filled_eur
        logger.info(
            "Order %s partially filled: %.2f EUR @ %.2f, %.2f EUR remaining",
            order_id, filled_eur, fill_price, order.size_eur,
        )
        return partial

    def cancel_all(self, symbol: Optional[str] = None) -> List[PendingOrder]:
        """Cancel all pending orders, optionally filtered by symbol."""
        cancelled = []
        for oid in list(self.pending_orders):
            order = self.pending_orders[oid]
            if symbol and order.symbol != symbol:
                continue
            order.status = "cancelled"
            del self.pending_orders[oid]
            cancelled.append(order)
        return cancelled

    def get_pending_for_symbol(self, symbol: str) -> List[PendingOrder]:
        """Get all pending orders for a symbol."""
        return [o for o in self.pending_orders.values() if o.symbol == symbol]
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_limit_order_manager.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add bot/execution/__init__.py bot/execution/limit_order_manager.py tests/test_limit_order_manager.py
git commit -m "feat: add limit order manager for maker-first order execution"
```

---

### Task 12: Risk Engine Updates (Correlation Guard, Fee Gate Integration)

**Files:**
- Modify: `bot/risk/engine.py`
- Modify: `tests/test_risk_engine.py`
- Modify: `bot/risk/drawdown_guard.py`
- Modify: `tests/test_drawdown_guard.py`

- [ ] **Step 1: Write failing tests for correlation guard**

Add to `tests/test_risk_engine.py`:

```python
class TestCorrelationGuard:
    def test_reduces_size_at_3_correlated_positions(self):
        """3+ same-direction positions on correlated pairs → 50% size reduction."""
        from bot.risk.engine import check_correlation_guard
        open_positions = [
            {"symbol": "BTC-EUR", "direction": "LONG"},
            {"symbol": "ETH-EUR", "direction": "LONG"},
            {"symbol": "SOL-EUR", "direction": "LONG"},
        ]
        multiplier = check_correlation_guard(
            new_direction="LONG",
            open_positions=open_positions,
        )
        assert multiplier == 0.5

    def test_no_reduction_below_3(self):
        """Fewer than 3 same-direction positions → no reduction."""
        from bot.risk.engine import check_correlation_guard
        open_positions = [
            {"symbol": "BTC-EUR", "direction": "LONG"},
            {"symbol": "ETH-EUR", "direction": "LONG"},
        ]
        multiplier = check_correlation_guard(
            new_direction="LONG",
            open_positions=open_positions,
        )
        assert multiplier == 1.0

    def test_opposite_direction_no_reduction(self):
        """Positions in opposite direction don't count."""
        from bot.risk.engine import check_correlation_guard
        open_positions = [
            {"symbol": "BTC-EUR", "direction": "SHORT"},
            {"symbol": "ETH-EUR", "direction": "SHORT"},
            {"symbol": "SOL-EUR", "direction": "SHORT"},
        ]
        multiplier = check_correlation_guard(
            new_direction="LONG",
            open_positions=open_positions,
        )
        assert multiplier == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_risk_engine.py::TestCorrelationGuard -v`
Expected: FAIL

- [ ] **Step 3: Add correlation guard to bot/risk/engine.py**

```python
def check_correlation_guard(
    new_direction: str,
    open_positions: list,
    threshold: int = 3,
) -> float:
    """Returns position size multiplier based on directional concentration.

    If 3+ open positions are in the same direction as the new trade,
    reduce new position size by 50%.
    """
    same_direction = sum(
        1 for p in open_positions
        if p.get("direction") == new_direction
    )
    if same_direction >= threshold:
        return 0.5
    return 1.0
```

- [ ] **Step 4: Update drawdown_guard.py — remove hard halt, update daily loss default**

In `bot/risk/drawdown_guard.py`, remove the hard halt logic. Keep the daily loss circuit breaker but change default from 200 to 50. Keep soft drawdown tracking for informational purposes but don't block trades based on it.

- [ ] **Step 5: Run all risk tests**

Run: `python -m pytest tests/test_risk_engine.py tests/test_drawdown_guard.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add bot/risk/engine.py bot/risk/drawdown_guard.py tests/test_risk_engine.py tests/test_drawdown_guard.py
git commit -m "feat: add correlation guard, remove drawdown hard halt, update daily loss to EUR 50"
```

---

### Task 13: Backtest Engine Overhaul

**Files:**
- Modify: `bot/backtest/engine.py`

This is the largest single task. The engine needs:
1. SIGNAL_EVERY = 12 (1h signals)
2. Maker fee model
3. Fill rate model (price-based)
4. Remove trend_following and breakout evaluation
5. Add 1h strategy evaluation (orderflow, funding, range, squeeze)
6. New metrics (profit_per_fee, regime_breakdown)
7. Use fixed fractional sizing instead of Kelly
8. Pass 4h data to detect_regime
9. Integrate fee-aware gate
10. Integrate confluence gate

- [ ] **Step 1: Read the current engine.py to understand all integration points**

Read: `bot/backtest/engine.py` fully — note all references to:
- `kelly_size` (replace with `fixed_fractional_size`)
- `SIGNAL_EVERY` (change from 4 to 12)
- Strategy evaluation in `_evaluate_precomputed` (rewrite for new strategies)
- Fee calculation in `_close_position` (add maker fee path)
- Any `trend_following` or `breakout` references (remove)

- [ ] **Step 2: Update SIGNAL_EVERY, fee model constants, and range max hold**

```python
SIGNAL_EVERY = 12  # every 1h (12 x 5m candles)

# Fee model
MAKER_FEE_PCT = 0.0015   # 0.15%
TAKER_FEE_PCT = 0.0025   # 0.25%
```

Also update the `BacktestEngine.__init__` default for `max_open_positions` from 3 to 10 (matching config.py):
```python
# BEFORE:
def __init__(self, candles, initial_capital=10000.0, max_open_positions=3, ...):
# AFTER:
def __init__(self, candles, initial_capital=10000.0, max_open_positions=10, ...):
```

Also update `_range_max_hold_bars` from 288 to 864 (72h at 5m candles):
```python
# BEFORE:
_range_max_hold_bars = 288   # 24h
# AFTER:
_range_max_hold_bars = 864   # 72h (spec requires 72h max hold for range strategy)
```

- [ ] **Step 3: Replace Kelly import with fixed fractional**

```python
# Remove:
from bot.risk.position_sizer import kelly_size
# Add:
from bot.risk.position_sizer import fixed_fractional_size
from bot.risk.fee_gate import check_fee_gate
from bot.strategy.confluence import check_confluence
```

- [ ] **Step 4: Update _precompute_signals to add 4h indicators and squeeze detection**

Add to the pre-computation:
- 4h ADX and ATR arrays
- 1h BB bandwidth array
- Squeeze detection boolean array (bandwidth crosses from <2% to >3%)
- Remove trend_following and breakout specific computations

- [ ] **Step 5: Rewrite _evaluate_precomputed for new strategy roster**

Replace the current strategy evaluation with calls to:
- `orderflow.evaluate_1h(...)` using pre-computed candle structure
- `funding_contrarian.evaluate_1h(...)` using pre-computed RSI/MACD
- `range.evaluate_1h(...)` using pre-computed BB/ADX
- `squeeze.evaluate_1h(...)` using pre-computed BB bandwidth

Each call uses the pre-computed numpy arrays indexed by the current bar.

- [ ] **Step 6: Add fill rate model**

After signal evaluation, before opening position:
```python
# Deterministic fill rate model
limit_price = close  # approximate
if direction == "LONG":
    # Check if next candle's low reaches our limit
    if i + 1 < n and lows[i + 1] > limit_price:
        continue  # skip — limit not filled
else:
    if i + 1 < n and highs[i + 1] < limit_price:
        continue  # skip — limit not filled
```

- [ ] **Step 7: Update _open_position for fixed fractional sizing**

Replace Kelly sizing call with:
```python
size = fixed_fractional_size(
    equity=self._equity(current_price),
    base_risk_pct=params.get("base_risk_pct", 3.0),
    stop_distance_pct=stop_distance_pct,
    atr_pct=current_atr_pct,
    median_atr_pct=median_atr_pct,
)
```

- [ ] **Step 8: Update _close_position for maker/taker fee differentiation**

```python
if reason == "take_profit":
    exit_fee = slipped_exit * quantity * MAKER_FEE_PCT
else:
    exit_fee = slipped_exit * quantity * TAKER_FEE_PCT
```

- [ ] **Step 9: Add new metrics to BacktestResult**

```python
@dataclass
class BacktestResult:
    # ... existing fields ...
    profit_factor: float = 0.0       # gross_profit / gross_loss (used by walk-forward scoring)
    total_fees_paid: float = 0.0
    profit_per_fee: float = 0.0
    signals_generated: int = 0
    signals_filled: int = 0
    fill_rate: float = 0.0           # signals_filled / signals_generated
    quiet_hours_skipped: int = 0
    regime_pnl: Dict[str, float] = field(default_factory=dict)
```

Compute `profit_factor` and `fill_rate` in the results aggregation:
```python
gross_profit = sum(t.pnl_eur for t in trade_log if t.pnl_eur > 0)
gross_loss = abs(sum(t.pnl_eur for t in trade_log if t.pnl_eur < 0))
result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.0
result.fill_rate = result.signals_filled / result.signals_generated if result.signals_generated > 0 else 0.0
```

- [ ] **Step 10: Update trailing stop calls with activation_threshold**

Pass `activation_threshold=1.5` and `entry_price=pos.entry_price` to all `trail_stop()` calls.

- [ ] **Step 10a: Remove position_size_modifier() call from engine**

The engine currently calls `router.position_size_modifier(...)` — the new router no longer has this method. Find and remove all calls to `position_size_modifier()` in `engine.py`. The new sizing logic uses `fixed_fractional_size()` directly.

- [ ] **Step 10b: Remove update_hybrid_params() call from engine**

The engine `__init__` calls `router.update_hybrid_params(...)` — the new router no longer has this method (hybrid strategy is removed). Find and remove all references to `update_hybrid_params()` in `engine.py`.

- [ ] **Step 11: Write and run backtest engine tests**

Add tests to verify the new engine behavior. Create or update `tests/test_backtest_engine_overhaul.py`:

```python
# tests/test_backtest_engine_overhaul.py
"""Tests for the overhauled backtest engine (1h signals, maker fees, fill rate)."""

class TestSignalEvery:
    def test_signal_every_12(self):
        """Signals should be evaluated every 12 bars (1h)."""
        from bot.backtest.engine import SIGNAL_EVERY
        assert SIGNAL_EVERY == 12

class TestMakerFees:
    def test_tp_exit_uses_maker_fee(self):
        """Take-profit exits should use maker fee (0.15%), not taker."""
        # Run a minimal backtest, check that TP exits apply MAKER_FEE_PCT
        pass  # Implement with actual engine run

    def test_sl_exit_uses_taker_fee(self):
        """Stop-loss exits should use taker fee (0.25%)."""
        pass  # Implement with actual engine run

class TestFillRate:
    def test_fill_rate_computed(self):
        """BacktestResult should include fill_rate metric."""
        # Run minimal backtest, verify signals_generated, signals_filled, fill_rate
        pass  # Implement with actual engine run

class TestRangeMaxHold:
    def test_range_max_hold_72h(self):
        """Range strategy max hold should be 864 bars (72h)."""
        from bot.backtest.engine import BacktestEngine
        # Verify the _range_max_hold_bars attribute
        pass  # Implement by checking engine attribute
```

Run: `python -m pytest tests/test_backtest_*.py -v`
Fix any failures caused by the refactoring.

- [ ] **Step 12: Commit**

```bash
git add bot/backtest/engine.py
git commit -m "feat: overhaul backtest engine for 1h signals, maker fees, fill rate model"
```

---

## Chunk 4: Integration Layer

### Task 14: CandleCache 1h Aggregation

**Files:**
- Modify: `bot/data_loader.py`

- [ ] **Step 1: Read current data_loader.py to understand CandleCache**

Read: `bot/data_loader.py` — find the CandleCache class and understand how 5m candles are stored and how higher timeframes are currently derived.

- [ ] **Step 2: Add build_1h_candle method**

```python
def build_1h_candle(self, hour_start: datetime) -> Optional[CandleData]:
    """Aggregate the last 12 x 5m candles into a 1h candle.

    Args:
        hour_start: The start of the hour to build (e.g., 14:00:00).

    Returns:
        Aggregated 1h CandleData, or None if insufficient data.
    """
    hour_end = hour_start + timedelta(hours=1)
    candles_in_hour = [
        c for c in self._candles_5m
        if hour_start <= c.timestamp < hour_end
    ]
    if len(candles_in_hour) < 10:  # allow 2 missing
        return None

    return CandleData(
        symbol=candles_in_hour[0].symbol,
        interval="1h",
        timestamp=hour_start,
        open=candles_in_hour[0].open,
        high=max(c.high for c in candles_in_hour),
        low=min(c.low for c in candles_in_hour),
        close=candles_in_hour[-1].close,
        volume=sum(c.volume for c in candles_in_hour),
    )
```

- [ ] **Step 3: Commit**

```bash
git add bot/data_loader.py
git commit -m "feat: add 1h candle aggregation to CandleCache"
```

---

### Task 15: Trading Loop Changes

**Files:**
- Modify: `bot/trading_loop.py`

- [ ] **Step 1: Read current trading_loop.py**

Understand the current flow: WebSocket handlers, stop monitoring, signal evaluation cycle.

- [ ] **Step 2: Change stop-check trigger from "1m" to "5m"**

**CRITICAL SAFETY FIX:** In `on_candle()`, change:
```python
# BEFORE:
if data["interval"] == "1m":
    self._check_stops(...)
# AFTER:
if data["interval"] == "5m":
    self._check_stops(...)
```

- [ ] **Step 3: Add 1h candle detection with hour-crossing logic**

```python
# Add to __init__:
self._last_evaluated_hour = None

# In on_candle(), after the 5m stop check:
if data["interval"] == "5m":
    current_hour = candle.timestamp.replace(minute=0, second=0, microsecond=0)
    if current_hour != self._last_evaluated_hour:
        h1_candle = self._cache.build_1h_candle(
            current_hour - timedelta(hours=1)
        )
        if h1_candle is not None:
            await self._evaluate_1h_signals(h1_candle)
        self._last_evaluated_hour = current_hour
```

- [ ] **Step 4: Implement _evaluate_1h_signals**

New method that:
1. Detects regime using 4h data
2. Gets applicable strategies from router
3. Evaluates each strategy's `evaluate_1h()` method
4. Runs fee-aware gate on each signal
5. Checks confluence
6. Places limit orders via LimitOrderManager

- [ ] **Step 5: Replace kelly_size calls with fixed_fractional_size**

Find all `kelly_size(...)` calls in trading_loop.py and replace with `fixed_fractional_size(...)`.

- [ ] **Step 6: Update _check_stops for new trailing stop activation**

Pass `activation_threshold=1.5` and `entry_price` to `trail_stop()` calls.

- [ ] **Step 7: Add limit order management to 5m cycle**

In the 5m candle handler, after stop checks:
```python
# Check limit order timeouts
self._limit_mgr.check_timeouts()
# Check price deviations
self._limit_mgr.check_price_deviation(symbol, current_price)
```

- [ ] **Step 8: Test manually in paper mode**

Run the bot in paper mode briefly, verify:
- 5m stop checks fire correctly
- 1h signal evaluation triggers at hour boundaries
- No crashes or import errors

- [ ] **Step 9: Commit**

```bash
git add bot/trading_loop.py
git commit -m "feat: 1h signal evaluation, 5m stop monitoring, limit order integration"
```

---

### Task 16: Walk-Forward Optimization Changes

**Files:**
- Modify: `bot/learning/walk_forward.py`

- [ ] **Step 1: Read current walk_forward.py**

Understand the current Optuna setup, parameter space, window configuration.

- [ ] **Step 2: Update window constants**

```python
TRAIN_DAYS = 180
TEST_DAYS = 30
STEP_DAYS = 30
MIN_WINDOWS = 4
OPTUNA_TRIALS = 40  # keep same
```

- [ ] **Step 3: Update parameter search space**

Replace the current search space with:
```python
def _suggest_params(trial):
    return {
        "atr_multiplier": trial.suggest_float("atr_multiplier", 2.5, 5.0, step=0.5),
        "rr_ratio": trial.suggest_float("rr_ratio", 2.0, 4.0, step=0.5),
        "base_risk_pct": trial.suggest_float("base_risk_pct", 2.0, 5.0, step=0.5),
        "min_profit_multiple": trial.suggest_float("min_profit_multiple", 2.0, 4.0, step=0.5),
        "cooldown_hours": trial.suggest_int("cooldown_hours", 12, 72, step=12),
        "max_hold_hours": trial.suggest_int("max_hold_hours", 48, 240, step=24),
        "quiet_atr_threshold": trial.suggest_float("quiet_atr_threshold", 0.8, 1.5, step=0.1),
        "regime_adx_threshold": trial.suggest_float("regime_adx_threshold", 20, 30, step=2),
    }
```

- [ ] **Step 4: Update optimization target**

```python
def _objective_score(result):
    sharpe = result.sharpe_ratio
    pf = result.profit_factor if hasattr(result, 'profit_factor') else 1.0
    ppf = result.profit_per_fee if hasattr(result, 'profit_per_fee') else 0.0
    return sharpe * 0.4 + pf * 0.3 + ppf * 0.3
```

- [ ] **Step 5: Update adoption criteria**

```python
def _check_adoption(windows):
    sharpes = [w.sharpe for w in windows]
    pnls = [w.pnl for w in windows]
    ppfs = [w.profit_per_fee for w in windows if hasattr(w, 'profit_per_fee')]
    median_sharpe = statistics.median(sharpes)
    profitable = sum(1 for p in pnls if p > 0)
    sharpe_std = statistics.stdev(sharpes) if len(sharpes) > 1 else 999
    avg_ppf = statistics.mean(ppfs) if ppfs else 0.0

    return (
        len(windows) >= MIN_WINDOWS
        and median_sharpe > 0.3
        and profitable >= len(windows) * 0.70
        and statistics.mean(pnls) > 0
        and sharpe_std < 1.5
        and avg_ppf > 1.5  # per spec: walk-forward adoption threshold is 1.5
    )
```

Also update `WFWindow` to store `profit_per_fee`:
```python
@dataclass
class WFWindow:
    sharpe: float = 0.0
    pnl: float = 0.0
    params: dict = field(default_factory=dict)
    profit_per_fee: float = 0.0  # NEW: track for adoption criteria
```

- [ ] **Step 6: Update champion seed values**

```python
CHAMPION_PARAMS = {
    "atr_multiplier": 3.5,
    "rr_ratio": 2.5,
    "base_risk_pct": 3.0,
    "min_profit_multiple": 3.0,
    "cooldown_hours": 24,
    "max_hold_hours": 120,
    "quiet_atr_threshold": 1.0,
    "regime_adx_threshold": 25,
}
```

- [ ] **Step 7: Run walk-forward tests**

Run: `python -m pytest tests/test_walk_forward.py tests/test_wf_*.py -v`
Fix any failures.

- [ ] **Step 8: Commit**

```bash
git add bot/learning/walk_forward.py
git commit -m "feat: update walk-forward for new param space, 180d windows, stricter adoption"
```

---

### Task 17: Wire Everything in main.py

**Files:**
- Modify: `bot/main.py`

- [ ] **Step 1: Read current main.py**

Understand the startup flow, service creation, scheduler setup.

- [ ] **Step 2: Wire new components**

- Import and instantiate `LimitOrderManager`
- Pass it to the trading loop
- Update scheduler: remove 30s strategy cycle, add 5m stop cycle + 1h signal cycle
- Remove 1m candle WebSocket subscription
- Update config references to use new defaults

- [ ] **Step 3: Update WebSocket subscriptions**

```python
# Subscribe to 5m candles (not 1m)
await ws.subscribe_candles(pairs, interval="5m")
```

- [ ] **Step 4: Test startup**

```bash
docker compose up bot --no-deps -d && docker compose logs -f bot
```

Verify: bot starts without errors, connects to WebSocket, logs show 5m candle processing.

- [ ] **Step 5: Commit**

```bash
git add bot/main.py
git commit -m "feat: wire limit order manager, update scheduler cycles, remove 1m subscription"
```

---

### Task 18: 5-Year Backtest Validation

**Files:**
- Modify: `scripts/strategy_showdown.py`

- [ ] **Step 1: Update the showdown script for new engine**

Replace the 25-config list with the 4 new strategies + confluence combinations. Update to use new parameter names. Compare head-to-head with old results.

- [ ] **Step 2: Run the 5-year backtest**

```bash
python scripts/strategy_showdown.py
```

Expected runtime: ~20-30 minutes for 3 pairs x 5 years.

- [ ] **Step 3: Verify targets**

Check output against spec success criteria:
- [ ] Net positive total P&L (old: -2,955 EUR)
- [ ] Sharpe > 0.5
- [ ] Profit per fee EUR > 2.0
- [ ] Fewer than 500 total trades

- [ ] **Step 4: If targets not met, iterate**

Adjust parameters and re-run. The walk-forward optimization can help find better params.

- [ ] **Step 5: Commit results and final code**

```bash
git add scripts/strategy_showdown.py
git commit -m "feat: validate new engine with 5-year backtest"
```

---

### Task 19: Docker Rebuild and Paper Trading Launch

- [ ] **Step 1: Rebuild Docker image**

```bash
docker compose build bot
```

- [ ] **Step 2: Start in paper trading mode**

```bash
docker compose up -d
```

- [ ] **Step 3: Monitor for 24 hours**

Check logs for:
- 1h signal evaluations firing at hour boundaries
- Fee-aware gate rejecting low-quality setups
- QUIET regime skipping during low-vol periods
- Limit order placement and fill/timeout behavior
- Stop checks every 5m candle

- [ ] **Step 4: Commit any fixes discovered during monitoring**

```bash
git add -A
git commit -m "fix: address issues found during paper trading monitoring"
```
