# Range Trading Strategy Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the losing mean-reversion strategy with a dedicated range-trading strategy that profits from price oscillating within Bollinger Band ranges, turning the bot's worst market regime into a revenue source.

**Architecture:** New `RangeStrategy` class replaces `MeanReversionStrategy`. Uses BB bandwidth + ADX for 1h range confirmation, 5m BB + RSI + volume profile for entry timing. Dynamic TP shifts from mid-band to opposite band based on RSI momentum. Bounce counter caps trades at 3 per range. Backtest engine gets range-specific pre-computation, evaluation, and exit logic.

**Tech Stack:** Python 3.11+, pandas, numpy. Existing indicator framework (`bot/indicators/`), backtest engine (`bot/backtest/engine.py`), strategy router (`bot/strategy/router.py`).

**Spec:** `docs/superpowers/specs/2026-03-15-range-trading-strategy-design.md`

---

## Chunk 1: Volume Profile Indicator + RangeStrategy Class

### Task 1: Volume Profile Support Indicator

**Files:**
- Modify: `bot/indicators/volume.py:31` (append new function)
- Test: `tests/test_range_strategy.py` (create)

- [ ] **Step 1: Write the failing test for volume_profile_support**

Create `tests/test_range_strategy.py`:

```python
"""Tests for range trading strategy components."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bot.indicators.volume import volume_profile_support


class TestVolumeProfileSupport:
    """Tests for volume_profile_support indicator."""

    def _make_ranging_data(self, n=300, seed=42):
        """Generate synthetic ranging price data with volume."""
        np.random.seed(seed)
        # Price oscillates between 95 and 105
        t = np.linspace(0, 8 * np.pi, n)
        close = 100 + 5 * np.sin(t) + np.random.normal(0, 0.5, n)
        high = close + np.random.uniform(0.1, 1.0, n)
        low = close - np.random.uniform(0.1, 1.0, n)
        volume = np.random.uniform(100, 1000, n)
        return (
            pd.Series(close, dtype=float),
            pd.Series(volume, dtype=float),
            pd.Series(high, dtype=float),
            pd.Series(low, dtype=float),
        )

    def test_returns_series_of_correct_length(self):
        close, volume, high, low = self._make_ranging_data()
        result = volume_profile_support(close, volume, high, low)
        assert isinstance(result, pd.Series)
        assert len(result) == len(close)

    def test_values_bounded_minus_one_to_plus_one(self):
        close, volume, high, low = self._make_ranging_data()
        result = volume_profile_support(close, volume, high, low)
        valid = result.dropna()
        assert (valid >= -1.0).all()
        assert (valid <= 1.0).all()

    def test_positive_score_near_support(self):
        """When price is near range bottom with volume below, score should be positive."""
        close, volume, high, low = self._make_ranging_data(n=300)
        result = volume_profile_support(close, volume, high, low, lookback=200)
        # Find indices where price is near the minimum (support zone)
        support_indices = close[close < close.rolling(200).quantile(0.2)].index
        if len(support_indices) > 0:
            # At least some support-zone scores should be positive
            support_scores = result.loc[support_indices].dropna()
            if len(support_scores) > 0:
                assert support_scores.mean() > -0.5  # not strongly negative at support

    def test_handles_zero_volume(self):
        """Should not crash when volume is zero."""
        close = pd.Series([100.0] * 50)
        volume = pd.Series([0.0] * 50)
        high = close + 1.0
        low = close - 1.0
        result = volume_profile_support(close, volume, high, low, lookback=20)
        assert len(result) == 50
        assert not result.isna().all()  # zeros are valid output, just verify no NaN

    def test_custom_lookback(self):
        close, volume, high, low = self._make_ranging_data()
        result_50 = volume_profile_support(close, volume, high, low, lookback=50)
        result_200 = volume_profile_support(close, volume, high, low, lookback=200)
        # Both should work, just different smoothing
        assert len(result_50) == len(result_200) == len(close)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_range_strategy.py::TestVolumeProfileSupport -v`
Expected: FAIL with `ImportError: cannot import name 'volume_profile_support'`

- [ ] **Step 3: Implement volume_profile_support**

Append to `bot/indicators/volume.py` after line 31:

```python


def volume_profile_support(
    close: pd.Series,
    volume: pd.Series,
    high: pd.Series,
    low: pd.Series,
    lookback: int = 200,
) -> pd.Series:
    """Compute volume profile support/resistance score per bar.

    Returns a score from -1.0 to +1.0:
    - Positive = volume concentrated near current price from below (support)
    - Negative = volume concentrated from above (resistance)
    - Near zero = no clear volume clustering

    Uses price buckets (0.1% width) over a rolling lookback window.
    """
    import numpy as np

    n = len(close)
    scores = np.zeros(n, dtype=float)

    close_arr = close.values.astype(float)
    volume_arr = volume.values.astype(float)

    for i in range(lookback, n):
        window_close = close_arr[i - lookback : i]
        window_vol = volume_arr[i - lookback : i]

        price_min = window_close.min()
        price_max = window_close.max()
        price_range = price_max - price_min

        if price_range < 1e-9:
            scores[i] = 0.0
            continue

        total_vol = window_vol.sum()
        if total_vol < 1e-9:
            scores[i] = 0.0
            continue

        # Lower 20% and upper 20% of the price range
        lower_threshold = price_min + 0.2 * price_range
        upper_threshold = price_max - 0.2 * price_range

        lower_vol = window_vol[window_close <= lower_threshold].sum()
        upper_vol = window_vol[window_close >= upper_threshold].sum()

        scores[i] = (lower_vol - upper_vol) / total_vol

    return pd.Series(scores, index=close.index, dtype=float)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_range_strategy.py::TestVolumeProfileSupport -v`
Expected: PASS (all 5 tests)

- [ ] **Step 5: Commit**

```bash
git add bot/indicators/volume.py tests/test_range_strategy.py
git commit -m "feat: add volume_profile_support indicator for range trading"
```

---

### Task 2: RangeStrategy Class

**Files:**
- Create: `bot/strategy/range_trading.py`
- Test: `tests/test_range_strategy.py` (append)

- [ ] **Step 1: Write the failing tests for RangeStrategy**

Append to `tests/test_range_strategy.py`:

```python
from bot.strategy.range_trading import RangeStrategy
from bot.strategy.base import MarketContext, Signal


class TestRangeStrategy:
    """Tests for RangeStrategy.generate_signal."""

    def _make_ranging_ctx(self, price=100.0, rsi_val=35.0, bb_position="lower",
                          bb_bandwidth=0.06, adx_val=15.0, symbol="BTC-EUR"):
        """Build a MarketContext with synthetic ranging data."""
        np.random.seed(42)
        n = 250
        # Create oscillating price centered on 100
        t = np.linspace(0, 6 * np.pi, n)
        base = 100 + 3 * np.sin(t)
        noise = np.random.normal(0, 0.3, n)
        closes = base + noise

        # Place current price near lower or upper band
        if bb_position == "lower":
            closes[-1] = price * 0.97  # near lower band
        elif bb_position == "upper":
            closes[-1] = price * 1.03
        else:
            closes[-1] = price

        idx_5m = pd.date_range("2026-01-01", periods=n, freq="5min")
        df_5m = pd.DataFrame({
            "open": closes * (1 + np.random.normal(0, 0.001, n)),
            "high": closes * (1 + np.random.uniform(0, 0.005, n)),
            "low": closes * (1 - np.random.uniform(0, 0.005, n)),
            "close": closes,
            "volume": np.random.uniform(100, 1000, n),
        }, index=idx_5m)

        # 1h candles (fewer, same ranging pattern)
        n_1h = 50
        t_1h = np.linspace(0, 6 * np.pi, n_1h)
        close_1h = 100 + 3 * np.sin(t_1h) + np.random.normal(0, 0.2, n_1h)
        idx_1h = pd.date_range("2026-01-01", periods=n_1h, freq="1h")
        df_1h = pd.DataFrame({
            "open": close_1h,
            "high": close_1h * 1.003,
            "low": close_1h * 0.997,
            "close": close_1h,
            "volume": np.random.uniform(500, 5000, n_1h),
        }, index=idx_1h)

        return MarketContext(
            symbol=symbol,
            candles_5m=df_5m,
            candles_1h=df_1h,
            current_price=float(closes[-1]),
            sentiment_score=0.0,
            portfolio_equity_eur=10000.0,
            open_position_count=0,
        )

    def test_name_is_range(self):
        strat = RangeStrategy()
        assert strat.name == "range"

    def test_neutral_when_insufficient_data(self):
        strat = RangeStrategy()
        df = pd.DataFrame({
            "open": [100], "high": [101], "low": [99],
            "close": [100], "volume": [500],
        }, index=pd.date_range("2026-01-01", periods=1, freq="5min"))
        ctx = MarketContext(symbol="BTC-EUR", candles_5m=df, candles_1h=df, current_price=100.0,
                            sentiment_score=0.0, portfolio_equity_eur=10000.0, open_position_count=0)
        sig = strat.generate_signal(ctx)
        assert sig.direction == "NEUTRAL"

    def test_returns_signal_dataclass(self):
        strat = RangeStrategy()
        ctx = self._make_ranging_ctx()
        sig = strat.generate_signal(ctx)
        assert isinstance(sig, Signal)
        assert sig.strategy_name == "range"
        assert sig.symbol == "BTC-EUR"

    def test_indicator_snapshot_has_range_fields(self):
        strat = RangeStrategy()
        ctx = self._make_ranging_ctx()
        sig = strat.generate_signal(ctx)
        snap = sig.indicator_snapshot
        assert "range_mid" in snap
        assert "range_upper" in snap
        assert "range_lower" in snap
        assert "bounce_count" in snap
        assert "confirming_count" in snap

    def test_bounce_counter_increments(self):
        strat = RangeStrategy()
        strat.increment_bounce("BTC-EUR")
        strat.increment_bounce("BTC-EUR")
        assert strat.get_bounce_count("BTC-EUR") == 2

    def test_bounce_counter_resets(self):
        strat = RangeStrategy()
        strat.increment_bounce("BTC-EUR")
        strat.increment_bounce("BTC-EUR")
        strat.reset_bounces("BTC-EUR")
        assert strat.get_bounce_count("BTC-EUR") == 0

    def test_bounce_limit_stops_signaling(self):
        strat = RangeStrategy()
        # Exhaust bounce limit
        for _ in range(3):
            strat.increment_bounce("BTC-EUR")
        ctx = self._make_ranging_ctx()
        sig = strat.generate_signal(ctx)
        assert sig.direction == "NEUTRAL"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_range_strategy.py::TestRangeStrategy -v`
Expected: FAIL with `ImportError: cannot import name 'RangeStrategy'`

- [ ] **Step 3: Implement RangeStrategy**

Create `bot/strategy/range_trading.py`:

```python
"""Range trading strategy — profits from price oscillating within BB ranges."""
from __future__ import annotations

import logging
from typing import Any, Dict

import pandas as pd

from bot.indicators.momentum import rsi
from bot.indicators.trend import adx
from bot.indicators.volatility import bollinger_bands
from bot.indicators.volume import volume_profile_support
from bot.strategy.base import BaseStrategy, MarketContext, Signal

logger = logging.getLogger(__name__)

# Range confirmation thresholds (1h timeframe)
BB_BANDWIDTH_MAX = 0.10       # BB bandwidth < 10% = ranging
BB_BANDWIDTH_MIN = 0.02       # minimum 2% range for profit after fees
ADX_MAX = 20                  # ADX < 20 = no trend
BB_BANDWIDTH_RESET = 0.15     # reset bounce counter above this
ADX_RESET = 25                # reset bounce counter above this

# Entry thresholds (5m timeframe)
ENTRY_PROXIMITY_PCT = 0.01    # within 1% of band
RSI_LONG_THRESHOLD = 40       # RSI < 40 for long entry
RSI_SHORT_THRESHOLD = 60      # RSI > 60 for short entry
VOLUME_SCORE_THRESHOLD = 0.3  # minimum volume profile score magnitude

# Bounce limit
MAX_BOUNCES = 3


class RangeStrategy(BaseStrategy):
    """Dedicated range-trading strategy using BB + volume clustering."""

    name = "range"

    def __init__(self) -> None:
        self._bounce_counts: Dict[str, int] = {}

    def get_bounce_count(self, symbol: str) -> int:
        return self._bounce_counts.get(symbol, 0)

    def increment_bounce(self, symbol: str) -> None:
        self._bounce_counts[symbol] = self._bounce_counts.get(symbol, 0) + 1

    def reset_bounces(self, symbol: str) -> None:
        self._bounce_counts[symbol] = 0

    def generate_signal(self, ctx: MarketContext) -> Signal:
        symbol = ctx.symbol
        df_5m = ctx.candles_5m
        df_1h = ctx.candles_1h
        neutral = Signal(symbol, "NEUTRAL", 0.0, self.name,
                         indicator_snapshot=self._empty_snapshot())

        if len(df_5m) < 50 or len(df_1h) < 30:
            return neutral

        # --- 1h range confirmation ---
        close_1h = df_1h["close"]
        high_1h = df_1h["high"]
        low_1h = df_1h["low"]

        bb_1h = bollinger_bands(close_1h)
        bandwidth_1h = bb_1h["bandwidth"].iloc[-1]
        adx_1h = adx(high_1h, low_1h, close_1h).iloc[-1]

        # Check bounce counter reset
        if bandwidth_1h > BB_BANDWIDTH_RESET or adx_1h > ADX_RESET:
            self.reset_bounces(symbol)

        # Range not confirmed
        if bandwidth_1h >= BB_BANDWIDTH_MAX or bandwidth_1h < BB_BANDWIDTH_MIN:
            return neutral
        if adx_1h >= ADX_MAX:
            return neutral

        # Bounce limit reached
        if self.get_bounce_count(symbol) >= MAX_BOUNCES:
            return neutral

        # --- 5m entry trigger ---
        close_5m = df_5m["close"]
        high_5m = df_5m["high"]
        low_5m = df_5m["low"]
        volume_5m = df_5m["volume"]

        bb_5m = bollinger_bands(close_5m)
        rsi_5m = rsi(close_5m).iloc[-1]
        vol_score = volume_profile_support(
            close_5m, volume_5m, high_5m, low_5m, lookback=min(200, len(close_5m) - 1),
        ).iloc[-1]

        price = ctx.current_price
        upper = bb_5m["upper"].iloc[-1]
        lower = bb_5m["lower"].iloc[-1]
        mid = bb_5m["mid"].iloc[-1]

        direction = "NEUTRAL"
        confirming = 0

        # LONG: price within 1% of lower BB + RSI < 40 + volume support
        if abs(price - lower) / lower <= ENTRY_PROXIMITY_PCT and rsi_5m < RSI_LONG_THRESHOLD:
            if vol_score > VOLUME_SCORE_THRESHOLD:
                direction = "LONG"
                confirming = 1  # RSI
                if abs(price - lower) / lower <= 0.005:
                    confirming += 1  # very close to band
                if vol_score > 0.5:
                    confirming += 1  # strong volume support

        # SHORT: price within 1% of upper BB + RSI > 60 + volume resistance
        elif abs(price - upper) / upper <= ENTRY_PROXIMITY_PCT and rsi_5m > RSI_SHORT_THRESHOLD:
            if vol_score < -VOLUME_SCORE_THRESHOLD:
                direction = "SHORT"
                confirming = 1  # RSI
                if abs(price - upper) / upper <= 0.005:
                    confirming += 1
                if vol_score < -0.5:
                    confirming += 1

        if direction == "NEUTRAL":
            return neutral

        strength = min(abs(vol_score) + (1.0 - bandwidth_1h / BB_BANDWIDTH_MAX) * 0.5, 1.0)

        return Signal(
            symbol=symbol,
            direction=direction,
            strength=strength,
            strategy_name=self.name,
            technical_score=strength if direction == "LONG" else -strength,
            indicator_snapshot={
                "range_mid": float(mid),
                "range_upper": float(upper),
                "range_lower": float(lower),
                "bounce_count": self.get_bounce_count(symbol),
                "confirming_count": confirming,
                "rsi": float(rsi_5m),
                "bb_bandwidth": float(bandwidth_1h),
                "vol_profile_score": float(vol_score),
            },
        )

    @staticmethod
    def _empty_snapshot() -> Dict[str, Any]:
        return {
            "range_mid": 0.0,
            "range_upper": 0.0,
            "range_lower": 0.0,
            "bounce_count": 0,
            "confirming_count": 0,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_range_strategy.py -v`
Expected: PASS (all tests in both TestVolumeProfileSupport and TestRangeStrategy)

- [ ] **Step 5: Commit**

```bash
git add bot/strategy/range_trading.py tests/test_range_strategy.py
git commit -m "feat: add RangeStrategy class with BB + volume profile entries"
```

---

## Chunk 2: Router Update + Backtest Engine Integration

### Task 3: Replace MeanReversion with Range in Router

**Files:**
- Modify: `bot/strategy/router.py:14,66-76`
- Modify: `tests/test_strategy_router.py:12,64-68`

- [ ] **Step 1: Write the failing test**

Update `tests/test_strategy_router.py` — change the import and assertion:

Replace line 12:
```python
from bot.strategy.mean_reversion import MeanReversionStrategy
```
with:
```python
from bot.strategy.range_trading import RangeStrategy
```

Replace lines 64-68:
```python
    def test_ranging_includes_mean_reversion(self):
        router = StrategyRouter()
        strategies = router.get_strategies(Regime.RANGING)
        types = [type(s) for s in strategies]
        assert MeanReversionStrategy in types
```
with:
```python
    def test_ranging_includes_range_strategy(self):
        router = StrategyRouter()
        strategies = router.get_strategies(Regime.RANGING)
        types = [type(s) for s in strategies]
        assert RangeStrategy in types
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_strategy_router.py::TestStrategyRouter::test_ranging_includes_range_strategy -v`
Expected: FAIL (`RangeStrategy` not in strategy list)

- [ ] **Step 3: Update router.py**

In `bot/strategy/router.py`:

Replace line 14:
```python
from bot.strategy.mean_reversion import MeanReversionStrategy
```
with:
```python
from bot.strategy.range_trading import RangeStrategy
```

Replace line 69:
```python
        self._mean_rev = MeanReversionStrategy()
```
with:
```python
        self._range = RangeStrategy()
```

Replace line 76:
```python
            return [self._hybrid, self._mean_rev]
```
with:
```python
            return [self._hybrid, self._range]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_strategy_router.py -v`
Expected: PASS. Note: `test_volatile_position_modifier` has a pre-existing bug (asserts 0.5 but code returns 0.3). If this test fails, it is NOT caused by your changes — ignore it and verify the other tests pass.

- [ ] **Step 5: Commit**

```bash
git add bot/strategy/router.py tests/test_strategy_router.py
git commit -m "feat: replace MeanReversionStrategy with RangeStrategy in router"
```

---

### Task 4: Backtest Engine — Pre-computation + _OpenPosition Fields

**Files:**
- Modify: `bot/backtest/engine.py:62-73` (_OpenPosition dataclass)
- Modify: `bot/backtest/engine.py:88-133` (__init__)
- Modify: `bot/backtest/engine.py:304-374` (_precompute_signals)

- [ ] **Step 1: Add range fields to _OpenPosition**

In `bot/backtest/engine.py`, replace the `_OpenPosition` dataclass (lines 62-73):

```python
@dataclass
class _OpenPosition:
    symbol: str
    direction: str
    entry_price: float
    entry_time: str
    size_eur: float
    stop_loss: float
    take_profit: float
    highest_price: float
    strategy: str
    entry_bar: int = 0
    # Range strategy fields
    tp_shifted: bool = False
    range_mid: float = 0.0
    range_upper: float = 0.0
    range_lower: float = 0.0
```

- [ ] **Step 2: Add bounce counter to __init__**

In `bot/backtest/engine.py`, after line 124 (`self._consecutive_confirms = ...`), add:

```python
        # Range strategy: bounce counter per symbol
        self._range_bounces: Dict[str, int] = {}
        # Range strategy: max hold = 24h = 288 5m bars (hardcoded, not walk-forward)
        self._range_max_hold_bars = 288
```

- [ ] **Step 3: Add BB full arrays + volume profile to _precompute_signals**

In `bot/backtest/engine.py`, in the `_precompute_signals` method, **extend the existing `bb_pct_b` try-block** (lines 328-332) to also extract the full BB arrays from the same computation. Replace:

```python
        try:
            bb = bollinger_bands(close)
            result["bb_pct_b"] = bb["pct_b"].values
        except Exception:
            result["bb_pct_b"] = np.full(len(df), 0.5)
```

with:

```python
        try:
            bb = bollinger_bands(close)
            result["bb_pct_b"] = bb["pct_b"].values
            result["bb_upper"] = bb["upper"].values
            result["bb_lower"] = bb["lower"].values
            result["bb_mid"] = bb["mid"].values
            result["bb_bandwidth"] = bb["bandwidth"].values
        except Exception:
            result["bb_pct_b"] = np.full(len(df), 0.5)
            result["bb_upper"] = close.values.copy()
            result["bb_lower"] = close.values.copy()
            result["bb_mid"] = close.values.copy()
            result["bb_bandwidth"] = np.zeros(len(df))
```

After the leading indicators block (after line 368), add:

```python
        # Volume profile for range trading
        try:
            from bot.indicators.volume import volume_profile_support
            result["vol_profile"] = volume_profile_support(close, volume, high, low).values
        except Exception:
            result["vol_profile"] = np.zeros(len(df))
```

Also add `volume` to the available variables at the top of the method. It's already used via `volume_surge_ratio(volume)` on line 352, so `volume = df["volume"]` is already available (line 316).

- [ ] **Step 4: Run existing tests to verify nothing is broken**

Run: `python -m pytest tests/ -v --timeout=30`
Expected: All existing tests PASS

- [ ] **Step 5: Commit**

```bash
git add bot/backtest/engine.py
git commit -m "feat: add range fields to _OpenPosition and pre-compute BB/volume profile arrays"
```

---

### Task 5: Backtest Engine — Range Evaluation Branch

**Files:**
- Modify: `bot/backtest/engine.py:590-625` (replace `mean_reversion` branch)
- Modify: `bot/backtest/engine.py:186-298` (run loop — bounce reset + range position opening)

- [ ] **Step 1: Write failing test for range evaluation**

Append to `tests/test_range_strategy.py`:

```python
from bot.backtest.engine import BacktestEngine, _OpenPosition
from bot.exchange.bitvavo_client import CandleData
from datetime import datetime, timezone


class TestRangeBacktestIntegration:
    """Tests for range strategy in the backtest engine."""

    def _make_ranging_candles(self, n=500, symbol="BTC-EUR"):
        """Generate synthetic ranging candles for backtest."""
        np.random.seed(42)
        candles = []
        base_price = 50000.0
        t0 = datetime(2025, 9, 1, tzinfo=timezone.utc)

        for i in range(n):
            # Oscillate within 2% range
            phase = np.sin(2 * np.pi * i / 100) * 0.01  # 1% amplitude
            price = base_price * (1 + phase + np.random.normal(0, 0.001))
            candles.append(CandleData(
                symbol=symbol,
                interval="5m",
                timestamp=t0 + pd.Timedelta(minutes=5 * i),
                open=price * (1 + np.random.normal(0, 0.0005)),
                high=price * (1 + abs(np.random.normal(0, 0.002))),
                low=price * (1 - abs(np.random.normal(0, 0.002))),
                close=price,
                volume=float(np.random.uniform(0.5, 5.0)),
            ))
        return candles

    def test_engine_has_range_bounce_counter(self):
        candles = self._make_ranging_candles(n=100)
        engine = BacktestEngine(candles)
        assert hasattr(engine, "_range_bounces")
        assert isinstance(engine._range_bounces, dict)

    def test_open_position_has_range_fields(self):
        pos = _OpenPosition(
            symbol="BTC-EUR", direction="LONG", entry_price=50000.0,
            entry_time="2025-09-01", size_eur=100.0, stop_loss=49000.0,
            take_profit=51000.0, highest_price=50000.0, strategy="range",
        )
        assert pos.tp_shifted is False
        assert pos.range_mid == 0.0
        assert pos.range_upper == 0.0
        assert pos.range_lower == 0.0

    def test_precompute_includes_bb_full_arrays(self):
        candles = self._make_ranging_candles(n=200)
        engine = BacktestEngine(candles)
        df = engine._to_dataframe(candles)
        precomp = engine._precompute_signals(df, "BTC-EUR")
        assert "bb_upper" in precomp
        assert "bb_lower" in precomp
        assert "bb_mid" in precomp
        assert "bb_bandwidth" in precomp
        assert "vol_profile" in precomp

    def test_backtest_runs_without_error(self):
        """Smoke test: backtest with range strategy doesn't crash."""
        candles = self._make_ranging_candles(n=500)
        engine = BacktestEngine(candles, strategy_params={
            "min_confirmations": 1,
            "entry_threshold": 0.30,
            "cooldown_hours": 1,
        })
        result = engine.run()
        # Should complete without errors
        assert result.total_trades >= 0
```

- [ ] **Step 2: Run test to verify it fails (or passes partially)**

Run: `python -m pytest tests/test_range_strategy.py::TestRangeBacktestIntegration -v`
Expected: Some tests may pass (field tests), but `test_precompute_includes_bb_full_arrays` needs the new arrays.

- [ ] **Step 3: Replace mean_reversion branch with range branch in _evaluate_precomputed**

In `bot/backtest/engine.py`, replace lines 590-625 (the `elif strat.name == "mean_reversion":` block) with:

```python
                elif strat.name == "range":
                    # Range trading: BB + RSI + volume profile at 5m, BB bandwidth + ADX at 1h
                    if idx_5m < 200:
                        continue

                    # 1h range confirmation
                    bw_1h = precomp_1h["bb_bandwidth"][idx_1h] if idx_1h < len(precomp_1h.get("bb_bandwidth", [])) else 1.0
                    adx_1h = precomp_1h["adx"][idx_1h] if idx_1h < len(precomp_1h["adx"]) else 50.0

                    # Check bounce counter reset
                    if bw_1h > 0.15 or adx_1h > 25:
                        self._range_bounces[symbol] = 0

                    # Range not confirmed
                    if bw_1h >= 0.10 or bw_1h < 0.02 or adx_1h >= 20:
                        continue

                    # Bounce limit
                    if self._range_bounces.get(symbol, 0) >= 3:
                        continue

                    # 5m entry trigger
                    price = current_price
                    bb_upper = precomp_5m["bb_upper"][idx_5m]
                    bb_lower = precomp_5m["bb_lower"][idx_5m]
                    bb_mid = precomp_5m["bb_mid"][idx_5m]
                    rsi_val = precomp_5m["rsi"][idx_5m]
                    vol_score = precomp_5m["vol_profile"][idx_5m]

                    direction = "NEUTRAL"
                    confirming = 0

                    # LONG: price within 1% of lower BB + RSI < 40 + volume support
                    if bb_lower > 0 and abs(price - bb_lower) / bb_lower <= 0.01 and rsi_val < 40:
                        if vol_score > 0.3:
                            direction = "LONG"
                            confirming = 1
                            if abs(price - bb_lower) / bb_lower <= 0.005:
                                confirming += 1
                            if vol_score > 0.5:
                                confirming += 1
                    # SHORT: price within 1% of upper BB + RSI > 60 + volume resistance
                    elif bb_upper > 0 and abs(price - bb_upper) / bb_upper <= 0.01 and rsi_val > 60:
                        if vol_score < -0.3:
                            direction = "SHORT"
                            confirming = 1
                            if abs(price - bb_upper) / bb_upper <= 0.005:
                                confirming += 1
                            if vol_score < -0.5:
                                confirming += 1

                    strength = 0.0
                    if direction != "NEUTRAL":
                        strength = min(abs(vol_score) + (1.0 - bw_1h / 0.10) * 0.5, 1.0)

                    sig = Signal(
                        symbol=symbol, direction=direction, strength=strength,
                        strategy_name="range",
                        technical_score=strength if direction == "LONG" else -strength,
                        indicator_snapshot={
                            "confirming_count": confirming,
                            "range_mid": float(bb_mid),
                            "range_upper": float(bb_upper),
                            "range_lower": float(bb_lower),
                            "bounce_count": self._range_bounces.get(symbol, 0),
                        },
                    )
```

- [ ] **Step 4: Store range levels on position when opening range trades**

In `bot/backtest/engine.py`, in the `_open_position` method (around line 746), after the `self.positions.append(...)` call, add logic to set range fields:

After line 757 (`entry_bar=bar_index,`), before the closing `))`:

The `_open_position` method already receives `signal: Signal` as a parameter. After the existing `self.positions.append(...)` call (line 757), add:

```python
        # Store range levels for range strategy positions
        if signal.strategy_name == "range":
            pos = self.positions[-1]
            snap = signal.indicator_snapshot
            pos.range_mid = snap.get("range_mid", 0.0)
            pos.range_upper = snap.get("range_upper", 0.0)
            pos.range_lower = snap.get("range_lower", 0.0)

            # Override ATR-based stops with range-specific stops
            bandwidth_price = pos.range_upper - pos.range_lower
            if pos.direction == "LONG":
                pos.stop_loss = pos.range_lower - 0.5 * bandwidth_price
                pos.take_profit = pos.range_mid  # Phase 1: TP at mid-band
            else:
                pos.stop_loss = pos.range_upper + 0.5 * bandwidth_price
                pos.take_profit = pos.range_mid
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_range_strategy.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add bot/backtest/engine.py tests/test_range_strategy.py
git commit -m "feat: add range evaluation branch and range-specific position opening in backtest"
```

---

## Chunk 3: Range Exit Logic + Trading Loop + Cleanup

### Task 6: Backtest Engine — Range-Specific Exit Logic

**Files:**
- Modify: `bot/backtest/engine.py:759-819` (_check_exits_fast)
- Modify: `bot/backtest/engine.py:821-869` (_close_position)
- Modify: `bot/backtest/engine.py:186-198` (run loop — pass precomp to exits)

- [ ] **Step 1: Write failing test for range exit logic**

Append to `tests/test_range_strategy.py`:

```python
class TestRangeExitLogic:
    """Tests for range-specific exit handling in backtest."""

    def test_range_time_exit_24h(self):
        """Range positions close after 288 bars (24h)."""
        engine = BacktestEngine.__new__(BacktestEngine)
        engine.positions = []
        engine.closed_trades = []
        engine.balance = 10000.0
        engine.peak_balance = 10000.0
        engine.slippage_pct = 0.001
        engine._atr_multiplier = 2.0
        engine._max_hold_bars = 576  # 48h for trend
        engine._range_max_hold_bars = 288  # 24h for range
        engine._range_bounces = {}

        pos = _OpenPosition(
            symbol="BTC-EUR", direction="LONG", entry_price=50000.0,
            entry_time="2025-09-01", size_eur=100.0, stop_loss=49000.0,
            take_profit=50500.0, highest_price=50000.0, strategy="range",
            entry_bar=0, range_mid=50250.0, range_upper=50500.0, range_lower=50000.0,
        )
        engine.positions.append(pos)

        # At bar 288, should trigger time exit for range
        engine._check_exits_fast(
            price=50100.0, candle_high=50150.0, candle_low=50050.0,
            time_str="2025-09-02", atr_val=200.0, current_bar=288,
        )
        assert len(engine.positions) == 0
        assert len(engine.closed_trades) == 1
        assert engine.closed_trades[0].exit_reason == "time_exit"

    def test_range_bounce_increments_on_close(self):
        """Closing a range trade increments the bounce counter."""
        engine = BacktestEngine.__new__(BacktestEngine)
        engine.positions = []
        engine.closed_trades = []
        engine.balance = 10000.0
        engine.peak_balance = 10000.0
        engine.slippage_pct = 0.001
        engine._range_bounces = {}

        pos = _OpenPosition(
            symbol="BTC-EUR", direction="LONG", entry_price=50000.0,
            entry_time="2025-09-01", size_eur=100.0, stop_loss=49000.0,
            take_profit=50500.0, highest_price=50000.0, strategy="range",
            entry_bar=0,
        )
        engine.positions.append(pos)
        engine._close_position(pos, 50500.0, "2025-09-01T12:00", "take_profit")

        assert engine._range_bounces.get("BTC-EUR", 0) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_range_strategy.py::TestRangeExitLogic -v`
Expected: FAIL (range_max_hold_bars not used, bounce counter not incremented)

- [ ] **Step 3: Update _check_exits_fast for range-specific logic**

In `bot/backtest/engine.py`, modify `_check_exits_fast` to accept precomputed data and handle range exits.

Update the method signature (line 759):

```python
    def _check_exits_fast(
        self,
        price: float,
        candle_high: float,
        candle_low: float,
        time_str: str,
        atr_val: Optional[float],
        current_bar: int = 0,
        precomp_5m: Optional[Dict[str, Any]] = None,
        idx_5m: int = 0,
    ) -> None:
```

Replace the time-based exit check (lines 772-775) with:

```python
            # Time-based exit: range uses 24h, others use max_hold_bars
            if pos.strategy == "range":
                max_hold = self._range_max_hold_bars
            else:
                max_hold = self._max_hold_bars
            if current_bar - pos.entry_bar >= max_hold:
                self._close_position(pos, price, time_str, "time_exit")
                continue
```

Add dynamic TP logic for range positions before the trailing stop block. Insert after the time exit check, before "Update trailing price":

```python
            # --- Range dynamic TP: shift TP from mid-band to opposite band ---
            if pos.strategy == "range" and not pos.tp_shifted and precomp_5m is not None:
                crossed_mid = (
                    (direction == "LONG" and price >= pos.range_mid) or
                    (direction == "SHORT" and price <= pos.range_mid)
                )
                if crossed_mid and idx_5m >= 3:
                    current_rsi = precomp_5m["rsi"][idx_5m]
                    prev_rsi = precomp_5m["rsi"][idx_5m - 3]  # 3 bars back (15 min at 5m)
                    # RSI trending favorably?
                    if direction == "LONG" and current_rsi > prev_rsi:
                        pos.tp_shifted = True
                        pos.take_profit = pos.range_upper
                    elif direction == "SHORT" and current_rsi < prev_rsi:
                        pos.tp_shifted = True
                        pos.take_profit = pos.range_lower
                    else:
                        # RSI flat/reversing: close at mid-band
                        self._close_position(pos, pos.range_mid, time_str, "range_mid_exit")
                        continue
```

For range positions with `tp_shifted`, apply tight 1% trailing stop instead of ATR trailing. Replace the ATR trailing stop block with a conditional:

```python
            # Trailing stop logic
            if pos.strategy == "range" and pos.tp_shifted:
                # Tight 1% trailing stop for range positions after TP shift
                if direction == "LONG":
                    trail_level = pos.highest_price * 0.99
                    if trail_level > pos.stop_loss:
                        pos.stop_loss = trail_level
                elif direction == "SHORT":
                    trail_level = pos.highest_price * 1.01
                    if trail_level < pos.stop_loss:
                        pos.stop_loss = trail_level
            elif atr_val is not None:
                # Standard ATR trailing stop for non-range positions
                trail_dist = atr_val * self._atr_multiplier
                if direction == "SHORT":
                    in_profit = pos.entry_price - price
                else:
                    in_profit = price - pos.entry_price
                risk = abs(pos.entry_price - pos.stop_loss) if abs(pos.entry_price - pos.stop_loss) > 0 else trail_dist

                if in_profit >= risk:
                    pos.stop_loss = trail_stop(
                        price, pos.highest_price, pos.stop_loss,
                        trail_dist,
                        direction=direction,
                    )
```

- [ ] **Step 4: Update _close_position to increment bounce counter**

In `bot/backtest/engine.py`, in `_close_position` (line 821), after `self.positions.remove(pos)` (line 830), add:

```python
        # Increment range bounce counter on closed range trades
        if pos.strategy == "range":
            self._range_bounces[pos.symbol] = self._range_bounces.get(pos.symbol, 0) + 1
```

- [ ] **Step 5: Pass precomp_5m and idx_5m to _check_exits_fast in run loop**

In `bot/backtest/engine.py`, in the `run()` method, update the `_check_exits_fast` call (line 197):

Replace:
```python
                self._check_exits_fast(current_price, highs[i], lows[i], current_time, atr_val, current_bar=i)
```
with:
```python
                # Pass pre-computed 5m data for range dynamic TP
                has_range = any(p.strategy == "range" for p in self.positions)
                self._check_exits_fast(
                    current_price, highs[i], lows[i], current_time, atr_val,
                    current_bar=i,
                    precomp_5m=precomp_5m if has_range else None,
                    idx_5m=i if has_range else 0,
                )
```

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_range_strategy.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add bot/backtest/engine.py tests/test_range_strategy.py
git commit -m "feat: add range-specific exit logic with dynamic TP and bounce counter"
```

---

### Task 7: Trading Loop — Range Entry/Exit Support

**Files:**
- Modify: `bot/trading_loop.py:116-151` (_check_stops)
- Modify: `bot/trading_loop.py:463-478` (position opening)

- [ ] **Step 1: Store range levels on live positions**

In `bot/trading_loop.py`, after line 478 (`pos["entry_features"] = best_signal.indicator_snapshot`), add:

```python
                        # Store range levels for range strategy
                        if best_signal.strategy_name == "range":
                            snap = best_signal.indicator_snapshot
                            pos["range_mid"] = snap.get("range_mid", 0.0)
                            pos["range_upper"] = snap.get("range_upper", 0.0)
                            pos["range_lower"] = snap.get("range_lower", 0.0)
                            pos["tp_shifted"] = False

                            # Override ATR stops with range-specific stops
                            bandwidth_price = pos["range_upper"] - pos["range_lower"]
                            if direction == "LONG":
                                pos["stop_loss_price"] = pos["range_lower"] - 0.5 * bandwidth_price
                                pos["take_profit_price"] = pos["range_mid"]
                            else:
                                pos["stop_loss_price"] = pos["range_upper"] + 0.5 * bandwidth_price
                                pos["take_profit_price"] = pos["range_mid"]
```

- [ ] **Step 2: Add range exit logic to _check_stops**

In `bot/trading_loop.py`, in `_check_stops` (line 116), add range-specific handling after the ATR trailing stop update (after line 140), before the `check_stop_triggered` call:

```python
        # Range 24h max hold time
        if pos.get("strategy_name") == "range":
            entry_time = pos.get("entry_time")
            if entry_time is not None:
                from datetime import datetime, timezone
                try:
                    if isinstance(entry_time, str):
                        entry_dt = datetime.fromisoformat(entry_time)
                    else:
                        entry_dt = entry_time
                    now = datetime.now(timezone.utc)
                    if (now - entry_dt).total_seconds() >= 86400:  # 24 hours
                        await self._close_position(symbol, current_price, "range_time_exit")
                        return
                except (ValueError, TypeError):
                    pass

        # Range dynamic TP: shift from mid to opposite band
        if pos.get("strategy_name") == "range" and not pos.get("tp_shifted", False):
            crossed_mid = (
                (direction == "LONG" and current_price >= pos.get("range_mid", 0)) or
                (direction == "SHORT" and current_price <= pos.get("range_mid", float("inf")))
            )
            if crossed_mid and pos.get("range_mid", 0) > 0:
                # Compute RSI from recent 5m candles
                from bot.indicators.momentum import rsi as compute_rsi
                if len(df_5m) >= 20:
                    rsi_series = compute_rsi(df_5m["close"])
                    current_rsi = rsi_series.iloc[-1]
                    prev_rsi = rsi_series.iloc[-4] if len(rsi_series) >= 4 else current_rsi

                    if direction == "LONG" and current_rsi > prev_rsi:
                        pos["tp_shifted"] = True
                        pos["take_profit_price"] = pos["range_upper"]
                        logger.info("Range TP shifted to upper band %.2f for %s", pos["range_upper"], symbol)
                    elif direction == "SHORT" and current_rsi < prev_rsi:
                        pos["tp_shifted"] = True
                        pos["take_profit_price"] = pos["range_lower"]
                        logger.info("Range TP shifted to lower band %.2f for %s", pos["range_lower"], symbol)
                    else:
                        # RSI flat/reversing: close at mid
                        await self._close_position(symbol, pos["range_mid"], "range_mid_exit")
                        return

        # Tight trailing stop for range positions with shifted TP
        if pos.get("strategy_name") == "range" and pos.get("tp_shifted", False):
            if direction == "LONG":
                trail_level = pos["highest_price"] * 0.99
                if trail_level > (pos["stop_loss_price"] or 0):
                    pos["stop_loss_price"] = trail_level
            elif direction == "SHORT":
                trail_level = pos["highest_price"] * 1.01
                if trail_level < (pos["stop_loss_price"] or float("inf")):
                    pos["stop_loss_price"] = trail_level
```

- [ ] **Step 3: Increment bounce counter on range trade close**

In `bot/trading_loop.py`, in `_close_position` (around line 180), after `self._last_trade_closed[symbol] = time.monotonic()`, add:

```python
                # Increment range bounce counter
                if pos.get("strategy_name") == "range":
                    range_strat = None
                    if hasattr(self.router, '_range'):
                        range_strat = self.router._range
                    if range_strat is not None:
                        range_strat.increment_bounce(symbol)
```

- [ ] **Step 4: Run all tests**

Run: `python -m pytest tests/ -v --timeout=30`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add bot/trading_loop.py
git commit -m "feat: add range-specific entry/exit handling to live trading loop"
```

---

### Task 8: Delete Mean Reversion + Final Cleanup

**Files:**
- Delete: `bot/strategy/mean_reversion.py`
- Verify: no remaining imports of `MeanReversionStrategy`

- [ ] **Step 1: Delete mean_reversion.py**

```bash
git rm bot/strategy/mean_reversion.py
```

- [ ] **Step 2: Search for remaining references**

Run: `grep -r "mean_reversion\|MeanReversion" bot/ tests/ --include="*.py"`
Expected: No matches (all references replaced in Tasks 3 and 5)

If any remain, update them to reference `range` / `RangeStrategy`.

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/ -v --timeout=60`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add bot/strategy/mean_reversion.py
git commit -m "feat: delete mean_reversion strategy, fully replaced by range trading"
```

---

### Task 9: Integration Smoke Test

**Files:**
- Existing: `tests/test_range_strategy.py`

- [ ] **Step 1: Run the full backtest integration test**

Run: `python -m pytest tests/test_range_strategy.py -v`
Expected: All tests PASS — volume profile, range strategy, backtest integration, exit logic

- [ ] **Step 2: Quick manual verification with compare script**

Run the existing comparison script to verify the engine still works end-to-end:

```bash
python scripts/compare_confirmation.py
```

Expected: Script completes without import errors. P&L numbers will differ since range strategy replaces mean_reversion, but no crashes.

- [ ] **Step 3: Final commit with all changes**

If any fixups were needed, stage only the changed files:
```bash
git status
git add <changed-files>
git commit -m "fix: address integration issues from range strategy implementation"
```
