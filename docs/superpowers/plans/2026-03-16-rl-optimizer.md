# RL-Based Walk-Forward Parameter Optimizer — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Optuna with PPO reinforcement learning for walk-forward parameter optimization — one model per strategy, 21 parameters, 27 market features.

**Architecture:** CandleStore downloads 1m candles from Bitvavo into PostgreSQL. Feature extractor computes 27 normalized market features. Gymnasium environment wraps BacktestEngine for PPO training. RLOptimizer serves trained models for live walk-forward inference. BacktestEngine gains 13 new tunable parameters replacing hardcoded limits.

**Tech Stack:** stable-baselines3 (PPO), gymnasium, PostgreSQL, pandas, numpy, SQLAlchemy 2.0

**Spec:** `docs/superpowers/specs/2026-03-16-rl-optimizer-design.md`

---

## File Structure

### New Files
| File | Responsibility |
|------|---------------|
| `bot/data/candle_store.py` | Bulk 1m candle download (async) + sync read for training |
| `bot/learning/feature_extractor.py` | 27 normalized market features from 1m candles |
| `bot/learning/rl_environment.py` | Gymnasium env: observation → action → backtest → reward |
| `bot/learning/rl_trainer.py` | PPO training loop, validation, model save/load |
| `bot/learning/rl_optimizer.py` | Inference: load model, predict params from features |
| `bot/scripts/rl_bootstrap.py` | One-time bootstrap: download candles + train models |
| `tests/test_candle_store.py` | CandleStore unit tests |
| `tests/test_feature_extractor.py` | Feature extraction unit tests |
| `tests/test_rl_environment.py` | RL environment unit tests |
| `tests/test_rl_trainer.py` | RLTrainer validation gate and init tests |
| `tests/test_rl_optimizer.py` | RLOptimizer + action mapping tests |
| `tests/test_backtest_rl_params.py` | New engine parameters affect behavior |
| `tests/test_rl_integration.py` | End-to-end: train tiny model → predict → walk-forward |

### Modified Files
| File | Change |
|------|--------|
| `bot/data/models.py` | Add `Candle1m` SQLAlchemy model |
| `bot/backtest/engine.py` | Extract 13 hardcoded values into `strategy_params` |
| `bot/learning/walk_forward.py` | Replace `_run_optuna_window` with `_run_rl_window` method |
| `bot/scheduler.py` | Add `rl_retrain_run` callback |
| `bot/main.py` | Wire CandleStore, RLOptimizer, retrain scheduler |
| `requirements.txt` | Add stable-baselines3, gymnasium; remove optuna |

---

## Chunk 1: Foundation — Database Model, CandleStore, Dependencies

### Task 1: Add dependencies to requirements.txt

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add stable-baselines3 and gymnasium, remove optuna**

```python
# In requirements.txt, under "# Machine learning" section:
# REMOVE this line:
optuna>=3.6.0

# ADD these lines:
stable-baselines3>=2.3.0
gymnasium>=0.29.0
psycopg2-binary>=2.9.0
```

- [ ] **Step 2: Install dependencies**

Run: `pip install stable-baselines3>=2.3.0 gymnasium>=0.29.0 psycopg2-binary>=2.9.0`
Expected: Successfully installed

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "deps: replace optuna with stable-baselines3 and gymnasium"
```

---

### Task 2: Add Candle1m database model

**Files:**
- Modify: `bot/data/models.py`
- Test: `tests/test_candle_store.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_candle_store.py`:

```python
"""Tests for CandleStore and Candle1m model."""
from bot.data.models import Candle1m


def test_candle_1m_model_exists():
    """Candle1m model has correct table name and columns."""
    assert Candle1m.__tablename__ == "candle_1m"
    cols = {c.name for c in Candle1m.__table__.columns}
    assert cols == {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
    # Verify composite primary key
    pk_cols = {c.name for c in Candle1m.__table__.primary_key.columns}
    assert pk_cols == {"symbol", "timestamp"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_candle_store.py::test_candle_1m_model_exists -v`
Expected: FAIL with "cannot import name 'Candle1m'"

- [ ] **Step 3: Add Candle1m model to models.py**

Add at the end of `bot/data/models.py`:

```python
class Candle1m(Base):
    """1-minute candle data for RL training. Bulk-downloaded from Bitvavo."""
    __tablename__ = "candle_1m"
    __table_args__ = (
        Index("ix_candle_1m_symbol_ts", "symbol", "timestamp"),
    )

    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, primary_key=True)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_candle_store.py::test_candle_1m_model_exists -v`
Expected: PASS

- [ ] **Step 5: Generate Alembic migration**

Run: `alembic revision --autogenerate -m "add candle_1m table"`
Expected: New migration file created in `alembic/versions/`

- [ ] **Step 6: Verify migration**

Open the generated migration file and confirm it creates `candle_1m` table with all 7 columns and composite PK on `(symbol, timestamp)`. Fix if auto-generation missed anything.

- [ ] **Step 7: Commit**

```bash
git add bot/data/models.py alembic/versions/ tests/test_candle_store.py
git commit -m "feat: add Candle1m database model with Alembic migration"
```

---

### Task 3: Implement CandleStore

**Files:**
- Create: `bot/data/candle_store.py`
- Modify: `tests/test_candle_store.py`

- [ ] **Step 1: Write failing tests for CandleStore**

Append to `tests/test_candle_store.py`:

```python
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch

from bot.data.candle_store import CandleStore


def test_candle_store_init():
    """CandleStore accepts a database URL."""
    store = CandleStore(db_url="postgresql://test:test@localhost/test")
    assert store._db_url == "postgresql://test:test@localhost/test"


def test_get_candles_resamples_to_5m(tmp_path):
    """get_candles returns 5m resampled data from stored 1m candles."""
    store = CandleStore(db_url="sqlite:///test.db")
    # Create fake 1m data: 60 minutes = 60 rows
    timestamps = pd.date_range("2025-01-01", periods=60, freq="1min", tz="UTC")
    prices = np.linspace(100, 105, 60)
    df_1m = pd.DataFrame({
        "open": prices, "high": prices + 1, "low": prices - 1,
        "close": prices + 0.5, "volume": np.ones(60) * 100,
    }, index=timestamps)

    # Mock the DB read to return our 1m data
    store._read_candles_sync = MagicMock(return_value=df_1m)

    result = store.get_candles("BTC-EUR",
                               datetime(2025, 1, 1, tzinfo=timezone.utc),
                               datetime(2025, 1, 1, 1, tzinfo=timezone.utc),
                               resample="5m")
    assert len(result) == 12  # 60 min / 5 = 12 bars
    assert list(result.columns) == ["open", "high", "low", "close", "volume"]


def test_get_candles_returns_1m_when_requested():
    """get_candles returns raw 1m data when resample='1m'."""
    store = CandleStore(db_url="sqlite:///test.db")
    timestamps = pd.date_range("2025-01-01", periods=60, freq="1min", tz="UTC")
    prices = np.linspace(100, 105, 60)
    df_1m = pd.DataFrame({
        "open": prices, "high": prices + 1, "low": prices - 1,
        "close": prices + 0.5, "volume": np.ones(60) * 100,
    }, index=timestamps)
    store._read_candles_sync = MagicMock(return_value=df_1m)

    result = store.get_candles("BTC-EUR",
                               datetime(2025, 1, 1, tzinfo=timezone.utc),
                               datetime(2025, 1, 1, 1, tzinfo=timezone.utc),
                               resample="1m")
    assert len(result) == 60


@pytest.mark.asyncio
async def test_bulk_download_skips_short_history():
    """bulk_download skips symbols with <90 days of history."""
    store = CandleStore(db_url="postgresql://test:test@localhost/test")
    store._get_last_timestamp = MagicMock(return_value=None)
    store._get_data_span_days = MagicMock(return_value=30)  # only 30 days
    store._download_symbol = AsyncMock()
    await store.bulk_download(["SHORT-EUR"])
    store._download_symbol.assert_not_called()


@pytest.mark.asyncio
async def test_bulk_download_resumes_from_last_ts():
    """bulk_download resumes from last stored timestamp."""
    store = CandleStore(db_url="postgresql://test:test@localhost/test")
    last = datetime(2025, 6, 1, tzinfo=timezone.utc)
    store._get_last_timestamp = MagicMock(return_value=last)
    store._get_data_span_days = MagicMock(return_value=180)
    store._download_symbol = AsyncMock()
    await store.bulk_download(["BTC-EUR"])
    store._download_symbol.assert_called_once_with("BTC-EUR")
```

**Note:** Add `import pytest` at the top of the test file alongside the other imports.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_candle_store.py -v -k "candle_store"`
Expected: FAIL with "cannot import name 'CandleStore'"

- [ ] **Step 3: Implement CandleStore**

Create `bot/data/candle_store.py`:

```python
"""CandleStore — bulk 1m candle download and storage for RL training."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)


class CandleStore:
    """Two access modes:
    - async methods for I/O-bound download (Bitvavo API)
    - sync methods for CPU-bound training (SubprocVecEnv workers)
    Both use the same PostgreSQL candle_1m table.
    """

    def __init__(self, db_url: str) -> None:
        self._db_url = db_url
        # Convert async URL to sync for psycopg2
        self._sync_url = db_url.replace("+asyncpg", "").replace("postgresql+psycopg2", "postgresql")
        if self._sync_url.startswith("postgresql+"):
            self._sync_url = "postgresql" + self._sync_url[self._sync_url.index("://"):]

    # ------------------------------------------------------------------
    # Async download (called from scheduler / bootstrap)
    # ------------------------------------------------------------------

    async def bulk_download(self, symbols: List[str],
                            min_days: int = 90) -> None:
        """Pull all available 1m candles from Bitvavo for each symbol.
        Resumes from last stored timestamp per symbol.
        Skips symbols with <min_days of history (too new for training).
        """
        import asyncio
        for symbol in symbols:
            try:
                span = self._get_data_span_days(symbol)
                if span is not None and span < min_days:
                    logger.info("CandleStore: skipping %s (only %d days of data)", symbol, span)
                    continue
                await self._download_symbol(symbol)
            except Exception as e:
                logger.error("CandleStore: failed to download %s: %s", symbol, e)

    async def incremental_update(self, symbols: List[str]) -> None:
        """Pull new 1m candles since last download."""
        await self.bulk_download(symbols)  # same logic, resumes from last ts

    async def _download_symbol(self, symbol: str) -> None:
        """Download 1m candles for one symbol from Bitvavo."""
        import asyncio
        from bot.exchange.bitvavo_client import get_client

        loop = asyncio.get_running_loop()
        client = get_client()

        # Find last stored timestamp
        last_ts = self._get_last_timestamp(symbol)
        start_ms = int(last_ts.timestamp() * 1000) if last_ts else 0

        total_stored = 0
        retries = 0
        max_retries = 3

        while True:
            try:
                candles = await loop.run_in_executor(
                    None,
                    lambda: client.bitvavo.candles(symbol.replace("-", ""), "1m",
                                                    {"start": start_ms, "limit": 1440}),
                )
            except Exception as e:
                retries += 1
                if retries > max_retries:
                    logger.warning("CandleStore: giving up on %s after %d retries: %s",
                                   symbol, max_retries, e)
                    break
                wait = min(2 ** retries, 30)
                logger.info("CandleStore: rate limited on %s, waiting %ds", symbol, wait)
                await asyncio.sleep(wait)
                continue

            retries = 0
            if not candles:
                break

            self._store_candles(symbol, candles)
            total_stored += len(candles)

            # Move start forward
            last_candle_ts = max(c[0] for c in candles)
            start_ms = last_candle_ts + 60_000  # next minute

            # Rate limit: small delay between requests
            await asyncio.sleep(0.1)

        if total_stored > 0:
            logger.info("CandleStore: stored %d candles for %s", total_stored, symbol)

    def _get_last_timestamp(self, symbol: str) -> Optional[datetime]:
        """Get last stored timestamp for a symbol (sync)."""
        conn = psycopg2.connect(self._sync_url)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT MAX(timestamp) FROM candle_1m WHERE symbol = %s",
                    (symbol,),
                )
                row = cur.fetchone()
                return row[0] if row and row[0] else None
        finally:
            conn.close()

    def _get_data_span_days(self, symbol: str) -> Optional[int]:
        """Return number of days of stored data for a symbol, or None if no data."""
        date_range = self.get_date_range(symbol)
        if date_range is None:
            return None  # no data yet, allow download
        return (date_range[1] - date_range[0]).days

    def _store_candles(self, symbol: str, candles: list) -> None:
        """Bulk insert candles into candle_1m table (sync)."""
        conn = psycopg2.connect(self._sync_url)
        try:
            with conn.cursor() as cur:
                values = []
                for c in candles:
                    # Bitvavo returns: [timestamp_ms, open, high, low, close, volume]
                    ts = datetime.fromtimestamp(c[0] / 1000.0, tz=timezone.utc)
                    values.append((symbol, ts, float(c[1]), float(c[2]),
                                   float(c[3]), float(c[4]), float(c[5])))
                psycopg2.extras.execute_values(
                    cur,
                    """INSERT INTO candle_1m (symbol, timestamp, open, high, low, close, volume)
                       VALUES %s ON CONFLICT (symbol, timestamp) DO NOTHING""",
                    values,
                )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Sync read (called from training workers / feature extraction)
    # ------------------------------------------------------------------

    def get_candles(self, symbol: str, start: datetime, end: datetime,
                    resample: str = "5m") -> pd.DataFrame:
        """Fetch candles for a date range, optionally resampled.
        SYNC — safe to call from SubprocVecEnv workers.
        """
        df = self._read_candles_sync(symbol, start, end)
        if df.empty:
            return df

        if resample == "1m":
            return df

        return df.resample(resample).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

    def _read_candles_sync(self, symbol: str, start: datetime,
                           end: datetime) -> pd.DataFrame:
        """Read 1m candles from DB (sync psycopg2 connection)."""
        conn = psycopg2.connect(self._sync_url)
        try:
            df = pd.read_sql(
                "SELECT timestamp, open, high, low, close, volume "
                "FROM candle_1m WHERE symbol = %s AND timestamp >= %s AND timestamp < %s "
                "ORDER BY timestamp",
                conn,
                params=(symbol, start, end),
            )
            if df.empty:
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")
            return df
        finally:
            conn.close()

    def get_date_range(self, symbol: str) -> Optional[tuple]:
        """Return (min_ts, max_ts) for a symbol, or None if no data."""
        conn = psycopg2.connect(self._sync_url)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT MIN(timestamp), MAX(timestamp) FROM candle_1m WHERE symbol = %s",
                    (symbol,),
                )
                row = cur.fetchone()
                if row and row[0]:
                    return (row[0], row[1])
                return None
        finally:
            conn.close()

    def available_symbols(self) -> List[str]:
        """Return list of symbols with stored data (sync)."""
        conn = psycopg2.connect(self._sync_url)
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT symbol FROM candle_1m ORDER BY symbol")
                return [row[0] for row in cur.fetchall()]
        finally:
            conn.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_candle_store.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add bot/data/candle_store.py tests/test_candle_store.py
git commit -m "feat: implement CandleStore with async download and sync read"
```

---

## Chunk 2: Feature Extractor

### Task 4: Implement feature extractor with 27 normalized features

**Files:**
- Create: `bot/learning/feature_extractor.py`
- Create: `tests/test_feature_extractor.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_feature_extractor.py`:

```python
"""Tests for the 27-feature market feature extractor."""
import numpy as np
import pandas as pd
from bot.learning.feature_extractor import extract_features


def _make_candles_1m(days: int = 90, base_price: float = 100.0) -> pd.DataFrame:
    """Generate synthetic 1m candle data for testing."""
    n = days * 24 * 60  # 1m candles
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    np.random.seed(42)
    # Random walk with slight uptrend
    returns = np.random.normal(0.00001, 0.001, n)
    prices = base_price * np.exp(np.cumsum(returns))
    high = prices * (1 + np.abs(np.random.normal(0, 0.002, n)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.002, n)))
    volume = np.random.exponential(1000, n)

    return pd.DataFrame({
        "open": prices,
        "high": high,
        "low": low,
        "close": prices * (1 + np.random.normal(0, 0.0005, n)),
        "volume": volume,
    }, index=timestamps)


def test_extract_features_returns_27_floats():
    """extract_features returns a float32 array of shape (27,)."""
    df = _make_candles_1m()
    features = extract_features(df)
    assert features.shape == (27,)
    assert features.dtype == np.float32


def test_features_clipped_to_bounds():
    """All features are within [-1, 1]."""
    df = _make_candles_1m()
    features = extract_features(df)
    assert np.all(features >= -1.0), f"Min: {features.min()}"
    assert np.all(features <= 1.0), f"Max: {features.max()}"


def test_features_with_btc_candles():
    """BTC correlation features (25, 26) are computed when btc_candles provided."""
    df = _make_candles_1m()
    btc = _make_candles_1m(base_price=50000.0)
    features = extract_features(df, btc_candles_1m=btc)
    assert features.shape == (27,)
    # Features 25, 26 (0-indexed) should not be zero when BTC data is provided
    assert features[25] != 0.0 or features[26] != 0.0


def test_features_without_btc_candles():
    """BTC features default to 0.0 when no BTC data provided."""
    df = _make_candles_1m()
    features = extract_features(df, btc_candles_1m=None)
    assert features[25] == 0.0  # BTC correlation
    assert features[26] == 0.0  # BTC trend


def test_features_short_data_returns_zeros():
    """Insufficient data (< 1440 rows) returns zeros array."""
    n = 500  # well below 1440 threshold
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    prices = np.linspace(100, 105, n)
    df = pd.DataFrame({
        "open": prices, "high": prices + 1, "low": prices - 1,
        "close": prices + 0.5, "volume": np.ones(n) * 100,
    }, index=timestamps)
    features = extract_features(df)
    assert features.shape == (27,)
    assert np.all(features == 0.0)  # all zeros for insufficient data


def test_features_no_nan_or_inf():
    """Features never contain NaN or inf, even with edge-case data."""
    # Create data with zero volume and flat prices (can cause div-by-zero)
    n = 1440 * 3
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    price = 100.0
    df = pd.DataFrame({
        "open": np.full(n, price), "high": np.full(n, price),
        "low": np.full(n, price), "close": np.full(n, price),
        "volume": np.zeros(n),
    }, index=timestamps)
    features = extract_features(df)
    assert features.shape == (27,)
    assert not np.any(np.isnan(features)), f"NaN found: {features}"
    assert not np.any(np.isinf(features)), f"Inf found: {features}"
    assert np.all(features >= -1.0) and np.all(features <= 1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_feature_extractor.py -v`
Expected: FAIL with "cannot import name 'extract_features'"

- [ ] **Step 3: Implement feature extractor**

Create `bot/learning/feature_extractor.py`:

```python
"""27-feature market feature extractor for RL training and inference."""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from bot.indicators.momentum import rsi
from bot.indicators.trend import adx, ema, macd
from bot.indicators.volatility import atr, bollinger_bands

logger = logging.getLogger(__name__)

N_FEATURES = 27


def extract_features(
    candles_1m: pd.DataFrame,
    btc_candles_1m: pd.DataFrame = None,
) -> np.ndarray:
    """Compute 27 normalized features from a window of 1m candles.

    Resamples 1m to daily/4h/1h internally. All values hard-clipped to [-1, 1].
    Returns float32 array of shape (27,).
    """
    features = np.zeros(N_FEATURES, dtype=np.float32)

    if candles_1m is None or len(candles_1m) < 1440:  # need at least 1 day
        return features

    try:
        close_1m = candles_1m["close"]
        high_1m = candles_1m["high"]
        low_1m = candles_1m["low"]
        vol_1m = candles_1m["volume"]

        # Resample to higher timeframes
        df_1h = candles_1m.resample("1h").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()
        df_4h = candles_1m.resample("4h").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()
        df_1d = candles_1m.resample("1D").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()

        if len(df_1d) < 5 or len(df_1h) < 30:
            return features

        price = float(close_1m.iloc[-1])
        if price <= 0:
            return features

        # --- Trend features (0-3) ---
        ema200_d = ema(df_1d["close"], 200)
        ema50_d = ema(df_1d["close"], 50)

        if len(ema200_d.dropna()) > 0:
            features[0] = ((price - float(ema200_d.iloc[-1])) / float(ema200_d.iloc[-1]) * 100.0) / 20.0

        if len(ema50_d.dropna()) > 0:
            features[1] = ((price - float(ema50_d.iloc[-1])) / float(ema50_d.iloc[-1]) * 100.0) / 10.0

        if len(ema50_d.dropna()) >= 10:
            slope = float(ema50_d.iloc[-1]) - float(ema50_d.iloc[-10])
            features[2] = slope / price

        period_return = (price / float(close_1m.iloc[0]) - 1.0) * 100.0
        features[3] = max(min(period_return, 50.0), -50.0) / 50.0

        # --- Volatility features (4-7) ---
        atr_4h = atr(df_4h["high"], df_4h["low"], df_4h["close"])
        if len(atr_4h.dropna()) > 0:
            close_4h = df_4h["close"]
            atr_pct = (atr_4h / close_4h * 100.0).dropna()
            if len(atr_pct) > 0:
                features[4] = float(atr_pct.mean()) / 10.0
                features[5] = float(atr_pct.std()) / 5.0 if len(atr_pct) > 1 else 0.0
                avg = float(atr_pct.mean())
                if avg > 0:
                    features[6] = min(float(atr_pct.iloc[-1]) / avg, 3.0) / 3.0
                features[7] = float(atr_pct.max()) / 20.0

        # --- Regime features (8-11) ---
        adx_4h = adx(df_4h["high"], df_4h["low"], df_4h["close"])
        if len(adx_4h.dropna()) > 0:
            adx_vals = adx_4h.dropna()
            features[8] = float((adx_vals > 25).mean())   # % trending
            features[9] = float((adx_vals < 20).mean())   # % ranging
            features[10] = float(adx_vals.iloc[-1]) / 50.0

        rsi_1h = rsi(df_1h["close"])
        if len(rsi_1h.dropna()) > 0:
            features[11] = float(rsi_1h.iloc[-1]) / 100.0

        # --- Volume features (12-13) ---
        if len(vol_1m) > 100:
            half = len(vol_1m) // 2
            first_half_vol = float(vol_1m.iloc[:half].mean())
            second_half_vol = float(vol_1m.iloc[half:].mean())
            if first_half_vol > 0:
                vol_trend = (second_half_vol / first_half_vol - 1.0) * 100.0
                features[12] = max(min(vol_trend, 100.0), -100.0) / 100.0

            avg_vol = float(vol_1m.mean())
            if avg_vol > 0:
                features[13] = min(float(vol_1m.iloc[-1]) / avg_vol, 5.0) / 5.0

        # --- Price structure (14-16) ---
        period_high = float(high_1m.max())
        period_low = float(low_1m.min())
        if period_high > 0:
            features[14] = (price - period_high) / period_high * 100.0 / -50.0
        if period_low > 0:
            features[15] = (price - period_low) / period_low * 100.0 / 50.0

        # Count significant reversals (>5% moves)
        daily_returns = df_1d["close"].pct_change().dropna() * 100.0
        reversals = 0
        prev_dir = 0
        cumulative = 0.0
        for ret in daily_returns:
            if prev_dir == 0:
                prev_dir = 1 if ret > 0 else -1
                cumulative = ret
            elif (ret > 0 and prev_dir > 0) or (ret < 0 and prev_dir < 0):
                cumulative += ret
            else:
                if abs(cumulative) > 5.0:
                    reversals += 1
                prev_dir = 1 if ret > 0 else -1
                cumulative = ret
        features[16] = min(reversals, 20) / 20.0

        # --- Bollinger Band features (17-18) ---
        bb = bollinger_bands(df_1h["close"])
        if "bandwidth" in bb and len(bb["bandwidth"].dropna()) > 0:
            features[17] = min(float(bb["bandwidth"].iloc[-1]) / 0.2, 1.0)
        if "upper" in bb and "lower" in bb:
            upper = float(bb["upper"].iloc[-1])
            lower = float(bb["lower"].iloc[-1])
            if upper > lower:
                pct_b = (price - lower) / (upper - lower)
                features[18] = max(min(pct_b, 1.0), 0.0)

        # --- Momentum features (19-20) ---
        rsi_vals = rsi(df_1h["close"])
        if len(rsi_vals.dropna()) >= 14:
            rsi_roc = float(rsi_vals.iloc[-1]) - float(rsi_vals.iloc[-14])
            features[19] = rsi_roc / 50.0

        macd_data = macd(df_1h["close"])
        if "histogram" in macd_data and len(macd_data["histogram"].dropna()) > 0:
            hist = float(macd_data["histogram"].iloc[-1])
            features[20] = 1.0 if hist > 0 else -1.0

        # --- Candle structure (21-22) ---
        if len(candles_1m) > 100:
            sample = candles_1m.tail(1440)  # last day of 1m candles
            bodies = (sample["close"] - sample["open"]).abs()
            upper_wicks = sample["high"] - sample[["open", "close"]].max(axis=1)
            lower_wicks = sample[["open", "close"]].min(axis=1) - sample["low"]
            total_wicks = upper_wicks + lower_wicks
            valid = total_wicks > 0
            if valid.sum() > 0:
                ratios = bodies[valid] / total_wicks[valid]
                features[21] = min(float(ratios.mean()), 2.0) / 2.0

            longer_lower = (lower_wicks > upper_wicks).mean()
            longer_upper = (upper_wicks > lower_wicks).mean()
            features[22] = float(longer_lower - longer_upper)

        # --- Time features (23-24) ---
        last_ts = candles_1m.index[-1]
        features[23] = math.sin(2 * math.pi * last_ts.dayofweek / 7.0)
        features[24] = math.sin(2 * math.pi * last_ts.hour / 24.0)

        # --- BTC correlation (25-26) ---
        if btc_candles_1m is not None and len(btc_candles_1m) > 0:
            try:
                # Resample both to daily for correlation
                target_daily = close_1m.resample("1D").last().dropna()
                btc_daily = btc_candles_1m["close"].resample("1D").last().dropna()
                # Align and compute correlation
                aligned = pd.concat([target_daily, btc_daily], axis=1, join="inner")
                if len(aligned) >= 30:
                    corr = aligned.iloc[:, 0].rolling(30).corr(aligned.iloc[:, 1])
                    if len(corr.dropna()) > 0:
                        features[25] = float(corr.iloc[-1])

                btc_ema50 = ema(btc_candles_1m["close"].resample("1D").last().dropna(), 50)
                if len(btc_ema50.dropna()) >= 10:
                    btc_price = float(btc_candles_1m["close"].iloc[-1])
                    if btc_price > 0:
                        btc_slope = float(btc_ema50.iloc[-1]) - float(btc_ema50.iloc[-10])
                        features[26] = btc_slope / btc_price
            except Exception:
                pass  # BTC features stay 0.0

    except Exception as e:
        logger.warning("Feature extraction failed: %s", e)
        return np.zeros(N_FEATURES, dtype=np.float32)

    # Replace NaN/inf with 0.0, then hard clip to [-1, 1]
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(features, -1.0, 1.0).astype(np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_feature_extractor.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add bot/learning/feature_extractor.py tests/test_feature_extractor.py
git commit -m "feat: implement 27-feature market feature extractor"
```

---

## Chunk 3: BacktestEngine New Parameters

### Task 5: Add 13 new tunable parameters to BacktestEngine

**Files:**
- Modify: `bot/backtest/engine.py:132-155` (param extraction in `__init__`)
- Modify: `bot/backtest/engine.py:282` (volatile_atr_threshold)
- Modify: `bot/backtest/engine.py:338-341` (ema200_filter_pct)
- Modify: `bot/backtest/engine.py:347` (signal_strength_min)
- Modify: `bot/backtest/engine.py:655` (confluence_boost)
- Modify: `bot/backtest/engine.py:670` (tf_weights — after confluence gate)
- Modify: `bot/backtest/engine.py:707` (drawdown_scale_pct)
- Modify: `bot/backtest/engine.py:725` (max_position_pct)
- Modify: `bot/backtest/engine.py:857` (trail_activation_mult)
- Create: `tests/test_backtest_rl_params.py`

- [ ] **Step 1: Write failing tests for new parameters**

Create `tests/test_backtest_rl_params.py`:

```python
"""Tests that new RL-tunable parameters affect BacktestEngine behavior."""
from bot.backtest.engine import BacktestEngine


def _make_candles(n=2000, base_price=100.0):
    """Generate simple candle data as list of dicts."""
    import numpy as np
    np.random.seed(42)
    candles = []
    price = base_price
    for i in range(n):
        ret = np.random.normal(0, 0.005)
        price *= (1 + ret)
        candles.append({
            "timestamp": f"2024-01-01T{i // 12:02d}:{(i % 12) * 5:02d}:00Z",
            "open": price * 0.999,
            "high": price * 1.003,
            "low": price * 0.997,
            "close": price,
            "volume": 1000.0,
            "symbol": "TEST-EUR",
        })
    return candles


def test_signal_strength_min_filters_weak_signals():
    """Higher signal_strength_min should reduce number of trades."""
    candles = _make_candles()
    r_low = BacktestEngine(candles, strategy_params={"signal_strength_min": 0.1}).run()
    r_high = BacktestEngine(candles, strategy_params={"signal_strength_min": 0.9}).run()
    assert r_high.total_trades <= r_low.total_trades


def test_max_position_pct_caps_size():
    """Smaller max_position_pct should result in smaller trades."""
    candles = _make_candles()
    r_big = BacktestEngine(candles, strategy_params={"max_position_pct": 0.5}).run()
    r_small = BacktestEngine(candles, strategy_params={"max_position_pct": 0.1}).run()
    if r_big.total_trades > 0 and r_small.total_trades > 0:
        avg_big = sum(t.size_eur for t in r_big.trade_log) / len(r_big.trade_log)
        avg_small = sum(t.size_eur for t in r_small.trade_log) / len(r_small.trade_log)
        assert avg_small <= avg_big


def test_ema200_filter_disabled_at_zero():
    """ema200_filter_pct=0 disables the trend filter (more trades possible)."""
    candles = _make_candles()
    r_strict = BacktestEngine(candles, strategy_params={"ema200_filter_pct": 2.0}).run()
    r_disabled = BacktestEngine(candles, strategy_params={"ema200_filter_pct": 0.0}).run()
    assert r_disabled.total_trades >= r_strict.total_trades


def test_volatile_atr_threshold_affects_regime():
    """Lower volatile_atr_threshold classifies more bars as VOLATILE (fewer trades)."""
    candles = _make_candles()
    r_low = BacktestEngine(candles, strategy_params={"volatile_atr_threshold": 1.0}).run()
    r_high = BacktestEngine(candles, strategy_params={"volatile_atr_threshold": 10.0}).run()
    assert r_high.total_trades >= r_low.total_trades


def test_drawdown_scale_pct_default():
    """drawdown_scale_pct is extractable from params."""
    engine = BacktestEngine([], strategy_params={"drawdown_scale_pct": 5.0})
    assert engine._drawdown_scale_pct == 5.0


def test_trail_activation_mult_default():
    """trail_activation_mult is extractable from params."""
    engine = BacktestEngine([], strategy_params={"trail_activation_mult": 2.5})
    assert engine._trail_activation_mult == 2.5


def test_confluence_boost_default():
    """confluence_boost is extractable from params."""
    engine = BacktestEngine([], strategy_params={"confluence_boost": 1.5})
    assert engine._confluence_boost == 1.5


def test_confidence_size_scaling_default():
    """confidence_size_scaling is extractable from params."""
    engine = BacktestEngine([], strategy_params={"confidence_size_scaling": 1.0})
    assert engine._confidence_size_scaling == 1.0


def test_range_max_hold_hours():
    """range_max_hold_hours overrides hardcoded 72h."""
    engine = BacktestEngine([], strategy_params={"range_max_hold_hours": 24})
    assert engine._range_max_hold_bars == 24 * 12


def test_tf_weights_default():
    """tf_weight params are extractable."""
    engine = BacktestEngine([], strategy_params={
        "tf_weight_1h": 0.5, "tf_weight_4h": 0.8, "tf_weight_1d": 0.3,
    })
    assert engine._tf_weight_1h == 0.5
    assert engine._tf_weight_4h == 0.8
    assert engine._tf_weight_1d == 0.3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_backtest_rl_params.py -v`
Expected: FAIL (attributes don't exist)

- [ ] **Step 3: Add parameter extraction to BacktestEngine.__init__**

In `bot/backtest/engine.py`, after line 150 (`self._consecutive_confirms = ...`), add:

```python
        # RL-tunable parameters (replacing hardcoded values)
        self._signal_strength_min = params.get("signal_strength_min", 0.0)
        self._tf_weight_1h = params.get("tf_weight_1h", 1.0)
        self._tf_weight_4h = params.get("tf_weight_4h", 1.0)
        self._tf_weight_1d = params.get("tf_weight_1d", 1.0)
        self._confidence_size_scaling = params.get("confidence_size_scaling", 0.0)
        self._ema200_filter_pct = params.get("ema200_filter_pct", 2.0)
        self._volatile_atr_threshold = params.get("volatile_atr_threshold", 4.0)
        self._confluence_boost = params.get("confluence_boost", 1.2)
        self._drawdown_scale_pct = params.get("drawdown_scale_pct", 3.0)
        self._max_position_pct = params.get("max_position_pct", 0.30)
        self._trail_activation_mult = params.get("trail_activation_mult", 1.5)
        if "max_concurrent_positions" in params:
            self.max_open = params["max_concurrent_positions"]
```

Replace hardcoded `self._range_max_hold_bars = 864` (line 155) with:

```python
        self._range_max_hold_bars = int(params.get("range_max_hold_hours", 72) * 12)
```

- [ ] **Step 4: Replace hardcoded values in engine methods**

In `run()` method:

**Line 282** — volatile regime detection:
```python
# REPLACE: elif atr_pct_4h > 4.0:
elif atr_pct_4h > self._volatile_atr_threshold:
```

**Lines 337-341** — EMA200 filter. Keep lines 334-336 (ema200_1d and ema200_dist_pct computation) as-is, then REPLACE only the comparison logic at lines 337-341 (the hardcoded `2.0`) with:
```python
                # Lines 334-336 remain unchanged (ema200_1d, ema200_dist_pct computation)
                # REPLACE lines 337-341 (hardcoded 2.0 comparisons) with:
                if self._ema200_filter_pct > 0:
                    if ema200_dist_pct < -self._ema200_filter_pct and best_signal.direction == "LONG":
                        best_signal = None
                    elif ema200_dist_pct > self._ema200_filter_pct and best_signal.direction == "SHORT":
                        best_signal = None
```

**Line 347** — signal strength:
```python
# REPLACE: and best_signal.strength > 0
and best_signal.strength >= self._signal_strength_min
```

In `_evaluate_precomputed()`:

**Line 655** — confluence boost:
```python
# REPLACE: strength=min(confluence.strength * 1.2, 1.0),
strength=min(confluence.strength * self._confluence_boost, 1.0),
```

**After the confluence gate** — add `_apply_tf_weights` helper method and apply it to all 3 return paths in `_evaluate_precomputed`. Create the helper method on `BacktestEngine`:

> **Note:** `min_profit_multiple` is intentionally NOT in this chunk — it is already read from `strategy_params` dynamically at line 384 of the engine.

```python
    def _apply_tf_weights(self, direction, strength, precomp_4h, precomp_1d, h4_idx, h1d_idx):
        """Scale signal strength by timeframe trend agreement."""
        tf_score = self._tf_weight_1h
        if precomp_4h and 0 <= h4_idx < len(precomp_4h.get("ema50", [])):
            lookback = min(3, h4_idx)
            if lookback > 0:
                slope = precomp_4h["ema50"][h4_idx] - precomp_4h["ema50"][h4_idx - lookback]
                if (direction == "LONG" and slope > 0) or (direction == "SHORT" and slope < 0):
                    tf_score += self._tf_weight_4h
        if precomp_1d and 0 <= h1d_idx < len(precomp_1d.get("ema50", [])):
            lookback = min(3, h1d_idx)
            if lookback > 0:
                slope = precomp_1d["ema50"][h1d_idx] - precomp_1d["ema50"][h1d_idx - lookback]
                if (direction == "LONG" and slope > 0) or (direction == "SHORT" and slope < 0):
                    tf_score += self._tf_weight_1d
        total = self._tf_weight_1h + self._tf_weight_4h + self._tf_weight_1d
        if total > 0:
            strength *= tf_score / total
        return strength
```

Then apply `_apply_tf_weights` to each of the 3 return paths explicitly:

**Return path 1 — isolation (around line 647):**
```python
# BEFORE the isolation Signal is created:
strength = self._apply_tf_weights(
    best["direction"], strength, precomp_4h, precomp_1d, h4_idx, h1d_idx
)
```

**Return path 2 — confluence (around line 652):**
```python
# BEFORE the confluence Signal is created:
final_strength = self._apply_tf_weights(
    confluence.direction, min(confluence.strength * self._confluence_boost, 1.0),
    precomp_4h, precomp_1d, h4_idx, h1d_idx
)
# Use final_strength instead of the original strength in the Signal constructor
```

**Return path 3 — best-of (around line 663):**
```python
# BEFORE the best-of Signal is created:
strength = self._apply_tf_weights(
    best["direction"], strength, precomp_4h, precomp_1d, h4_idx, h1d_idx
)
```

In `_compute_position_size()`:

**Line 707** — drawdown scaling:
```python
# REPLACE: if drawdown_pct >= 3.0:
if drawdown_pct >= self._drawdown_scale_pct:
```

In `_open_position()`:

**Line 725** — max position cap:
```python
# REPLACE: max_position = self.initial_capital * 0.30
max_position = self.initial_capital * self._max_position_pct
```

After the position size is computed (before the `if size_eur < 10.0` check), add confidence scaling:
```python
        if self._confidence_size_scaling > 0:
            size_eur *= max(0.2, 1.0 + (signal.strength - 0.5) * self._confidence_size_scaling)
```

In `_check_exits_fast()`:

**Line 857** — trail activation:
```python
# REPLACE: activation_threshold=1.5,
activation_threshold=self._trail_activation_mult,
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_backtest_rl_params.py -v`
Expected: All PASS

- [ ] **Step 6: Run existing backtest tests to verify no regressions**

Run: `pytest tests/test_backtest_engine_overhaul.py tests/test_backtest_target_strategy.py tests/test_backtest_drawdown.py tests/test_backtest_ema.py tests/test_backtest_slippage.py -v`
Expected: All PASS (defaults match old hardcoded values)

- [ ] **Step 7: Commit**

```bash
git add bot/backtest/engine.py tests/test_backtest_rl_params.py
git commit -m "feat: replace 13 hardcoded engine limits with tunable strategy_params"
```

---

## Chunk 4: RL Environment

### Task 6: Implement Gymnasium environment and action-to-parameter mapping

**Files:**
- Create: `bot/learning/rl_environment.py`
- Create: `tests/test_rl_environment.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_rl_environment.py`:

```python
"""Tests for TradingParamEnv — RL environment wrapping BacktestEngine."""
import numpy as np
import math
from unittest.mock import MagicMock, patch
from bot.learning.rl_environment import TradingParamEnv, action_to_params
from bot.backtest.engine import BacktestResult


def test_action_to_params_shape_non_range():
    """action_to_params maps 21-element action to param dict."""
    action = np.zeros(21, dtype=np.float32)
    params = action_to_params(action, strategy="squeeze")
    assert isinstance(params, dict)
    assert "atr_multiplier" in params
    assert "range_max_hold_hours" not in params
    assert len(params) == 21  # 21 params for non-range


def test_action_to_params_shape_range():
    """action_to_params maps 22-element action to param dict with range_max_hold_hours."""
    action = np.zeros(22, dtype=np.float32)
    params = action_to_params(action, strategy="range")
    assert "range_max_hold_hours" in params
    assert len(params) == 22


def test_action_to_params_bounds():
    """Actions at -1 and +1 produce correct parameter bounds."""
    # All -1: should produce minimum values
    action_min = np.full(21, -1.0, dtype=np.float32)
    p_min = action_to_params(action_min, strategy="squeeze")
    assert abs(p_min["atr_multiplier"] - 1.0) < 0.01
    assert abs(p_min["rr_ratio"] - 1.0) < 0.01

    # All +1: should produce maximum values
    action_max = np.full(21, 1.0, dtype=np.float32)
    p_max = action_to_params(action_max, strategy="squeeze")
    assert abs(p_max["atr_multiplier"] - 15.0) < 0.01
    assert abs(p_max["rr_ratio"] - 8.0) < 0.01


def test_action_to_params_integer_params():
    """Integer parameters are rounded correctly."""
    action = np.full(21, 0.0, dtype=np.float32)  # midpoint
    params = action_to_params(action, strategy="orderflow")
    assert isinstance(params["max_hold_hours"], int)
    assert isinstance(params["max_concurrent_positions"], int)
    assert isinstance(params["consecutive_confirms"], int)


def test_compute_reward_basic():
    """Reward computation produces reasonable values."""
    env = TradingParamEnv.__new__(TradingParamEnv)
    result = BacktestResult(
        total_pnl=500.0,
        sharpe_ratio=1.5,
        profit_factor=2.0,
        profit_per_fee=3.0,
        max_drawdown_pct=10.0,
        total_trades=20,
        win_rate=55.0,
    )
    reward = env._compute_reward(result)
    assert reward > 0  # good result should be positive


def test_compute_reward_penalties():
    """Reward applies penalties for bad results."""
    env = TradingParamEnv.__new__(TradingParamEnv)

    # High drawdown penalty
    result_dd = BacktestResult(
        total_pnl=100.0, sharpe_ratio=0.5, profit_factor=1.5,
        profit_per_fee=1.0, max_drawdown_pct=25.0, total_trades=10, win_rate=50.0,
    )
    reward_dd = env._compute_reward(result_dd)

    # Too few trades penalty
    result_few = BacktestResult(
        total_pnl=100.0, sharpe_ratio=0.5, profit_factor=1.5,
        profit_per_fee=1.0, max_drawdown_pct=5.0, total_trades=2, win_rate=50.0,
    )
    reward_few = env._compute_reward(result_few)

    # Low win rate penalty
    result_wr = BacktestResult(
        total_pnl=100.0, sharpe_ratio=0.5, profit_factor=1.5,
        profit_per_fee=1.0, max_drawdown_pct=5.0, total_trades=10, win_rate=15.0,
    )
    reward_wr = env._compute_reward(result_wr)

    # No penalties baseline
    result_ok = BacktestResult(
        total_pnl=100.0, sharpe_ratio=0.5, profit_factor=1.5,
        profit_per_fee=1.0, max_drawdown_pct=5.0, total_trades=10, win_rate=50.0,
    )
    reward_ok = env._compute_reward(result_ok)

    assert reward_dd < reward_ok   # drawdown penalty
    assert reward_few < reward_ok  # too-few-trades penalty
    assert reward_wr < reward_ok   # low win rate penalty


def test_reset_returns_valid_observation():
    """reset() returns observation with correct shape and range."""
    mock_store = MagicMock()
    mock_store.get_date_range.return_value = None
    env = TradingParamEnv(mock_store, ["BTC-EUR"], "squeeze")
    obs, info = env.reset()
    assert obs.shape == (27,)
    assert obs.dtype == np.float32
    assert np.all(obs >= -1.0) and np.all(obs <= 1.0)
    assert isinstance(info, dict)


def test_step_returns_correct_tuple():
    """step() returns (obs, reward, terminated, truncated, info)."""
    mock_store = MagicMock()
    mock_store.get_date_range.return_value = None
    env = TradingParamEnv(mock_store, ["BTC-EUR"], "squeeze")
    env.reset()
    action = env.action_space.sample()
    result = env.step(action)
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert obs.shape == (27,)
    assert isinstance(reward, float)
    assert terminated is True   # single-step episode
    assert truncated is False
    assert isinstance(info, dict)


def test_reward_for_default_backtest_result():
    """Reward for default BacktestResult (0 trades) is negative (penalty)."""
    env = TradingParamEnv.__new__(TradingParamEnv)
    reward = env._compute_reward(BacktestResult())
    assert reward < 0  # penalty for < 3 trades
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rl_environment.py -v`
Expected: FAIL with "cannot import name 'TradingParamEnv'"

- [ ] **Step 3: Implement RL environment**

Create `bot/learning/rl_environment.py`:

```python
"""Gymnasium environment for PPO-based trading parameter optimization."""
from __future__ import annotations

import logging
import math
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces

from bot.backtest.engine import BacktestEngine, BacktestResult
from bot.learning.feature_extractor import extract_features

logger = logging.getLogger(__name__)

# Parameter mapping table: (name, min, max, is_int)
PARAM_TABLE: List[Tuple[str, float, float, bool]] = [
    ("atr_multiplier",          1.0,  15.0, False),
    ("rr_ratio",                1.0,   8.0, False),
    ("base_risk_pct",           1.0,   8.0, False),
    ("min_profit_multiple",     0.5,   8.0, False),
    ("max_hold_hours",          6.0, 720.0, True),
    ("quiet_atr_threshold",     0.1,   5.0, False),
    ("regime_adx_threshold",    5.0,  50.0, False),
    ("ranging_adx_threshold",   5.0,  50.0, False),
    ("signal_strength_min",     0.1,   1.0, False),
    ("tf_weight_1h",            0.0,   1.0, False),
    ("tf_weight_4h",            0.0,   1.0, False),
    ("tf_weight_1d",            0.0,   1.0, False),
    ("max_concurrent_positions", 1.0, 10.0, True),
    ("confidence_size_scaling",  0.0,  2.0, False),
    ("ema200_filter_pct",       0.0,  10.0, False),
    ("volatile_atr_threshold",  1.0,  10.0, False),
    ("consecutive_confirms",    1.0,   5.0, True),
    ("confluence_boost",        1.0,   2.0, False),
    ("drawdown_scale_pct",      1.0,  15.0, False),
    ("max_position_pct",        0.1,   0.5, False),
    ("trail_activation_mult",   0.5,   3.0, False),
]

RANGE_EXTRA = ("range_max_hold_hours", 6.0, 168.0, True)


def action_to_params(action: np.ndarray, strategy: str) -> Dict[str, Any]:
    """Map [-1, 1] action array to parameter dict."""
    table = list(PARAM_TABLE)
    if strategy == "range":
        table.append(RANGE_EXTRA)

    params = {}
    for i, (name, lo, hi, is_int) in enumerate(table):
        if i >= len(action):
            break
        # Linear map: [-1, 1] -> [lo, hi]
        val = (action[i] + 1.0) / 2.0 * (hi - lo) + lo
        if is_int:
            val = int(round(val))
        params[name] = val
    return params


class TradingParamEnv(gymnasium.Env):
    """Single-step RL environment for trading parameter optimization.

    reset(): pick random symbol + 90-day window, compute features
    step(action): map to params, backtest, return reward
    """

    metadata = {"render_modes": []}

    def __init__(self, candle_store, symbols: List[str], strategy: str,
                 window_days: int = 90):
        super().__init__()
        self._candle_store = candle_store
        self._symbols = symbols
        self._strategy = strategy
        self._window_days = window_days

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(27,), dtype=np.float32,
        )
        n_actions = 22 if strategy == "range" else 21
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_actions,), dtype=np.float32,
        )

        # Pre-compute available windows per symbol
        self._windows: List[Tuple[str, datetime, datetime]] = []
        self._build_window_list()

        self._current_obs: Optional[np.ndarray] = None
        self._current_candles_5m = None
        self._current_symbol: Optional[str] = None

    def _build_window_list(self) -> None:
        """Build list of (symbol, start, end) windows from available data."""
        step = timedelta(days=14)
        window = timedelta(days=self._window_days)
        for sym in self._symbols:
            date_range = self._candle_store.get_date_range(sym)
            if date_range is None:
                continue
            min_ts, max_ts = date_range
            if not hasattr(min_ts, 'tzinfo') or min_ts.tzinfo is None:
                min_ts = min_ts.replace(tzinfo=timezone.utc)
                max_ts = max_ts.replace(tzinfo=timezone.utc)
            total_days = (max_ts - min_ts).days
            if total_days < self._window_days:
                continue
            # Hold out last 10% for validation
            usable_end = min_ts + timedelta(days=int(total_days * 0.9))
            start = min_ts
            while start + window <= usable_end:
                self._windows.append((sym, start, start + window))
                start += step

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if not self._windows:
            self._current_obs = np.zeros(27, dtype=np.float32)
            return self._current_obs, {}

        # Pick random window
        idx = self.np_random.integers(0, len(self._windows))
        sym, start, end = self._windows[idx]
        self._current_symbol = sym

        # Get 1m candles for features
        candles_1m = self._candle_store.get_candles(sym, start, end, resample="1m")

        # Get BTC candles for correlation features
        btc_1m = None
        if sym != "BTC-EUR":
            btc_1m = self._candle_store.get_candles("BTC-EUR", start, end, resample="1m")

        # Compute features
        self._current_obs = extract_features(candles_1m, btc_1m)

        # Get 5m candles for backtesting
        self._current_candles_5m = self._candle_store.get_candles(sym, start, end, resample="5m")

        return self._current_obs, {}

    def step(self, action):
        params = action_to_params(action, self._strategy)

        # Run backtest
        try:
            if self._current_candles_5m is None or len(self._current_candles_5m) < 100:
                result = BacktestResult()
            else:
                candle_dicts = []
                for ts, row in self._current_candles_5m.iterrows():
                    candle_dicts.append({
                        "timestamp": str(ts),
                        "open": row["open"], "high": row["high"],
                        "low": row["low"], "close": row["close"],
                        "volume": row["volume"],
                        "symbol": self._current_symbol or "TEST-EUR",
                    })
                engine = BacktestEngine(
                    candle_dicts,
                    strategy_params=params,
                    target_strategy=self._strategy,
                )
                result = engine.run()
        except Exception as e:
            logger.warning("Backtest failed in env: %s", e)
            result = BacktestResult()

        reward = self._compute_reward(result)
        return self._current_obs, reward, True, False, {"result": result}

    def _compute_reward(self, result: BacktestResult) -> float:
        pnl_norm = max(min(result.total_pnl / 1000.0, 3.0), -3.0)
        sharpe_clipped = max(min(result.sharpe_ratio, 3.0), -3.0)
        pf = result.profit_factor
        pf_clipped = min(pf, 5.0) if not (math.isinf(pf) or math.isnan(pf)) else 5.0
        ppf_clipped = max(min(result.profit_per_fee, 5.0), -5.0)

        base = (sharpe_clipped * 0.30
                + pnl_norm * 0.30
                + pf_clipped * 0.20
                + ppf_clipped * 0.20)

        penalty = 0.0
        if result.max_drawdown_pct > 20.0:
            penalty -= 5.0
        if result.total_trades < 3:
            penalty -= 3.0
        if result.total_trades > 0 and result.win_rate < 20.0:
            penalty -= 2.0

        return base + penalty
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_rl_environment.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add bot/learning/rl_environment.py tests/test_rl_environment.py
git commit -m "feat: implement Gymnasium RL environment with reward function"
```

---

## Chunk 5: RL Trainer, Optimizer, Walk-Forward Integration

### Task 7: Implement RLTrainer

**Files:**
- Create: `bot/learning/rl_trainer.py`
- Create: `tests/test_rl_trainer.py`

- [ ] **Step 1: Write failing tests for RLTrainer**

Create `tests/test_rl_trainer.py`:

```python
"""Tests for RLTrainer — validation gate and atomic save."""
import os
from unittest.mock import MagicMock, patch
from bot.learning.rl_trainer import RLTrainer


def test_trainer_init_creates_model_dir(tmp_path):
    """RLTrainer creates model directory on init."""
    model_dir = str(tmp_path / "models")
    store = MagicMock()
    trainer = RLTrainer(store, ["squeeze"], model_dir=model_dir)
    assert os.path.isdir(model_dir)


def test_trainer_skips_when_no_symbols():
    """train() returns empty dict when no candle data available."""
    store = MagicMock()
    store.available_symbols.return_value = []
    trainer = RLTrainer(store, ["squeeze"], model_dir="/tmp/rl_test")
    result = trainer.train()
    assert result == {}


def test_validate_returns_float():
    """_validate returns average reward as float."""
    store = MagicMock()
    store.get_date_range.return_value = None
    trainer = RLTrainer(store, ["squeeze"], model_dir="/tmp/rl_test")
    mock_model = MagicMock()
    mock_model.predict.return_value = (
        __import__("numpy").zeros(21, dtype=__import__("numpy").float32), None
    )
    avg = trainer._validate(mock_model, "squeeze", ["BTC-EUR"], n_episodes=2)
    assert isinstance(avg, float)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rl_trainer.py -v`
Expected: FAIL with "cannot import name 'RLTrainer'"

- [ ] **Step 3: Implement RLTrainer**

Create `bot/learning/rl_trainer.py`:

```python
"""PPO training loop for trading parameter optimization."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class RLTrainer:
    """Train and validate PPO models for each strategy."""

    def __init__(self, candle_store, strategies: List[str],
                 model_dir: str = "models/rl_optimizer") -> None:
        self._candle_store = candle_store
        self._strategies = strategies
        self._model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def train(self, total_timesteps: int = 25_000) -> Dict[str, float]:
        """Full training run. Returns {strategy: avg_reward}."""
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

        from bot.learning.rl_environment import TradingParamEnv

        symbols = self._candle_store.available_symbols()
        if not symbols:
            logger.warning("RLTrainer: no symbols with candle data, skipping")
            return {}

        n_envs = min(os.cpu_count() or 1, 4)
        results = {}

        for strategy in self._strategies:
            logger.info("Training %s model (%d timesteps, %d envs)...",
                        strategy, total_timesteps, n_envs)

            def make_env(s=strategy):
                return TradingParamEnv(self._candle_store, symbols, s)

            # Try SubprocVecEnv, fall back to DummyVecEnv
            try:
                vec_env = SubprocVecEnv([make_env for _ in range(n_envs)])
            except Exception as e:
                logger.warning("SubprocVecEnv failed (%s), using DummyVecEnv", e)
                vec_env = DummyVecEnv([make_env for _ in range(n_envs)])

            try:
                model = PPO(
                    "MlpPolicy", vec_env,
                    learning_rate=3e-4,
                    n_steps=2048,
                    batch_size=64,
                    policy_kwargs={"net_arch": [256, 256, 256]},
                    verbose=0,
                )
                model.learn(total_timesteps=total_timesteps)

                # Validate
                avg_reward = self._validate(model, strategy, symbols)
                results[strategy] = avg_reward

                # Check against existing model
                existing_path = os.path.join(self._model_dir, f"{strategy}_ppo.zip")
                if os.path.exists(existing_path):
                    old_model = PPO.load(existing_path)
                    old_reward = self._validate(old_model, strategy, symbols)
                    if avg_reward <= old_reward:
                        logger.warning(
                            "%s: new model (%.3f) not better than old (%.3f), keeping old",
                            strategy, avg_reward, old_reward,
                        )
                        continue

                # Atomic save
                tmp_path = os.path.join(self._model_dir, f"{strategy}_ppo.tmp.zip")
                final_path = os.path.join(self._model_dir, f"{strategy}_ppo.zip")
                model.save(tmp_path)
                os.replace(tmp_path, final_path)
                logger.info("%s: model saved (avg_reward=%.3f)", strategy, avg_reward)

            finally:
                vec_env.close()

        return results

    def retrain(self, fine_tune_months: int = 6,
                total_timesteps: int = 5_000) -> Dict[str, float]:
        """Weekly incremental retrain on recent data."""
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        from bot.learning.rl_environment import TradingParamEnv

        symbols = self._candle_store.available_symbols()
        if not symbols:
            return {}

        results = {}
        for strategy in self._strategies:
            model_path = os.path.join(self._model_dir, f"{strategy}_ppo.zip")
            if not os.path.exists(model_path):
                logger.warning("No existing model for %s, skipping retrain", strategy)
                continue

            env = DummyVecEnv([
                lambda s=strategy: TradingParamEnv(self._candle_store, symbols, s)
            ])

            try:
                model = PPO.load(model_path, env=env)
                model.learn(total_timesteps=total_timesteps)

                avg_reward = self._validate(model, strategy, symbols)
                results[strategy] = avg_reward

                # Validation gate
                old_model = PPO.load(model_path)
                old_reward = self._validate(old_model, strategy, symbols)
                if avg_reward <= old_reward:
                    logger.warning(
                        "%s retrain: new (%.3f) not better than old (%.3f), keeping old",
                        strategy, avg_reward, old_reward,
                    )
                    continue

                tmp_path = os.path.join(self._model_dir, f"{strategy}_ppo.tmp.zip")
                model.save(tmp_path)
                os.replace(tmp_path, model_path)
                logger.info("%s: retrained model saved (%.3f)", strategy, avg_reward)

            finally:
                env.close()

        return results

    def _validate(self, model, strategy: str, symbols: List[str],
                  n_episodes: int = 20) -> float:
        """Run model on validation episodes, return average reward."""
        from bot.learning.rl_environment import TradingParamEnv

        env = TradingParamEnv(self._candle_store, symbols, strategy)
        total_reward = 0.0
        for _ in range(n_episodes):
            obs, _ = env.reset()
            action, _ = model.predict(obs, deterministic=True)
            _, reward, _, _, _ = env.step(action)
            total_reward += reward
        return total_reward / max(n_episodes, 1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_rl_trainer.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add bot/learning/rl_trainer.py tests/test_rl_trainer.py
git commit -m "feat: implement RLTrainer with PPO training, validation, and atomic save"
```

---

### Task 8: Implement RLOptimizer (inference)

> **Note:** The spec defines `RLOptimizer.predict` accepting raw candles and doing feature extraction internally. This plan instead accepts pre-computed features (`np.ndarray`) for cleaner separation of concerns — the caller (walk-forward) handles feature extraction. This is an intentional deviation from the spec.

**Files:**
- Create: `bot/learning/rl_optimizer.py`
- Create: `tests/test_rl_optimizer.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_rl_optimizer.py`:

```python
"""Tests for RLOptimizer — inference from trained models."""
import numpy as np
from unittest.mock import MagicMock, patch
from bot.learning.rl_optimizer import RLOptimizer


def test_predict_fallback_no_model():
    """Returns CHAMPION_DEFAULTS when no model loaded."""
    from bot.strategy.adopted_universe import CHAMPION_DEFAULTS
    opt = RLOptimizer(model_dir="/nonexistent")
    result = opt.predict(np.zeros(27, dtype=np.float32), "orderflow")
    assert result == CHAMPION_DEFAULTS


def test_predict_with_mock_model():
    """Returns valid param dict when model is loaded."""
    opt = RLOptimizer.__new__(RLOptimizer)
    opt._model_dir = "/tmp"
    opt._models = {}

    # Mock a model that returns all-zeros action
    mock_model = MagicMock()
    mock_model.predict.return_value = (np.zeros(21, dtype=np.float32), None)
    opt._models["squeeze"] = mock_model

    result = opt.predict(np.zeros(27, dtype=np.float32), "squeeze")
    assert isinstance(result, dict)
    assert "atr_multiplier" in result
    # Midpoint of [1.0, 15.0] at action=0 is 8.0
    assert abs(result["atr_multiplier"] - 8.0) < 0.01


def test_predict_range_model_has_extra_param():
    """Range model returns range_max_hold_hours."""
    opt = RLOptimizer.__new__(RLOptimizer)
    opt._model_dir = "/tmp"
    opt._models = {}

    mock_model = MagicMock()
    mock_model.predict.return_value = (np.zeros(22, dtype=np.float32), None)
    opt._models["range"] = mock_model

    result = opt.predict(np.zeros(27, dtype=np.float32), "range")
    assert "range_max_hold_hours" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rl_optimizer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement RLOptimizer**

Create `bot/learning/rl_optimizer.py`:

```python
"""RLOptimizer — load trained PPO models and predict parameters."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import numpy as np

from bot.learning.rl_environment import action_to_params
from bot.strategy.adopted_universe import CHAMPION_DEFAULTS

logger = logging.getLogger(__name__)


class RLOptimizer:
    """Load trained PPO models and predict trading parameters from market features."""

    def __init__(self, model_dir: str = "models/rl_optimizer") -> None:
        self._model_dir = model_dir
        self._models: Dict[str, Any] = {}  # strategy -> PPO model

    def load_models(self) -> None:
        """Load all strategy models from disk."""
        from stable_baselines3 import PPO

        if not os.path.isdir(self._model_dir):
            logger.warning("RLOptimizer: model dir %s not found", self._model_dir)
            return

        loaded = 0
        for fname in os.listdir(self._model_dir):
            if fname.endswith("_ppo.zip") and not fname.endswith(".tmp.zip"):
                strategy = fname.replace("_ppo.zip", "")
                path = os.path.join(self._model_dir, fname)
                try:
                    self._models[strategy] = PPO.load(path)
                    loaded += 1
                    logger.info("Loaded RL model for %s", strategy)
                except Exception as e:
                    logger.warning("Failed to load model %s: %s", path, e)

        if loaded == 0:
            logger.warning("RLOptimizer: no models loaded, will use CHAMPION_DEFAULTS")

    def predict(self, features: np.ndarray, strategy: str) -> Dict[str, Any]:
        """Predict parameters from 27-element feature vector.

        Returns parameter dict. Falls back to CHAMPION_DEFAULTS if model not loaded.
        """
        model = self._models.get(strategy)
        if model is None:
            logger.debug("No model for %s, using CHAMPION_DEFAULTS", strategy)
            return dict(CHAMPION_DEFAULTS)

        try:
            action, _ = model.predict(features, deterministic=True)
            return action_to_params(action, strategy)
        except Exception as e:
            logger.warning("Prediction failed for %s: %s", strategy, e)
            return dict(CHAMPION_DEFAULTS)

    def has_model(self, strategy: str) -> bool:
        """Check if a model is loaded for the given strategy."""
        return strategy in self._models
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_rl_optimizer.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add bot/learning/rl_optimizer.py tests/test_rl_optimizer.py
git commit -m "feat: implement RLOptimizer with model loading and CHAMPION_DEFAULTS fallback"
```

---

### Task 9: Replace Optuna with RL in walk_forward.py

**Files:**
- Modify: `bot/learning/walk_forward.py`

- [ ] **Step 1: Add required imports and modify WalkForwardOptimizer constructor**

In `bot/learning/walk_forward.py`, add these imports at the top of the file:
```python
import numpy as np
from bot.backtest.engine import BacktestResult
```

Then change `__init__`:

```python
def __init__(self, rl_optimizer=None, candle_store=None) -> None:
    self._rl_optimizer = rl_optimizer
    self._candle_store = candle_store
    self._latest_result: Optional[WFResult] = None
    self._results_by_symbol: Dict[str, WFResult] = {}
    self._running = False
```

- [ ] **Step 2: Add `_run_rl_window` method**

Add to `WalkForwardOptimizer`:

```python
def _run_rl_window(self, train_candles_5m, test_candles_5m,
                    target_strategy=None):
    """RL-driven parameter optimization for a single window."""
    from bot.learning.feature_extractor import extract_features
    from bot.strategy.adopted_universe import CHAMPION_DEFAULTS

    if self._rl_optimizer is None or not self._rl_optimizer.has_model(target_strategy or ""):
        return dict(CHAMPION_DEFAULTS), BacktestResult()

    # Get 1m candles for feature extraction from CandleStore
    features = np.zeros(27, dtype=np.float32)
    if self._candle_store is not None and train_candles_5m:
        try:
            # Determine date range from 5m candles
            if isinstance(train_candles_5m[0], dict):
                first_ts = train_candles_5m[0].get("timestamp", "")
                last_ts = train_candles_5m[-1].get("timestamp", "")
            else:
                first_ts = train_candles_5m[0].timestamp
                last_ts = train_candles_5m[-1].timestamp

            from datetime import datetime, timezone
            start = datetime.fromisoformat(str(first_ts).replace("Z", "+00:00"))
            end = datetime.fromisoformat(str(last_ts).replace("Z", "+00:00"))

            symbol = (train_candles_5m[0].get("symbol", "BTC-EUR")
                      if isinstance(train_candles_5m[0], dict)
                      else getattr(train_candles_5m[0], "symbol", "BTC-EUR"))

            candles_1m = self._candle_store.get_candles(symbol, start, end, resample="1m")
            btc_1m = None
            if symbol != "BTC-EUR":
                btc_1m = self._candle_store.get_candles("BTC-EUR", start, end, resample="1m")
            features = extract_features(candles_1m, btc_1m)
        except Exception as e:
            logger.warning("Feature extraction failed in walk-forward: %s", e)

    # Predict parameters
    params = self._rl_optimizer.predict(features, target_strategy or "orderflow")

    # Run backtest on test candles (OOS validation) — inline, not via
    # _run_single_backtest which returns a tuple instead of BacktestResult
    try:
        from bot.backtest.engine import BacktestEngine
        test_engine = BacktestEngine(
            test_candles_5m,
            strategy_params=params,
            target_strategy=target_strategy,
        )
        test_result = test_engine.run()
    except Exception:
        test_result = BacktestResult()

    return params, test_result
```

- [ ] **Step 3: Update `_execute_from_candles` to use `_run_rl_window`**

Replace calls to `_run_optuna_window` with `self._run_rl_window`. In the `_execute_from_candles` method, change:

```python
# OLD:
best_params, test_result = await loop.run_in_executor(
    None, _run_optuna_window, train_candles, test_candles, max_workers
)

# NEW:
best_params, test_result = await loop.run_in_executor(
    None, self._run_rl_window, train_candles, test_candles, target_strategy
)
```

In `_execute` (around line 329-331), also replace the `_run_optuna_window` call:
```python
# OLD:
best_params, test_result = await loop.run_in_executor(
    None, _run_optuna_window, train_candles, test_candles, max_workers
)

# NEW (note: _execute does not have target_strategy, pass None):
best_params, test_result = await loop.run_in_executor(
    None, self._run_rl_window, train_candles, test_candles, None
)
```

- [ ] **Step 4: Remove `_run_optuna_window` function, optuna imports, and OPTUNA_TRIALS references**

Delete the entire `_run_optuna_window` top-level function.
Remove `import optuna` and the `OPTUNA_TRIALS` constant.
Update all log messages that reference `OPTUNA_TRIALS` or Optuna:

**Line ~307-308** (in `_execute`): Replace:
```python
logger.info("Walk-forward [%s]: Optuna Bayesian optimization, %d trials/window, %d windows", ...)
```
With:
```python
logger.info("Walk-forward [%s]: RL optimization, %d windows", symbol, len(windows_spec))
```

**Line ~326-327** (in `_execute`): Replace any `OPTUNA_TRIALS` or Optuna reference with RL terminology.

**Line ~396-398** (in `_execute_from_candles`): Same — update log messages to reference RL instead of Optuna.

Search for ALL remaining references to `OPTUNA_TRIALS`, `optuna`, or `Optuna` in the file and remove/update them.

- [ ] **Step 5: Run existing walk-forward tests**

Run: `pytest tests/test_walk_forward.py tests/test_wf_integration.py -v`
Expected: Tests may need adjustment if they depend on Optuna. Fix any that reference `_run_optuna_window` directly.

- [ ] **Step 6: Commit**

```bash
git add bot/learning/walk_forward.py
git commit -m "feat: replace Optuna with RL optimizer in walk-forward"
```

---

## Chunk 6: Wiring, Scheduler, Bootstrap, Integration Test

### Task 10: Wire scheduler and main.py

**Files:**
- Modify: `bot/scheduler.py`
- Modify: `bot/main.py`

- [ ] **Step 1: Add `rl_retrain_run` to BotScheduler**

In `bot/scheduler.py`, add `rl_retrain_run: Optional[Callable] = None` parameter to `__init__`:

```python
def __init__(
    self,
    ...,  # all existing params unchanged
    rl_retrain_run: Optional[Callable] = None,
) -> None:
    ...  # existing assignments
    self._rl_retrain_run = rl_retrain_run
```

In `run_all()`, add:
```python
if self._rl_retrain_run is not None:
    coros.append(self._rl_retrain_loop(168))
```

Add loop method:
```python
async def _rl_retrain_loop(self, interval_hours: int) -> None:
    await asyncio.sleep(600)  # let walk-forward complete first
    while True:
        try:
            logger.info("RL retrain starting...")
            await self._rl_retrain_run()
        except Exception as e:
            logger.error("RL retrain error: %s", e)
        await asyncio.sleep(interval_hours * 3600)
```

- [ ] **Step 2: Wire in main.py**

In `bot/main.py`, add imports and wiring after DB init:

```python
from bot.data.candle_store import CandleStore
from bot.learning.rl_optimizer import RLOptimizer
from bot.learning.rl_trainer import RLTrainer
```

After settings/DB are initialized:
```python
candle_store = CandleStore(db_url=settings.database_url)
rl_optimizer = RLOptimizer()
rl_optimizer.load_models()
```

Update `WalkForwardOptimizer` creation:
```python
walk_forward = WalkForwardOptimizer(rl_optimizer=rl_optimizer, candle_store=candle_store)
```

Add retrain callback:
```python
async def _retrain_rl():
    trainer = RLTrainer(candle_store, ALL_STRATEGIES)
    symbols = get_tradeable_symbols() or ["BTC-EUR"]
    await candle_store.incremental_update(symbols)
    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(None, trainer.retrain)
    rl_optimizer.load_models()
    logger.info("RL retrain complete: %s", results)
```

Pass to scheduler:
```python
scheduler = BotScheduler(
    ...,  # existing args
    rl_retrain_run=_retrain_rl,
)
```

- [ ] **Step 3: Commit**

```bash
git add bot/scheduler.py bot/main.py
git commit -m "feat: wire RL optimizer, candle store, and retrain scheduler"
```

---

### Task 11: Create bootstrap script

**Files:**
- Create: `bot/scripts/__init__.py`
- Create: `bot/scripts/rl_bootstrap.py`

- [ ] **Step 1: Create bootstrap script**

Create `bot/scripts/__init__.py` (empty file).

Create `bot/scripts/rl_bootstrap.py`:

```python
"""One-time bootstrap: download 1m candles and train initial RL models.

Usage: python -m bot.scripts.rl_bootstrap
"""
from __future__ import annotations

import asyncio
import logging
import subprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


async def main():
    from bot.config import get_settings
    from bot.data.candle_store import CandleStore
    from bot.learning.rl_trainer import RLTrainer
    from bot.strategy.adopted_universe import ALL_STRATEGIES

    settings = get_settings()
    candle_store = CandleStore(db_url=settings.database_url)

    # Step 1: Run Alembic migration (must happen before download writes to candle_1m)
    logger.info("Running alembic upgrade head to ensure candle_1m table exists...")
    subprocess.run(["alembic", "upgrade", "head"], check=True)

    # Step 2: Download candles
    symbols = None
    try:
        from bot.main import get_tradeable_symbols
        symbols = get_tradeable_symbols()
    except Exception as e:
        logger.warning("Could not get tradeable symbols: %s", e)
    if not symbols:
        symbols = ["BTC-EUR", "ETH-EUR"]
        logger.warning("Could not get tradeable symbols, using fallback: %s", symbols)

    logger.info("Downloading 1m candles for %d symbols...", len(symbols))
    await candle_store.bulk_download(symbols)
    logger.info("Download complete.")

    # Step 3: Train models
    logger.info("Training RL models for strategies: %s", ALL_STRATEGIES)
    trainer = RLTrainer(candle_store, ALL_STRATEGIES)
    results = trainer.train()
    logger.info("Training complete: %s", results)


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Commit**

```bash
git add bot/scripts/
git commit -m "feat: add rl_bootstrap script for initial candle download and training"
```

---

### Task 12: Integration test

**Files:**
- Create: `tests/test_rl_integration.py`

- [ ] **Step 1: Write integration test**

Create `tests/test_rl_integration.py`:

```python
"""Integration test: train tiny model -> predict -> verify valid params.

This tests the full train-save-load-predict pipeline end-to-end.
"""
import os
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from bot.learning.rl_environment import TradingParamEnv, action_to_params
from bot.learning.rl_optimizer import RLOptimizer
from bot.strategy.adopted_universe import CHAMPION_DEFAULTS


def _make_mock_candle_store():
    """Create a mock CandleStore that returns synthetic candle data."""
    store = MagicMock()
    # Provide enough data for a 90-day window
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 6, 1, tzinfo=timezone.utc)
    store.get_date_range.return_value = (start, end)
    store.available_symbols.return_value = ["BTC-EUR"]

    n = 1440 * 90  # 90 days of 1m candles
    timestamps = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    np.random.seed(42)
    prices = 50000.0 * np.exp(np.cumsum(np.random.normal(0, 0.0001, n)))

    df_1m = pd.DataFrame({
        "open": prices, "high": prices * 1.001, "low": prices * 0.999,
        "close": prices, "volume": np.random.exponential(100, n),
    }, index=timestamps)

    df_5m = df_1m.resample("5min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()

    def get_candles(symbol, s, e, resample="5m"):
        if resample == "1m":
            return df_1m.loc[s:e]
        return df_5m.loc[s:e]

    store.get_candles.side_effect = get_candles
    return store


@pytest.mark.slow
def test_train_predict_pipeline(tmp_path):
    """Train a tiny PPO model, save, load via RLOptimizer, predict valid params."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    store = _make_mock_candle_store()
    model_dir = str(tmp_path / "models")
    os.makedirs(model_dir)

    # Train a tiny model (100 timesteps for speed)
    env = DummyVecEnv([lambda: TradingParamEnv(store, ["BTC-EUR"], "squeeze")])
    model = PPO("MlpPolicy", env, n_steps=32, batch_size=32, verbose=0)
    model.learn(total_timesteps=64)

    # Save model
    model_path = os.path.join(model_dir, "squeeze_ppo.zip")
    model.save(model_path)
    env.close()

    # Load via RLOptimizer and predict
    optimizer = RLOptimizer(model_dir=model_dir)
    optimizer.load_models()
    assert optimizer.has_model("squeeze")

    features = np.random.randn(27).astype(np.float32)
    features = np.clip(features, -1.0, 1.0)
    params = optimizer.predict(features, "squeeze")

    # Verify all expected params are present and within bounds
    assert isinstance(params, dict)
    assert params["atr_multiplier"] >= 1.0 and params["atr_multiplier"] <= 15.0
    assert params["rr_ratio"] >= 1.0 and params["rr_ratio"] <= 8.0
    assert params["base_risk_pct"] >= 1.0 and params["base_risk_pct"] <= 8.0
    assert "range_max_hold_hours" not in params  # squeeze, not range
    assert isinstance(params["max_hold_hours"], int)
    assert isinstance(params["consecutive_confirms"], int)


def test_action_to_params_all_strategies_all_bounds():
    """All action values produce valid parameter ranges for all strategies."""
    for strategy in ["orderflow", "range", "squeeze", "funding_contrarian"]:
        n = 22 if strategy == "range" else 21
        for val in [-1.0, 0.0, 1.0]:
            action = np.full(n, val, dtype=np.float32)
            params = action_to_params(action, strategy)
            assert params["atr_multiplier"] >= 1.0
            assert params["atr_multiplier"] <= 15.0
            if strategy == "range":
                assert "range_max_hold_hours" in params
                assert 6 <= params["range_max_hold_hours"] <= 168


def test_optimizer_fallback_without_models():
    """RLOptimizer returns CHAMPION_DEFAULTS when no models exist."""
    opt = RLOptimizer(model_dir="/nonexistent_dir")
    opt.load_models()
    result = opt.predict(np.zeros(27, dtype=np.float32), "squeeze")
    assert result == CHAMPION_DEFAULTS
```

- [ ] **Step 2: Run integration tests**

Run: `pytest tests/test_rl_integration.py -v`
Expected: All PASS

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v --timeout=120`
Expected: All tests pass (existing + new)

- [ ] **Step 4: Commit**

```bash
git add tests/test_rl_integration.py
git commit -m "test: add RL optimizer integration tests"
```

---

## Summary

| Chunk | Tasks | New Files | Modified Files |
|-------|-------|-----------|---------------|
| 1: Foundation | 1-3 | candle_store.py, test_candle_store.py | requirements.txt, models.py |
| 2: Feature Extractor | 4 | feature_extractor.py, test_feature_extractor.py | — |
| 3: Engine Params | 5 | test_backtest_rl_params.py | engine.py |
| 4: RL Environment | 6 | rl_environment.py, test_rl_environment.py | — |
| 5: Trainer + Optimizer + WF | 7-9 | rl_trainer.py, test_rl_trainer.py, rl_optimizer.py, test_rl_optimizer.py | walk_forward.py |
| 6: Wiring + Bootstrap | 10-12 | rl_bootstrap.py, test_rl_integration.py | scheduler.py, main.py |

**Total: 12 tasks, 13 new files, 6 modified files**

After all tasks complete, run the bootstrap script to download candles and train models:
```bash
python -m bot.scripts.rl_bootstrap
```
