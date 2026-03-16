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


import pytest
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
                               resample="5m")
    assert len(result) == 12
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
    store._get_data_span_days = MagicMock(return_value=30)
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
