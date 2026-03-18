"""Tests for external data pipeline — fetching, caching, staleness."""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest


def _make_5m_index(days: int = 30) -> pd.DatetimeIndex:
    """Create a 5-minute DatetimeIndex for testing."""
    end = datetime(2024, 6, 1, tzinfo=timezone.utc)
    start = end - timedelta(days=days)
    return pd.date_range(start, end, freq="5min", tz=timezone.utc)


class TestExternalDataProvider:
    """Tests for the ExternalDataProvider class.

    All tests mock fetch_all_training() to avoid real HTTP requests.
    This ensures tests are fast, deterministic, and work offline/in CI.
    """

    def _make_mock_training_data(self, idx):
        """Create a synthetic training data dict (same keys as fetch_all_training)."""
        n = len(idx)
        return {
            "fear_greed": np.random.rand(n) * 0.5 + 0.25,
            "fear_greed_mom": np.random.randn(n) * 0.1,
            "gtrends_bitcoin": np.zeros(n),
            "gtrends_crypto": np.zeros(n),
            "dxy_return": np.random.randn(n) * 0.1,
            "sp500_return": np.random.randn(n) * 0.1,
            "gold_return": np.random.randn(n) * 0.1,
            "vix": np.random.rand(n) * 0.5,
            "treasury_10y": np.random.rand(n) * 0.5,
            "yield_spread": np.random.randn(n) * 0.2,
            "nvt": np.random.rand(n) * 0.5,
            "mvrv": np.random.rand(n) * 0.5,
            "sopr": np.random.randn(n) * 0.2,
            "puell": np.random.rand(n) * 0.5,
            "hashrate": np.random.randn(n) * 0.1,
            "eth_active_addr": np.random.randn(n) * 0.1,
            "stable_supply_change": np.random.randn(n) * 0.05,
            "tvl_change": np.random.randn(n) * 0.05,
            "oi_change_7d": np.zeros(n),
            "liq_ratio": np.zeros(n),
            "taker_buy_ratio": np.random.rand(n) * 0.3 + 0.35,
            "dvol": np.random.rand(n) * 0.5,
            "funding_24h_avg": np.random.randn(n) * 0.05,
            "btc_dom_change": np.zeros(n),
            "stable_btc_ratio": np.zeros(n),
            "funding_rate_current": np.random.randn(n) * 0.05,
            "funding_7d_avg": np.random.randn(n) * 0.03,
            "oi_change_24h": np.zeros(n),
            "long_liq_24h": np.zeros(n),
            "short_liq_24h": np.zeros(n),
            "exchange_netflow": np.random.randn(n) * 0.1,
            "active_addr_change_7d": np.random.randn(n) * 0.1,
        }

    def test_build_training_features_returns_correct_shape(self):
        """Training features should have 32 columns (7 for 55-61 + 25 for 65-89)."""
        from unittest.mock import patch
        from bot.data.external_features import ExternalDataProvider
        provider = ExternalDataProvider(cache_dir=None)
        idx = _make_5m_index(30)
        mock_data = self._make_mock_training_data(idx)
        with patch.object(provider, "fetch_all_training", return_value=mock_data):
            result = provider.build_training_features(idx)
        assert isinstance(result, np.ndarray)
        assert result.shape == (len(idx), 32), f"Expected (N, 32), got {result.shape}"

    def test_build_training_features_dtype_float32(self):
        from unittest.mock import patch
        from bot.data.external_features import ExternalDataProvider
        provider = ExternalDataProvider(cache_dir=None)
        idx = _make_5m_index(7)
        mock_data = self._make_mock_training_data(idx)
        with patch.object(provider, "fetch_all_training", return_value=mock_data):
            result = provider.build_training_features(idx)
        assert result.dtype == np.float32

    def test_build_training_features_all_finite(self):
        from unittest.mock import patch
        from bot.data.external_features import ExternalDataProvider
        provider = ExternalDataProvider(cache_dir=None)
        idx = _make_5m_index(7)
        mock_data = self._make_mock_training_data(idx)
        with patch.object(provider, "fetch_all_training", return_value=mock_data):
            result = provider.build_training_features(idx)
        assert np.all(np.isfinite(result)), "Non-finite values in training features"

    def test_build_training_features_empty_returns_zeros(self):
        """When fetch_all_training returns empty dict, all features should be zero."""
        from unittest.mock import patch
        from bot.data.external_features import ExternalDataProvider
        provider = ExternalDataProvider(cache_dir=None)
        idx = _make_5m_index(7)
        with patch.object(provider, "fetch_all_training", return_value={}):
            result = provider.build_training_features(idx)
        assert np.all(result == 0.0)

    def test_build_live_features_returns_correct_shape(self):
        from bot.data.external_features import ExternalDataProvider
        provider = ExternalDataProvider(cache_dir=None)
        result = provider.build_live_features()
        assert isinstance(result, np.ndarray)
        assert result.shape == (25,), "Live features cover indices 65-89 (25 features)"
        assert result.dtype == np.float32

    def test_build_live_features_all_finite(self):
        from bot.data.external_features import ExternalDataProvider
        provider = ExternalDataProvider(cache_dir=None)
        result = provider.build_live_features()
        assert np.all(np.isfinite(result))


class TestStalenessTracking:
    """Tests for data source staleness tracking."""

    def test_initial_staleness_all_fresh(self):
        from bot.data.external_features import ExternalDataProvider
        provider = ExternalDataProvider(cache_dir=None)
        assert not provider.is_critically_stale()

    def test_mark_source_stale(self):
        from bot.data.external_features import ExternalDataProvider
        provider = ExternalDataProvider(cache_dir=None)
        stale_time = datetime.now(timezone.utc) - timedelta(days=30)
        for source in provider._last_fetched:
            provider._last_fetched[source] = stale_time
        assert provider.is_critically_stale()

    def test_staleness_recovers(self):
        from bot.data.external_features import ExternalDataProvider
        provider = ExternalDataProvider(cache_dir=None)
        stale_time = datetime.now(timezone.utc) - timedelta(days=30)
        for source in provider._last_fetched:
            provider._last_fetched[source] = stale_time
        assert provider.is_critically_stale()
        fresh_time = datetime.now(timezone.utc)
        for source in provider._last_fetched:
            provider._last_fetched[source] = fresh_time
        assert not provider.is_critically_stale()

    def test_partial_staleness_below_threshold(self):
        from bot.data.external_features import ExternalDataProvider
        provider = ExternalDataProvider(cache_dir=None)
        fresh_time = datetime.now(timezone.utc)
        stale_time = datetime.now(timezone.utc) - timedelta(days=30)
        sources = list(provider._last_fetched.keys())
        for s in sources:
            provider._last_fetched[s] = fresh_time
        if sources:
            provider._last_fetched[sources[0]] = stale_time
        assert not provider.is_critically_stale()


class TestDataAlignment:
    """Tests for data alignment to 5m index."""

    def test_align_daily_to_5m_forward_fills(self):
        from bot.data.external_features import align_to_5m
        idx_5m = _make_5m_index(3)
        daily_idx = pd.date_range("2024-05-29", "2024-06-01", freq="D", tz=timezone.utc)
        daily_data = pd.Series([10.0, 20.0, 30.0, 40.0], index=daily_idx)
        aligned = align_to_5m(daily_data, idx_5m)
        assert len(aligned) == len(idx_5m)
        assert np.all(np.isfinite(aligned))

    def test_align_fills_zeros_before_inception(self):
        from bot.data.external_features import align_to_5m
        idx_5m = _make_5m_index(30)
        short_idx = pd.date_range("2024-05-27", "2024-06-01", freq="D", tz=timezone.utc)
        short_data = pd.Series(range(len(short_idx)), index=short_idx, dtype=float)
        aligned = align_to_5m(short_data, idx_5m)
        assert len(aligned) == len(idx_5m)
        assert aligned[0] == 0.0


class TestParquetCache:
    """Tests for parquet file caching."""

    def test_save_and_load_cache(self, tmp_path):
        from bot.data.external_features import save_cache, load_cache
        df = pd.DataFrame({"value": [1.0, 2.0, 3.0]}, index=pd.date_range("2024-01-01", periods=3))
        cache_path = tmp_path / "test_cache.parquet"
        save_cache(df, str(cache_path))
        loaded = load_cache(str(cache_path))
        assert loaded is not None
        assert len(loaded) == 3
        assert list(loaded.columns) == ["value"]

    def test_load_nonexistent_returns_none(self, tmp_path):
        from bot.data.external_features import load_cache
        result = load_cache(str(tmp_path / "nonexistent.parquet"))
        assert result is None


class TestBuildLiveFeatures:
    """Tests for live feature building from cache."""

    def test_live_features_reflect_cache(self):
        from bot.data.external_features import ExternalDataProvider
        provider = ExternalDataProvider(cache_dir=None)
        provider._live_cache["fear_greed"] = 0.75
        provider._live_cache["vix"] = 0.3
        result = provider.build_live_features()
        assert result[0] == pytest.approx(0.75)
        assert result[7] == pytest.approx(0.3)

    def test_live_features_default_zero_for_missing(self):
        from bot.data.external_features import ExternalDataProvider
        provider = ExternalDataProvider(cache_dir=None)
        result = provider.build_live_features()
        assert np.all(result == 0.0)
