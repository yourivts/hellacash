"""Test that backtest precompute signals includes EMA keys."""
import numpy as np
import pandas as pd
import pytest
from bot.backtest.engine import BacktestEngine


class TestEMATrendScoring:
    """After the engine overhaul, _score_from_precomputed was removed.

    We now test that the precomputed signals still contain the expected
    EMA keys used by the new strategy evaluate_1h() paths.
    """

    def test_ema200_key_exists_in_precomp(self):
        """_precompute_signals must include ema200 key."""
        n = 250
        df = pd.DataFrame({
            "open": np.linspace(100, 110, n),
            "high": np.linspace(101, 111, n),
            "low": np.linspace(99, 109, n),
            "close": np.linspace(100, 110, n),
            "volume": np.full(n, 1000.0),
        }, index=pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC"))
        result = BacktestEngine._precompute_signals(df, "BTC-EUR")
        assert "ema200" in result, "ema200 missing from precomputed signals"

    def test_ema50_key_exists_in_precomp(self):
        """_precompute_signals must include ema50 key (used for squeeze slope)."""
        n = 250
        df = pd.DataFrame({
            "open": np.linspace(100, 110, n),
            "high": np.linspace(101, 111, n),
            "low": np.linspace(99, 109, n),
            "close": np.linspace(100, 110, n),
            "volume": np.full(n, 1000.0),
        }, index=pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC"))
        result = BacktestEngine._precompute_signals(df, "BTC-EUR")
        assert "ema50" in result, "ema50 missing from precomputed signals"

    def test_bb_bandwidth_key_exists_in_precomp(self):
        """_precompute_signals must include bb_bandwidth for range/squeeze."""
        n = 250
        df = pd.DataFrame({
            "open": np.linspace(100, 110, n),
            "high": np.linspace(101, 111, n),
            "low": np.linspace(99, 109, n),
            "close": np.linspace(100, 110, n),
            "volume": np.full(n, 1000.0),
        }, index=pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC"))
        result = BacktestEngine._precompute_signals(df, "BTC-EUR")
        assert "bb_bandwidth" in result, "bb_bandwidth missing from precomputed signals"
