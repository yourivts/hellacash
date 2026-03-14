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


class TestSignalProviderProtocol:
    def test_valid_provider_matches_protocol(self):
        p = _DummyProvider()
        assert isinstance(p, SignalProvider)

    def test_invalid_provider_does_not_match(self):
        p = _BadProvider()
        assert not isinstance(p, SignalProvider)


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
        result = voter.vote(df_15m=df, df_1h=df, df_4h=df, df_1d=df, regime="trending")
        assert isinstance(result, MTFResult)
        assert -1.0 <= result.mtf_score <= 1.0
        assert 0.0 <= result.agreement_ratio <= 1.0
        assert len(result.per_tf_scores) == 4

    def test_regime_weights_used(self):
        assert "trending" in REGIME_WEIGHTS
        assert "ranging" in REGIME_WEIGHTS
        w = REGIME_WEIGHTS["trending"]
        assert abs(sum(w.values()) - 1.0) < 0.01

    def test_insufficient_data_returns_neutral(self):
        tiny = _make_ohlcv(5)
        voter = MTFVoter()
        result = voter.vote(df_15m=tiny, df_1h=tiny, df_4h=tiny, df_1d=tiny, regime="unknown")
        assert result.mtf_score == 0.0
