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
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


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
