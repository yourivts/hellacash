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

    def test_entry_threshold_default_is_050(self):
        strat = HybridStrategy()
        assert strat.entry_threshold == 0.50
