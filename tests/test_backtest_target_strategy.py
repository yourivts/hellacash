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
        engine_of = BacktestEngine(
            candles, strategy_params={"base_risk_pct": 3.0},
            target_strategy="orderflow",
        )
        result_of = engine_of.run()
        for trade in result_of.trade_log:
            assert trade.strategy == "orderflow", f"Expected orderflow, got {trade.strategy}"

    def test_target_strategy_none_unchanged(self):
        candles = _make_candles(5000)
        engine = BacktestEngine(candles, strategy_params={"base_risk_pct": 3.0})
        result = engine.run()
        assert result is not None

    def test_confluence_disabled_with_target(self):
        """When target_strategy is set, confluence should be skipped."""
        candles = _make_candles(5000)
        engine = BacktestEngine(
            candles, strategy_params={"base_risk_pct": 3.0},
            target_strategy="range",
        )
        result = engine.run()
        assert result is not None
        for trade in result.trade_log:
            assert trade.strategy == "range", f"Expected range, got {trade.strategy}"


class TestRegimeParamsFromStrategy:
    def test_regime_uses_strategy_params(self):
        candles = _make_candles(5000)
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
        assert result is not None
