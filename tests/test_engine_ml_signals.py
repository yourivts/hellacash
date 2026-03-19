# tests/test_engine_ml_signals.py
"""Tests for backtest engine ML signal integration."""
import json
import numpy as np
import pandas as pd
import pytest


def _make_5m_candles(bars: int = 3000) -> list:
    np.random.seed(42)
    timestamps = pd.date_range("2024-01-01", periods=bars, freq="5min")
    base = 50000.0
    returns = np.random.randn(bars) * 0.001
    close = base * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.randn(bars)) * 0.002)
    low = close * (1 - np.abs(np.random.randn(bars)) * 0.002)
    open_ = close * (1 + np.random.randn(bars) * 0.001)
    volume = np.random.exponential(100, bars)
    candles = []
    for i in range(bars):
        candles.append({
            "timestamp": str(timestamps[i]),
            "open": float(open_[i]), "high": float(high[i]),
            "low": float(low[i]), "close": float(close[i]),
            "volume": float(volume[i]),
        })
    return candles


class MockMLSignalGenerator:
    """Mock ML signal generator for testing."""
    def __init__(self, direction="LONG", prob=0.6):
        self._direction = direction
        self._prob = prob
        self.predict_count = 0

    def predict(self, df_5m, symbol="BTC-EUR", **kwargs):
        self.predict_count += 1
        up = self._prob if self._direction == "LONG" else 0.3
        down = self._prob if self._direction == "SHORT" else 0.3
        return {
            "probabilities": [up, down] * 6,
            "direction": self._direction,
            "net_up": up,
            "net_down": down,
        }


class TestEngineMlSignalPath:
    def test_engine_accepts_ml_signal_generator(self):
        from bot.backtest.engine import BacktestEngine
        candles = _make_5m_candles(3000)
        gen = MockMLSignalGenerator()
        engine = BacktestEngine(
            candles, ml_signal_generator=gen,
            initial_capital=10_000,
        )
        result = engine.run()
        assert result is not None
        assert gen.predict_count > 0

    def test_ml_path_generates_trades(self):
        from bot.backtest.engine import BacktestEngine
        candles = _make_5m_candles(3000)
        gen = MockMLSignalGenerator(direction="LONG", prob=0.7)
        engine = BacktestEngine(
            candles, ml_signal_generator=gen,
            initial_capital=10_000,
        )
        result = engine.run()
        assert result.total_trades > 0

    def test_ml_path_with_signal_evaluator(self):
        from bot.backtest.engine import BacktestEngine
        from bot.learning.rl_signal_evaluator import RLSignalEvaluator
        candles = _make_5m_candles(3000)
        gen = MockMLSignalGenerator(direction="LONG", prob=0.7)
        evaluator = RLSignalEvaluator()
        engine = BacktestEngine(
            candles, ml_signal_generator=gen,
            signal_evaluator=evaluator,
            online_learning=True,
            initial_capital=10_000,
        )
        result = engine.run()
        assert result is not None

    def test_ml_path_no_direction_skips_signal(self):
        from bot.backtest.engine import BacktestEngine
        candles = _make_5m_candles(3000)
        gen = MockMLSignalGenerator(direction=None, prob=0.3)
        gen._direction = None
        engine = BacktestEngine(
            candles, ml_signal_generator=gen,
            initial_capital=10_000,
        )
        result = engine.run()
        assert result.total_trades == 0
