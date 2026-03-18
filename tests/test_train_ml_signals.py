"""Tests for ML signal training pipeline."""
import numpy as np
import pandas as pd
import pytest


def _make_5m_candles(bars: int = 3000) -> pd.DataFrame:
    np.random.seed(42)
    timestamps = pd.date_range("2024-01-01", periods=bars, freq="5min")
    base = 50000.0
    returns = np.random.randn(bars) * 0.001
    close = base * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.randn(bars)) * 0.002)
    low = close * (1 - np.abs(np.random.randn(bars)) * 0.002)
    open_ = close * (1 + np.random.randn(bars) * 0.001)
    volume = np.random.exponential(100, bars)
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume,
    }, index=timestamps)


class TestComputeLabels:
    def test_label_shape(self):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from scripts.train_ml_signals import compute_labels
        df = _make_5m_candles(3000)
        labels = compute_labels(df)
        assert labels.shape[1] == 12
        assert labels.shape[0] == len(df)

    def test_labels_binary(self):
        from scripts.train_ml_signals import compute_labels
        df = _make_5m_candles(3000)
        labels = compute_labels(df)
        assert set(np.unique(labels[~np.isnan(labels)])).issubset({0.0, 1.0})

    def test_future_bars_have_nan(self):
        from scripts.train_ml_signals import compute_labels
        df = _make_5m_candles(3000)
        labels = compute_labels(df)
        assert np.all(np.isnan(labels[-864:, -2:]))


class TestBuildTrainingTargets:
    def test_lstm_targets_shape(self):
        from scripts.train_ml_signals import build_lstm_targets
        df = _make_5m_candles(3000)
        targets = build_lstm_targets(df)
        assert targets.shape[1] == 60

    def test_targets_aligned_with_candles(self):
        from scripts.train_ml_signals import build_lstm_targets
        df = _make_5m_candles(3000)
        targets = build_lstm_targets(df)
        assert targets.shape[0] == len(df) - 12
