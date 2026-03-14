# tests/test_chart_snapshot.py
"""Tests for chart snapshot generation."""
from __future__ import annotations
import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from bot.charts.snapshot import generate_entry_chart, generate_exit_chart


def _ohlcv(n=100):
    np.random.seed(42)
    c = np.cumsum(np.random.normal(0, 1, n)) + 100
    idx = pd.date_range("2026-01-01", periods=n, freq="5min")
    return pd.DataFrame({
        "open": c * 0.999, "high": c * 1.005,
        "low": c * 0.995, "close": c,
        "volume": np.random.uniform(100, 500, n),
    }, index=idx)


class TestChartGeneration:
    def test_entry_chart_creates_png(self):
        df = _ohlcv(100)
        with tempfile.TemporaryDirectory() as d:
            path = generate_entry_chart(
                df, entry_price=100.0, symbol="BTC-EUR",
                trade_id=1, output_dir=d,
            )
            assert os.path.exists(path)
            assert path.endswith(".png")
            assert os.path.getsize(path) > 1000

    def test_exit_chart_creates_png(self):
        df = _ohlcv(100)
        with tempfile.TemporaryDirectory() as d:
            path = generate_exit_chart(
                df, entry_price=100.0, exit_price=105.0,
                stop_loss=95.0, take_profit=110.0,
                symbol="BTC-EUR", trade_id=1, output_dir=d,
            )
            assert os.path.exists(path)
            assert path.endswith(".png")
