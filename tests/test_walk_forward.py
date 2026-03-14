# tests/test_walk_forward.py
"""Tests for walk-forward optimizer."""
from __future__ import annotations
import pytest
from bot.backtest.engine import BacktestEngine


class TestBacktestParameterization:
    def test_accepts_strategy_params(self):
        engine = BacktestEngine(
            candles=[],
            strategy_params={"sentiment_weight": 0.30, "entry_threshold": 0.45},
        )
        # Should not raise; params applied to router
        assert engine._router is not None
