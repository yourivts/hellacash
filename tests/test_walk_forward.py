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


from bot.learning.walk_forward import WalkForwardOptimizer, WFWindow, WFResult


class TestWalkForwardOptimizer:
    def test_window_generation(self):
        wfo = WalkForwardOptimizer()
        windows = wfo._generate_windows(total_days=120)
        assert len(windows) >= 3

    def test_adoption_criteria_rejects_poor_results(self):
        wfo = WalkForwardOptimizer()
        windows = [
            WFWindow(sharpe=-0.5, pnl=-100, params={}),
            WFWindow(sharpe=0.3, pnl=-50, params={}),
            WFWindow(sharpe=0.2, pnl=-20, params={}),
        ]
        assert not wfo._should_adopt(windows)

    def test_adoption_criteria_accepts_good_results(self):
        wfo = WalkForwardOptimizer()
        windows = [
            WFWindow(sharpe=1.2, pnl=200, params={"sentiment_weight": 0.3}),
            WFWindow(sharpe=0.8, pnl=150, params={"sentiment_weight": 0.3}),
            WFWindow(sharpe=1.0, pnl=180, params={"sentiment_weight": 0.3}),
        ]
        assert wfo._should_adopt(windows)
