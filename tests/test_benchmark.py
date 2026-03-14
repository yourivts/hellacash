# tests/test_benchmark.py
"""Tests for benchmark comparison engine."""
from __future__ import annotations
import pytest
from bot.analytics.benchmark import BenchmarkEntry, compute_buy_hold_return


class TestBuyHoldReturn:
    def test_positive_return(self):
        entry = compute_buy_hold_return(
            symbol="BTC-EUR", start_price=40000, end_price=50000,
            initial_capital=10000,
        )
        assert isinstance(entry, BenchmarkEntry)
        assert entry.buy_hold_return_pct == pytest.approx(25.0)
        assert entry.buy_hold_pnl_eur == pytest.approx(2500.0)

    def test_negative_return(self):
        entry = compute_buy_hold_return(
            symbol="ETH-EUR", start_price=3000, end_price=2400,
            initial_capital=10000,
        )
        assert entry.buy_hold_return_pct == pytest.approx(-20.0)
        assert entry.buy_hold_pnl_eur == pytest.approx(-2000.0)

    def test_zero_start_price(self):
        entry = compute_buy_hold_return(
            symbol="X-EUR", start_price=0, end_price=100,
            initial_capital=10000,
        )
        assert entry.buy_hold_return_pct == 0.0
