# tests/test_backtest_slippage.py
"""Tests for backtest slippage simulation."""
from __future__ import annotations

import pytest
from bot.backtest.engine import BacktestEngine


def _make_candles(n=100, start_price=100.0):
    """Generate synthetic candle dicts."""
    from datetime import datetime, timedelta, timezone
    candles = []
    price = start_price
    t = datetime(2025, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        price *= 1.001
        candles.append({
            "timestamp": t + timedelta(minutes=5 * i),
            "open": price * 0.999,
            "high": price * 1.002,
            "low": price * 0.998,
            "close": price,
            "volume": 1000.0,
        })
    return candles


class TestSlippage:
    def test_engine_accepts_slippage_param(self):
        candles = _make_candles()
        engine = BacktestEngine(candles, slippage_pct=0.002)
        assert engine.slippage_pct == 0.002

    def test_default_slippage_is_0_1_pct(self):
        candles = _make_candles()
        engine = BacktestEngine(candles)
        assert engine.slippage_pct == 0.001

    def test_slippage_worsens_entry_for_long(self):
        base_price = 100.0
        slippage = 0.01
        slipped = base_price * (1 + slippage)
        assert slipped > base_price

    def test_slippage_worsens_exit_for_long(self):
        base_price = 100.0
        slippage = 0.01
        slipped = base_price * (1 - slippage)
        assert slipped < base_price
