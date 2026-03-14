# tests/test_orderbook.py
"""Tests for order book analysis provider."""
from __future__ import annotations
import pytest
from bot.indicators.orderbook import (
    OrderBookProvider, _BookState, DepthAnalysis, PriceLevel, DepthBucket,
)
from bot.strategy.base import SignalProvider


class TestBookState:
    def test_apply_delta_add(self):
        bs = _BookState()
        bs.apply_delta("bid", 50000.0, 1.5)
        assert bs.bids[50000.0] == 1.5

    def test_apply_delta_remove(self):
        bs = _BookState()
        bs.apply_delta("bid", 50000.0, 1.5)
        bs.apply_delta("bid", 50000.0, 0)
        assert 50000.0 not in bs.bids

    def test_apply_delta_ask(self):
        bs = _BookState()
        bs.apply_delta("ask", 51000.0, 2.0)
        assert bs.asks[51000.0] == 2.0


class TestOrderBookProvider:
    def test_implements_signal_provider(self):
        p = OrderBookProvider()
        assert isinstance(p, SignalProvider)

    def test_imbalance_bullish(self):
        p = OrderBookProvider()
        # Lots of bids, few asks
        for i in range(25):
            p.update_level("BTC-EUR", "bid", 50000 - i * 10, 2.0)
            p.update_level("BTC-EUR", "ask", 50100 + i * 10, 0.5)
        score = p.score("BTC-EUR")
        assert score > 0  # bullish imbalance

    def test_imbalance_bearish(self):
        p = OrderBookProvider()
        for i in range(25):
            p.update_level("BTC-EUR", "bid", 50000 - i * 10, 0.5)
            p.update_level("BTC-EUR", "ask", 50100 + i * 10, 2.0)
        score = p.score("BTC-EUR")
        assert score < 0  # bearish imbalance

    def test_no_data_returns_zero(self):
        p = OrderBookProvider()
        assert p.score("BTC-EUR") == 0.0

    def test_analyze_returns_depth_analysis(self):
        p = OrderBookProvider()
        for i in range(25):
            p.update_level("BTC-EUR", "bid", 50000 - i * 10, 1.0 + (i == 5) * 5.0)
            p.update_level("BTC-EUR", "ask", 50100 + i * 10, 1.0)
        analysis = p.analyze("BTC-EUR")
        assert isinstance(analysis, DepthAnalysis)
        assert -1.0 <= analysis.imbalance <= 1.0
        assert analysis.spread_pct >= 0
