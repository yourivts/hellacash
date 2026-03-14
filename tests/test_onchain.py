# tests/test_onchain.py
"""Tests for on-chain metrics provider."""
from __future__ import annotations
import pytest
from unittest.mock import patch, MagicMock
from bot.indicators.onchain import (
    OnchainProvider, BITVAVO_TO_BINANCE, _parse_funding_rate, _parse_open_interest,
)
from bot.strategy.base import SignalProvider


class TestSymbolMapping:
    def test_btc_maps(self):
        assert BITVAVO_TO_BINANCE["BTC-EUR"] == "BTCUSDT"

    def test_eth_maps(self):
        assert BITVAVO_TO_BINANCE["ETH-EUR"] == "ETHUSDT"

    def test_unknown_returns_none(self):
        assert BITVAVO_TO_BINANCE.get("FAKE-EUR") is None


class TestParsers:
    def test_parse_funding_positive(self):
        # Funding > 0.01% → bearish (-0.5)
        score = _parse_funding_rate(0.0002)  # 0.02%
        assert score == pytest.approx(-0.5)

    def test_parse_funding_negative(self):
        # Funding < -0.01% → bullish (+0.5)
        score = _parse_funding_rate(-0.0002)
        assert score == pytest.approx(0.5)

    def test_parse_funding_neutral(self):
        score = _parse_funding_rate(0.00005)  # 0.005%
        assert score == 0.0

    def test_parse_oi_rising_price_rising(self):
        score = _parse_open_interest(
            current_oi=1_000_000, prev_oi=900_000,
            current_price=50000, prev_price=49000,
        )
        assert score > 0  # trend confirmation

    def test_parse_oi_rising_price_falling(self):
        score = _parse_open_interest(
            current_oi=1_000_000, prev_oi=900_000,
            current_price=48000, prev_price=49000,
        )
        assert score < 0  # bearish pressure


class TestOnchainProvider:
    def test_implements_signal_provider(self):
        p = OnchainProvider()
        assert isinstance(p, SignalProvider)

    def test_score_returns_zero_when_no_data(self):
        p = OnchainProvider()
        assert p.score("BTC-EUR") == 0.0

    def test_is_available_false_initially(self):
        p = OnchainProvider()
        assert not p.is_available()

    def test_score_after_cache_populated(self):
        p = OnchainProvider()
        p._cache["BTC-EUR"] = {"funding": -0.5, "oi": 0.3}
        p._has_data = True
        score = p.score("BTC-EUR")
        assert -1.0 <= score <= 1.0
