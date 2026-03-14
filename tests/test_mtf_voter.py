"""Tests for multi-timeframe voter and SignalProvider protocol."""
from __future__ import annotations
import pytest
from bot.strategy.base import SignalProvider


class _DummyProvider:
    name = "dummy"
    def score(self, symbol: str, **kwargs) -> float:
        return 0.5
    def is_available(self) -> bool:
        return True


class _BadProvider:
    name = "bad"


class TestSignalProviderProtocol:
    def test_valid_provider_matches_protocol(self):
        p = _DummyProvider()
        assert isinstance(p, SignalProvider)

    def test_invalid_provider_does_not_match(self):
        p = _BadProvider()
        assert not isinstance(p, SignalProvider)
