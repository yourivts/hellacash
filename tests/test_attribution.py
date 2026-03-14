"""Tests for performance attribution engine."""
from __future__ import annotations
from datetime import datetime, timezone
import pytest
from bot.analytics.attribution import (
    AttributionEngine, _classify_session, StrategyAttribution, AttributionReport,
)


class TestClassifySession:
    def test_asia(self):
        dt = datetime(2026, 1, 1, 3, 0, tzinfo=timezone.utc)
        assert _classify_session(dt) == "Asia"

    def test_europe(self):
        dt = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
        assert _classify_session(dt) == "Europe"

    def test_us(self):
        dt = datetime(2026, 1, 1, 16, 0, tzinfo=timezone.utc)
        assert _classify_session(dt) == "US"

    def test_overlap(self):
        dt = datetime(2026, 1, 1, 22, 0, tzinfo=timezone.utc)
        assert _classify_session(dt) == "Overlap/Off-hours"


class TestAttributionEngine:
    def test_compute_empty_trades(self):
        engine = AttributionEngine()
        report = engine.compute_from_trades([])
        assert isinstance(report, AttributionReport)
        assert report.total_pnl == 0.0
        assert report.by_strategy == []

    def test_compute_groups_by_strategy(self):
        trades = [
            _fake_trade("hybrid", 50.0), _fake_trade("hybrid", -20.0),
            _fake_trade("trend_following", 30.0),
        ]
        engine = AttributionEngine()
        report = engine.compute_from_trades(trades)
        assert len(report.by_strategy) == 2
        hybrid = next(s for s in report.by_strategy if s.strategy_name == "hybrid")
        assert hybrid.trade_count == 2
        assert hybrid.total_pnl == pytest.approx(30.0)


def _fake_trade(strategy: str, pnl: float):
    from types import SimpleNamespace
    return SimpleNamespace(
        strategy_name=strategy, net_pnl=pnl, roi_pct=pnl / 100,
        direction="LONG", market_regime="trending",
        created_at=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
    )
