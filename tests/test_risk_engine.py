"""Tests for bot.risk.engine.RiskEngine."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from bot.config import Settings
from bot.risk.engine import RiskEngine, RiskDecision


def _make_settings(**overrides) -> Settings:
    defaults = {
        "bitvavo_api_key": "", "bitvavo_api_secret": "",
        "database_url": "postgresql+asyncpg://x:x@localhost/x",
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _approve_defaults(**overrides) -> dict:
    """Default args that pass all gates."""
    base = dict(
        symbol="BTC-EUR", side="buy", position_size_eur=1000.0,
        signal_confidence=0.75, expected_roi_pct=5.0,
        portfolio_equity_eur=10000.0, open_position_count=0,
        daily_loss_eur=0.0, current_drawdown_pct=0.0,
    )
    base.update(overrides)
    return base


class TestAllGatesPass:
    def test_all_gates_pass(self):
        engine = RiskEngine(_make_settings())
        decision = engine.approve(**_approve_defaults())
        assert decision.approved is True
        assert decision.reasons == []
        assert "signal_confidence" in decision.gate_details
        assert decision.gate_details["signal_confidence"]["passed"] is True


class TestSignalConfidenceGate:
    def test_low_confidence_rejected(self):
        engine = RiskEngine(_make_settings())
        decision = engine.approve(**_approve_defaults(signal_confidence=0.40))
        assert decision.approved is False
        assert decision.gate_details["signal_confidence"]["passed"] is False
        assert decision.gate_details["signal_confidence"]["value"] == 0.40


class TestDrawdownGate:
    def test_high_drawdown_rejected(self):
        engine = RiskEngine(_make_settings())
        decision = engine.approve(**_approve_defaults(current_drawdown_pct=9.0))
        assert decision.approved is False
        assert decision.gate_details["drawdown"]["passed"] is False


class TestDailyLossGate:
    def test_daily_loss_exceeded(self):
        engine = RiskEngine(_make_settings())
        decision = engine.approve(**_approve_defaults(daily_loss_eur=250.0))
        assert decision.approved is False
        assert decision.gate_details["daily_loss"]["passed"] is False


class TestPositionSizeGate:
    def test_oversized_position(self):
        engine = RiskEngine(_make_settings())
        decision = engine.approve(**_approve_defaults(position_size_eur=3000.0))
        assert decision.approved is False
        assert decision.gate_details["position_size"]["passed"] is False


class TestMaxPositionsGate:
    def test_too_many_positions(self):
        engine = RiskEngine(_make_settings())
        decision = engine.approve(**_approve_defaults(open_position_count=10))
        assert decision.approved is False
        assert decision.gate_details["max_positions"]["passed"] is False


class TestMinROIGate:
    def test_low_roi(self):
        engine = RiskEngine(_make_settings())
        decision = engine.approve(**_approve_defaults(expected_roi_pct=0.1))
        assert decision.approved is False
        assert decision.gate_details["min_roi"]["passed"] is False


class TestMultipleGatesFail:
    def test_multiple_gates_fail(self):
        engine = RiskEngine(_make_settings())
        decision = engine.approve(**_approve_defaults(
            signal_confidence=0.40, current_drawdown_pct=9.0
        ))
        assert decision.approved is False
        assert len(decision.reasons) == 2
        assert decision.gate_details["signal_confidence"]["passed"] is False
        assert decision.gate_details["drawdown"]["passed"] is False


class TestRingBuffer:
    def test_recent_decisions_stored(self):
        engine = RiskEngine(_make_settings())
        engine.approve(**_approve_defaults())
        engine.approve(**_approve_defaults(signal_confidence=0.40))
        decisions = engine.get_recent_decisions(limit=10)
        assert len(decisions) == 2
        assert decisions[0]["approved"] is False  # most recent first
        assert decisions[1]["approved"] is True


class TestCorrelationGuard:
    def test_reduces_size_at_3_correlated_positions(self):
        from bot.risk.engine import check_correlation_guard
        open_positions = [
            {"symbol": "BTC-EUR", "direction": "LONG"},
            {"symbol": "ETH-EUR", "direction": "LONG"},
            {"symbol": "SOL-EUR", "direction": "LONG"},
        ]
        multiplier = check_correlation_guard(new_direction="LONG", open_positions=open_positions)
        assert multiplier == 0.5

    def test_no_reduction_below_3(self):
        from bot.risk.engine import check_correlation_guard
        open_positions = [
            {"symbol": "BTC-EUR", "direction": "LONG"},
            {"symbol": "ETH-EUR", "direction": "LONG"},
        ]
        multiplier = check_correlation_guard(new_direction="LONG", open_positions=open_positions)
        assert multiplier == 1.0

    def test_opposite_direction_no_reduction(self):
        from bot.risk.engine import check_correlation_guard
        open_positions = [
            {"symbol": "BTC-EUR", "direction": "SHORT"},
            {"symbol": "ETH-EUR", "direction": "SHORT"},
            {"symbol": "SOL-EUR", "direction": "SHORT"},
        ]
        multiplier = check_correlation_guard(new_direction="LONG", open_positions=open_positions)
        assert multiplier == 1.0
