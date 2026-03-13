"""Tests for bot.risk.position_sizer.kelly_size."""
from __future__ import annotations

import pytest

from bot.risk.position_sizer import kelly_size


class TestKellySizeBasic:
    def test_kelly_size_basic(self):
        size = kelly_size(
            win_rate=0.6,
            avg_win_pct=0.04,
            avg_loss_pct=0.02,
            portfolio_eur=10_000.0,
        )
        assert size > 0
        assert size <= 10_000.0


class TestKellySizeRespectsMaxPosition:
    def test_kelly_size_respects_max_position(self):
        size = kelly_size(
            win_rate=0.9,
            avg_win_pct=0.10,
            avg_loss_pct=0.01,
            portfolio_eur=10_000.0,
            max_position_pct=5.0,
        )
        # Should not exceed 5% of portfolio = 500 EUR
        assert size <= 500.0


class TestKellySizeReducesDuringDrawdown:
    def test_kelly_size_reduces_during_drawdown(self):
        size_normal = kelly_size(
            win_rate=0.6,
            avg_win_pct=0.04,
            avg_loss_pct=0.02,
            portfolio_eur=10_000.0,
            current_drawdown_pct=0.0,
        )
        size_dd = kelly_size(
            win_rate=0.6,
            avg_win_pct=0.04,
            avg_loss_pct=0.02,
            portfolio_eur=10_000.0,
            current_drawdown_pct=6.0,
        )
        assert size_dd < size_normal


class TestKellySizeZeroWhenLosing:
    def test_kelly_size_zero_when_losing(self):
        """When win_rate is very low, Kelly formula yields 0 edge."""
        size = kelly_size(
            win_rate=0.2,
            avg_win_pct=0.02,
            avg_loss_pct=0.04,
            portfolio_eur=10_000.0,
        )
        # Kelly formula gives negative edge -> clamped to 0
        assert size == 0.0
