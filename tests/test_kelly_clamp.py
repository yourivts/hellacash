# tests/test_kelly_clamp.py
"""Tests for Kelly sizing edge cases."""
from __future__ import annotations

import pytest

from bot.risk.position_sizer import kelly_size


class TestKellyClamp:
    def test_zero_win_rate_uses_1pct_fallback(self):
        size = kelly_size(
            win_rate=0.0, avg_win_pct=0.0, avg_loss_pct=0.01,
            portfolio_eur=10000.0,
        )
        # Should be 1% of 10000 = 100
        assert size == pytest.approx(100.0, abs=1.0)

    def test_negative_avg_loss_uses_fallback(self):
        size = kelly_size(
            win_rate=0.5, avg_win_pct=0.03, avg_loss_pct=0.0,
            portfolio_eur=10000.0,
        )
        assert size == pytest.approx(100.0, abs=1.0)

    def test_result_never_negative(self):
        size = kelly_size(
            win_rate=0.1, avg_win_pct=0.001, avg_loss_pct=0.05,
            portfolio_eur=10000.0,
        )
        assert size >= 0.0

    def test_result_capped_at_max_position_pct(self):
        size = kelly_size(
            win_rate=0.9, avg_win_pct=0.10, avg_loss_pct=0.01,
            portfolio_eur=10000.0, max_position_pct=5.0,
        )
        assert size <= 500.0  # 5% of 10k
