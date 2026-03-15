"""Tests for bot.risk.drawdown_guard.DrawdownGuard."""
from __future__ import annotations

from unittest.mock import patch
from datetime import date, datetime, timedelta, timezone

import pytest

from bot.risk.drawdown_guard import COOLDOWN_MINUTES, DrawdownGuard


class TestUpdateTracksPeakEquity:
    def test_update_tracks_peak_equity(self):
        guard = DrawdownGuard()
        guard.update(10_000.0)
        assert guard._peak_equity == 10_000.0
        guard.update(12_000.0)
        assert guard._peak_equity == 12_000.0
        # Peak should not decrease
        guard.update(11_000.0)
        assert guard._peak_equity == 12_000.0


class TestSoftDrawdown:
    def test_soft_drawdown_reduces_position_size(self):
        guard = DrawdownGuard(soft_drawdown_pct=3.0)
        guard.update(10_000.0)
        # 4% drawdown (above 3% soft limit)
        equity = 9_600.0
        assert guard.position_size_multiplier(equity) == 0.5

    def test_no_reduction_below_soft(self):
        guard = DrawdownGuard(soft_drawdown_pct=3.0)
        guard.update(10_000.0)
        equity = 9_800.0  # 2% drawdown
        assert guard.position_size_multiplier(equity) == 1.0


class TestHardDrawdown:
    def test_hard_drawdown_halts_trading(self):
        guard = DrawdownGuard(max_drawdown_pct=8.0)
        guard.update(10_000.0)
        equity = 9_100.0  # 9% drawdown
        allowed, reason = guard.is_trading_allowed(equity)
        assert allowed is False
        assert "hard limit" in reason.lower() or "exceeds" in reason.lower()


class TestDailyLossLimit:
    def test_daily_loss_limit_halts_trading(self):
        guard = DrawdownGuard(daily_loss_limit_eur=200.0)
        guard.update(10_000.0)
        guard.record_realized_loss(250.0)
        allowed, reason = guard.is_trading_allowed(10_000.0)
        assert allowed is False
        assert "daily loss" in reason.lower() or "limit" in reason.lower()

    def test_daily_loss_resets_at_midnight(self):
        guard = DrawdownGuard(daily_loss_limit_eur=200.0)
        guard.update(10_000.0)
        guard.record_realized_loss(250.0)
        assert guard._daily_loss == 250.0

        # Simulate a new day by patching date.today()
        tomorrow = date(2099, 1, 2)
        with patch("bot.risk.drawdown_guard.date") as mock_date:
            mock_date.today.return_value = tomorrow
            mock_date.side_effect = lambda *a, **k: date(*a, **k)
            guard.update(10_000.0)

        assert guard._daily_loss == 0.0


class TestResetHardHalt:
    def test_reset_hard_halt(self):
        guard = DrawdownGuard(max_drawdown_pct=8.0)
        guard.update(10_000.0)
        # Trigger hard halt
        guard.is_trading_allowed(9_100.0)
        assert guard._hard_halt is True

        # Even with full equity back, halt persists until reset
        allowed, _ = guard.is_trading_allowed(10_000.0)
        assert allowed is False

        guard.reset_hard_halt()
        allowed, _ = guard.is_trading_allowed(10_000.0)
        assert allowed is True


class TestPaperModeAutoRecovery:
    def test_paper_mode_resets_peak_after_cooldown(self):
        """Paper mode auto-resets peak equity when cooldown expires, even if drawdown > threshold."""
        guard = DrawdownGuard(max_drawdown_pct=8.0, paper_mode=True)
        guard.update(10_000.0)

        # Trigger hard halt (9% drawdown)
        allowed, _ = guard.is_trading_allowed(9_100.0)
        assert allowed is False

        # Simulate cooldown expiry
        guard._halt_triggered_at = datetime.now(timezone.utc) - timedelta(minutes=COOLDOWN_MINUTES + 1)

        # Drawdown is still 21% but paper mode should auto-recover
        allowed, _ = guard.is_trading_allowed(7_900.0)
        assert allowed is True
        assert guard._peak_equity == 7_900.0

    def test_live_mode_stays_halted_with_high_drawdown(self):
        """Live mode stays halted if drawdown hasn't recovered, even after cooldown."""
        guard = DrawdownGuard(max_drawdown_pct=8.0, paper_mode=False)
        guard.update(10_000.0)

        # Trigger hard halt
        guard.is_trading_allowed(9_100.0)

        # Simulate cooldown expiry
        guard._halt_triggered_at = datetime.now(timezone.utc) - timedelta(minutes=COOLDOWN_MINUTES + 1)

        # Still 21% drawdown — live mode should NOT resume
        allowed, reason = guard.is_trading_allowed(7_900.0)
        assert allowed is False
        assert "emergency halt" in reason.lower() or "drawdown" in reason.lower()

    def test_paper_mode_still_enforces_cooldown(self):
        """Paper mode doesn't skip the cooldown period itself."""
        guard = DrawdownGuard(max_drawdown_pct=8.0, paper_mode=True)
        guard.update(10_000.0)

        # Trigger hard halt
        guard.is_trading_allowed(9_100.0)

        # Cooldown NOT expired yet
        guard._halt_triggered_at = datetime.now(timezone.utc) - timedelta(minutes=5)

        # Should still be halted during cooldown
        allowed, _ = guard.is_trading_allowed(7_900.0)
        assert allowed is False
