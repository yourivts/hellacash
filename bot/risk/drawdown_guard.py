"""Portfolio-level circuit breaker."""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class DrawdownGuard:
    """
    Tracks portfolio drawdown and enforces circuit breakers.

    Levels:
        SOFT  (>3%):  reduce new position sizes by 50%
        HARD  (>8%):  halt all new trades
        DAILY (loss > limit): halt trading until next day
    """

    def __init__(
        self,
        max_drawdown_pct: float = 8.0,
        soft_drawdown_pct: float = 3.0,
        daily_loss_limit_eur: float = 200.0,
    ) -> None:
        self.max_drawdown_pct = max_drawdown_pct
        self.soft_drawdown_pct = soft_drawdown_pct
        self.daily_loss_limit_eur = daily_loss_limit_eur

        self._peak_equity: float = 0.0
        self._daily_loss: float = 0.0
        self._daily_loss_date: date = date.today()
        self._hard_halt: bool = False

    def update(self, current_equity: float) -> None:
        """Call on every portfolio snapshot."""
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

        # Reset daily loss counter at midnight
        today = date.today()
        if today != self._daily_loss_date:
            self._daily_loss = 0.0
            self._daily_loss_date = today

    def record_realized_loss(self, loss_eur: float) -> None:
        """Call when a trade closes at a loss (loss_eur should be positive)."""
        self._daily_loss += loss_eur

    def current_drawdown_pct(self, current_equity: float) -> float:
        if self._peak_equity <= 0:
            return 0.0
        return max(0.0, (self._peak_equity - current_equity) / self._peak_equity * 100.0)

    def is_trading_allowed(self, current_equity: float) -> tuple[bool, str]:
        """Returns (allowed, reason). reason is '' if allowed."""
        dd = self.current_drawdown_pct(current_equity)

        if self._hard_halt:
            return False, "Emergency halt active"

        if dd >= self.max_drawdown_pct:
            self._hard_halt = True
            reason = f"Drawdown {dd:.1f}% exceeds hard limit {self.max_drawdown_pct}%"
            logger.warning("CIRCUIT BREAKER: %s", reason)
            return False, reason

        if self._daily_loss >= self.daily_loss_limit_eur:
            reason = f"Daily loss €{self._daily_loss:.2f} exceeds limit €{self.daily_loss_limit_eur:.2f}"
            logger.warning("DAILY LOSS LIMIT: %s", reason)
            return False, reason

        return True, ""

    def position_size_multiplier(self, current_equity: float) -> float:
        """Returns 0.5 if in soft halt zone, else 1.0."""
        dd = self.current_drawdown_pct(current_equity)
        if dd >= self.soft_drawdown_pct:
            return 0.5
        return 1.0

    def reset_hard_halt(self) -> None:
        """Manual override to resume trading after investigation."""
        self._hard_halt = False
        logger.info("Hard halt reset manually")
