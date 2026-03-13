"""Risk engine — mandatory gate before every order."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from bot.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class RiskDecision:
    approved: bool
    reasons: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.approved:
            return "APPROVED"
        return "REJECTED: " + "; ".join(self.reasons)


class RiskEngine:
    """
    Validates every proposed trade against hard risk limits.
    Risk limits come from Settings and are NEVER modified by the learning system.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def approve(
        self,
        symbol: str,
        side: str,
        position_size_eur: float,
        signal_confidence: float,
        expected_roi_pct: float,
        portfolio_equity_eur: float,
        open_position_count: int,
        daily_loss_eur: float,
        current_drawdown_pct: float,
        taker_fee_pct: float = 0.25,
    ) -> RiskDecision:
        """Run all risk checks. Returns RiskDecision(approved, reasons)."""
        reasons: List[str] = []

        # 1. Signal confidence
        if signal_confidence < self.settings.min_signal_confidence:
            reasons.append(
                f"Signal confidence {signal_confidence:.2f} < {self.settings.min_signal_confidence}"
            )

        # 2. Drawdown circuit breaker
        if current_drawdown_pct >= self.settings.max_drawdown_pct:
            reasons.append(
                f"Drawdown {current_drawdown_pct:.1f}% ≥ limit {self.settings.max_drawdown_pct}%"
            )

        # 3. Daily loss limit
        if daily_loss_eur >= self.settings.max_daily_loss_eur:
            reasons.append(
                f"Daily loss €{daily_loss_eur:.2f} ≥ limit €{self.settings.max_daily_loss_eur:.2f}"
            )

        # 4. Position concentration
        if portfolio_equity_eur > 0:
            position_pct = (position_size_eur / portfolio_equity_eur) * 100.0
            if position_pct > self.settings.max_position_size_pct:
                reasons.append(
                    f"Position {position_pct:.1f}% > max {self.settings.max_position_size_pct}%"
                )

        # 5. Max open positions
        if side == "buy" and open_position_count >= self.settings.max_open_positions:
            reasons.append(
                f"Open positions {open_position_count} ≥ max {self.settings.max_open_positions}"
            )

        # 6. Minimum ROI potential (must beat 2× fee cost)
        min_roi = self.settings.min_trade_roi_pct + 2 * taker_fee_pct
        if expected_roi_pct < min_roi:
            reasons.append(
                f"Expected ROI {expected_roi_pct:.2f}% < required {min_roi:.2f}%"
            )

        approved = len(reasons) == 0
        if not approved:
            logger.info("RiskEngine REJECTED %s %s: %s", side, symbol, "; ".join(reasons))
        else:
            logger.debug("RiskEngine approved %s %s (confidence=%.2f)", side, symbol, signal_confidence)

        return RiskDecision(approved=approved, reasons=reasons)
