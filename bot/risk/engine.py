"""Risk engine — mandatory gate before every order."""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List

from bot.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class RiskDecision:
    approved: bool
    reasons: List[str] = field(default_factory=list)
    gate_details: Dict[str, Dict[str, Any]] = field(default_factory=dict)

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
        self._recent_decisions: deque = deque(maxlen=100)

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
        """Run all risk checks. Returns RiskDecision(approved, reasons, gate_details)."""
        reasons: List[str] = []
        gate_details: Dict[str, Dict[str, Any]] = {}

        # 1. Signal confidence
        conf_threshold = self.settings.min_signal_confidence
        conf_passed = signal_confidence >= conf_threshold
        gate_details["signal_confidence"] = {
            "value": signal_confidence,
            "threshold": conf_threshold,
            "passed": conf_passed,
        }
        if not conf_passed:
            reasons.append(
                f"Signal confidence {signal_confidence:.2f} < {conf_threshold}"
            )

        # 2. Drawdown circuit breaker
        drawdown_threshold = self.settings.max_drawdown_pct
        drawdown_passed = current_drawdown_pct < drawdown_threshold
        gate_details["drawdown"] = {
            "value": current_drawdown_pct,
            "threshold": drawdown_threshold,
            "passed": drawdown_passed,
        }
        if not drawdown_passed:
            reasons.append(
                f"Drawdown {current_drawdown_pct:.1f}% ≥ limit {drawdown_threshold}%"
            )

        # 3. Daily loss limit
        daily_loss_threshold = self.settings.max_daily_loss_eur
        daily_loss_passed = daily_loss_eur < daily_loss_threshold
        gate_details["daily_loss"] = {
            "value": daily_loss_eur,
            "threshold": daily_loss_threshold,
            "passed": daily_loss_passed,
        }
        if not daily_loss_passed:
            reasons.append(
                f"Daily loss €{daily_loss_eur:.2f} ≥ limit €{daily_loss_threshold:.2f}"
            )

        # 4. Position concentration
        position_pct = (
            (position_size_eur / portfolio_equity_eur) * 100.0
            if portfolio_equity_eur > 0
            else 0.0
        )
        pos_size_threshold = self.settings.max_position_size_pct
        pos_size_passed = position_pct <= pos_size_threshold
        gate_details["position_size"] = {
            "value": position_pct,
            "threshold": pos_size_threshold,
            "passed": pos_size_passed,
        }
        if not pos_size_passed:
            reasons.append(
                f"Position {position_pct:.1f}% > max {pos_size_threshold}%"
            )

        # 5. Max open positions
        max_pos_threshold = self.settings.max_open_positions
        max_pos_passed = not (side == "buy" and open_position_count >= max_pos_threshold)
        gate_details["max_positions"] = {
            "value": open_position_count,
            "threshold": max_pos_threshold,
            "passed": max_pos_passed,
        }
        if not max_pos_passed:
            reasons.append(
                f"Open positions {open_position_count} ≥ max {max_pos_threshold}"
            )

        # 6. Minimum ROI potential (must beat 2× fee cost)
        min_roi = self.settings.min_trade_roi_pct + 2 * taker_fee_pct
        min_roi_passed = expected_roi_pct >= min_roi
        gate_details["min_roi"] = {
            "value": expected_roi_pct,
            "threshold": min_roi,
            "passed": min_roi_passed,
        }
        if not min_roi_passed:
            reasons.append(
                f"Expected ROI {expected_roi_pct:.2f}% < required {min_roi:.2f}%"
            )

        approved = len(reasons) == 0
        if not approved:
            logger.info("RiskEngine REJECTED %s %s: %s", side, symbol, "; ".join(reasons))
        else:
            logger.debug("RiskEngine approved %s %s (confidence=%.2f)", side, symbol, signal_confidence)

        decision = RiskDecision(approved=approved, reasons=reasons, gate_details=gate_details)

        self._recent_decisions.appendleft({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "side": side,
            "approved": approved,
            "reasons": reasons,
            "gate_details": gate_details,
        })

        return decision

    def get_recent_decisions(self, limit: int = 20) -> list:
        """Return the most recent risk decisions, newest first."""
        return list(self._recent_decisions)[:limit]


def check_correlation_guard(
    new_direction: str,
    open_positions: list,
    threshold: int = 3,
) -> float:
    """Returns position size multiplier based on directional concentration.

    If 3+ open positions are in the same direction as the new trade,
    reduce new position size by 50%.
    """
    same_direction = sum(
        1 for p in open_positions
        if p.get("direction") == new_direction
    )
    if same_direction >= threshold:
        return 0.5
    return 1.0
