"""Tracks per-indicator accuracy and updates weights accordingly."""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List

logger = logging.getLogger(__name__)

INDICATOR_KEYS = [
    "rsi", "macd", "bollinger", "ema_trend", "supertrend", "volume", "cci"
]


class SignalEvaluator:
    """
    After each closed trade, updates per-indicator accuracy counters.
    Periodically re-computes indicator weights proportional to accuracy.
    """

    def __init__(self) -> None:
        self._correct: Dict[str, int] = defaultdict(int)
        self._total: Dict[str, int] = defaultdict(int)

    def record_trade(
        self,
        indicator_breakdown: Dict[str, float],
        trade_direction: str,  # "LONG" or "SHORT"
        was_profitable: bool,
    ) -> None:
        """
        For each indicator that voted in the trade direction:
        - increment total count
        - if trade was profitable, increment correct count
        """
        direction_sign = 1.0 if trade_direction == "LONG" else -1.0

        for indicator in INDICATOR_KEYS:
            vote = indicator_breakdown.get(indicator, 0.0)
            if vote * direction_sign > 0.05:  # indicator agreed with trade direction
                self._total[indicator] += 1
                if was_profitable:
                    self._correct[indicator] += 1

    def compute_weights(self) -> Dict[str, float]:
        """
        Compute new indicator weights proportional to accuracy.
        Falls back to default weights if not enough data.
        """
        from bot.indicators.composite import DEFAULT_WEIGHTS

        min_samples = 10
        accuracies: Dict[str, float] = {}

        for ind in INDICATOR_KEYS:
            total = self._total.get(ind, 0)
            if total >= min_samples:
                accuracies[ind] = self._correct.get(ind, 0) / total
            else:
                # Keep default weight if insufficient data
                accuracies[ind] = DEFAULT_WEIGHTS.get(ind, 0.1)

        # Normalize to sum to 1.0
        total_acc = sum(accuracies.values())
        if total_acc <= 0:
            return dict(DEFAULT_WEIGHTS)

        weights = {k: v / total_acc for k, v in accuracies.items()}

        logger.info(
            "Updated indicator weights: %s",
            {k: f"{v:.3f}" for k, v in weights.items()},
        )
        return weights

    def accuracy_report(self) -> Dict[str, str]:
        return {
            ind: f"{self._correct.get(ind, 0)}/{self._total.get(ind, 0)}"
            for ind in INDICATOR_KEYS
        }
