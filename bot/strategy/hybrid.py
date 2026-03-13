"""Primary hybrid strategy: technical composite + sentiment fusion."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from bot.indicators.composite import compute as compute_indicators, DEFAULT_WEIGHTS
from bot.strategy.base import BaseStrategy, MarketContext, Signal

logger = logging.getLogger(__name__)

# Sentiment weight in the final score (rest goes to technical)
DEFAULT_SENTIMENT_WEIGHT = 0.25
DEFAULT_ENTRY_THRESHOLD = 0.40


class HybridStrategy(BaseStrategy):
    """
    Combines technical indicator composite score with sentiment score.
    Final score = (1 - w_s) * technical + w_s * sentiment
    Emits LONG/SHORT if |score| > threshold AND ≥3 indicators agree.
    """

    name = "hybrid"

    def __init__(
        self,
        sentiment_weight: float = DEFAULT_SENTIMENT_WEIGHT,
        entry_threshold: float = DEFAULT_ENTRY_THRESHOLD,
        indicator_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.sentiment_weight = sentiment_weight
        self.entry_threshold = entry_threshold
        self.indicator_weights = indicator_weights or DEFAULT_WEIGHTS

    def generate_signal(self, ctx: MarketContext) -> Signal:
        if len(ctx.candles_5m) < 30:
            return Signal(
                symbol=ctx.symbol,
                direction="NEUTRAL",
                strength=0.0,
                strategy_name=self.name,
            )

        # Use ctx-level weights if provided (from DB optimizer)
        weights = ctx.indicator_weights or self.indicator_weights

        tech = compute_indicators(ctx.candles_5m, weights)
        technical_score = tech.technical_score
        sentiment = ctx.sentiment_score

        # Combine
        w_s = self.sentiment_weight
        final_score = (1 - w_s) * technical_score + w_s * sentiment

        # Higher timeframe alignment check (uses 1h for trend context)
        if len(ctx.candles_1h) >= 20:
            tech_1h = compute_indicators(ctx.candles_1h, weights)
            # If 1h disagrees strongly, dampen the signal
            if tech_1h.technical_score * final_score < -0.1:
                final_score *= 0.5
                logger.debug("1h timeframe dampened signal for %s", ctx.symbol)

        strength = min(abs(final_score), 1.0)
        direction = "NEUTRAL"
        if final_score > self.entry_threshold:
            direction = "LONG"
        elif final_score < -self.entry_threshold:
            direction = "SHORT"

        snapshot = {
            **tech.indicator_values,
            "confirming_count": tech.confirming_count,
            "final_score": final_score,
            "technical_score": technical_score,
            "sentiment_score": sentiment,
            "sentiment_weight": w_s,
        }

        logger.debug(
            "Hybrid %s: tech=%.3f sent=%.3f final=%.3f → %s (str=%.2f, confirm=%d)",
            ctx.symbol, technical_score, sentiment, final_score,
            direction, strength, tech.confirming_count,
        )

        return Signal(
            symbol=ctx.symbol,
            direction=direction,
            strength=strength,
            strategy_name=self.name,
            technical_score=technical_score,
            sentiment_score=sentiment,
            indicator_snapshot=snapshot,
        )

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "sentiment_weight": self.sentiment_weight,
            "entry_threshold": self.entry_threshold,
            "indicator_weights": self.indicator_weights,
        }

    def get_param_space(self) -> Dict[str, Any]:
        return {
            "sentiment_weight": {"type": "float", "low": 0.05, "high": 0.50},
            "entry_threshold": {"type": "float", "low": 0.25, "high": 0.65},
        }
