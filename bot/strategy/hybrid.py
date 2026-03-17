"""Primary hybrid strategy: 4-component fusion (MTF technical + sentiment + on-chain + order book)."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Set

from bot.config import get_settings
from bot.strategy.base import BaseStrategy, MarketContext, Signal
from bot.strategy.mtf_voter import MTFVoter

logger = logging.getLogger(__name__)

DEFAULT_ENTRY_THRESHOLD = 0.50


def redistribute_weights(
    weights: Dict[str, float], disabled: Set[str]
) -> Dict[str, float]:
    active = {k: v for k, v in weights.items() if k not in disabled}
    total = sum(active.values())
    if total <= 0:
        return active
    return {k: v / total for k, v in active.items()}


class HybridStrategy(BaseStrategy):
    """
    Combines MTF technical composite, sentiment, on-chain, and order book scores.
    final_score = w_tech * mtf + w_sent * sentiment + w_onchain * onchain + w_book * orderbook
    """

    name = "hybrid"

    def __init__(
        self,
        sentiment_weight: float = 0.20,
        entry_threshold: float = DEFAULT_ENTRY_THRESHOLD,
        indicator_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.sentiment_weight = sentiment_weight
        self.entry_threshold = entry_threshold
        self.indicator_weights = indicator_weights
        self._mtf_voter = MTFVoter()

    def generate_signal(self, ctx: MarketContext) -> Signal:
        if len(ctx.candles_5m) < 30:
            return Signal(
                symbol=ctx.symbol, direction="NEUTRAL",
                strength=0.0, strategy_name=self.name,
            )

        settings = get_settings()

        # MTF voting
        mtf_result = self._mtf_voter.vote(
            df_15m=ctx.candles_15m if ctx.candles_15m is not None else ctx.candles_5m,
            df_1h=ctx.candles_1h,
            df_4h=ctx.candles_4h if ctx.candles_4h is not None else ctx.candles_1h,
            df_1d=ctx.candles_1d if ctx.candles_1d is not None else ctx.candles_1h,
            regime=ctx.market_regime,
            indicator_weights=ctx.indicator_weights or self.indicator_weights,
        )

        tech_score = mtf_result.mtf_score
        sent_score = ctx.sentiment_score
        onchain_score = ctx.onchain_score
        book_score = ctx.orderbook_imbalance

        base_w = {
            "tech": settings.mtf_tech_weight,
            "sent": settings.mtf_sent_weight,
            "onchain": settings.mtf_onchain_weight,
            "book": settings.mtf_book_weight,
        }
        disabled = set()
        if not settings.onchain_enabled:
            disabled.add("onchain")
        if not settings.orderbook_enabled:
            disabled.add("book")
        w = redistribute_weights(base_w, disabled)

        final_score = (
            w.get("tech", 0) * tech_score
            + w.get("sent", 0) * sent_score
            + w.get("onchain", 0) * onchain_score
            + w.get("book", 0) * book_score
        )
        final_score = max(-1.0, min(1.0, final_score))

        strength = min(abs(final_score), 1.0)
        direction = "NEUTRAL"
        if final_score > self.entry_threshold:
            direction = "LONG"
        elif final_score < -self.entry_threshold:
            direction = "SHORT"

        confirming = sum(
            1 for s in mtf_result.per_tf_scores.values()
            if s * final_score > 0
        )

        snapshot = {
            "confirming_count": confirming,
            "final_score": final_score,
            "technical_score": tech_score,
            "sentiment_score": sent_score,
            "onchain_score": onchain_score,
            "orderbook_imbalance": book_score,
            "mtf_scores": mtf_result.per_tf_scores,
            "mtf_agreement": mtf_result.agreement_ratio,
            "market_regime": ctx.market_regime,
        }

        return Signal(
            symbol=ctx.symbol, direction=direction, strength=strength,
            strategy_name=self.name, technical_score=tech_score,
            sentiment_score=sent_score, indicator_snapshot=snapshot,
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
