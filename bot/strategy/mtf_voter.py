"""Multi-timeframe strategy voting — runs composite indicator on multiple TFs."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd

from bot.indicators.composite import compute as compute_indicators

logger = logging.getLogger(__name__)

BASE_WEIGHTS = {"15m": 0.15, "1h": 0.25, "4h": 0.35, "1d": 0.25}

REGIME_WEIGHTS: Dict[str, Dict[str, float]] = {
    "trending":  {"15m": 0.10, "1h": 0.20, "4h": 0.40, "1d": 0.30},
    "ranging":   {"15m": 0.30, "1h": 0.35, "4h": 0.25, "1d": 0.10},
    "volatile":  {"15m": 0.10, "1h": 0.25, "4h": 0.35, "1d": 0.30},
    "unknown":   BASE_WEIGHTS,
}

MIN_CANDLES = 30


@dataclass
class MTFResult:
    mtf_score: float = 0.0
    agreement_ratio: float = 0.0
    per_tf_scores: Dict[str, float] = field(default_factory=dict)


class MTFVoter:
    """Aggregate composite technical scores across multiple timeframes."""

    def vote(
        self,
        df_15m: pd.DataFrame,
        df_1h: pd.DataFrame,
        df_4h: pd.DataFrame,
        df_1d: pd.DataFrame,
        regime: str = "unknown",
        indicator_weights: Optional[Dict[str, float]] = None,
    ) -> MTFResult:
        frames = {"15m": df_15m, "1h": df_1h, "4h": df_4h, "1d": df_1d}
        tf_scores: Dict[str, float] = {}

        for tf, df in frames.items():
            if df is None or len(df) < MIN_CANDLES:
                continue
            try:
                result = compute_indicators(df, indicator_weights)
                tf_scores[tf] = result.technical_score
            except Exception as e:
                logger.warning("MTF %s failed: %s", tf, e)

        if not tf_scores:
            return MTFResult()

        weights = REGIME_WEIGHTS.get(regime, BASE_WEIGHTS)
        active = {tf: weights.get(tf, 0) for tf in tf_scores}
        total_w = sum(active.values())
        if total_w <= 0:
            return MTFResult()
        norm = {tf: w / total_w for tf, w in active.items()}

        mtf_score = sum(tf_scores[tf] * norm[tf] for tf in tf_scores)
        mtf_score = max(-1.0, min(1.0, mtf_score))

        # Agreement ratio: non-neutral TFs agreeing on majority direction
        non_neutral = {tf: s for tf, s in tf_scores.items() if abs(s) >= 0.15}
        if len(non_neutral) < 2:
            agreement_ratio = 0.0
        else:
            bullish = sum(1 for s in non_neutral.values() if s > 0)
            bearish = len(non_neutral) - bullish
            agreement_ratio = max(bullish, bearish) / len(non_neutral)

        if agreement_ratio < 0.5:
            mtf_score *= 0.5

        return MTFResult(
            mtf_score=mtf_score,
            agreement_ratio=agreement_ratio,
            per_tf_scores=tf_scores,
        )
