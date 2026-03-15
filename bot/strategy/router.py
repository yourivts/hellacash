"""Market regime detection and strategy selector."""
from __future__ import annotations

import logging
from typing import List

import pandas as pd

from bot.indicators.trend import adx
from bot.indicators.volatility import atr
from bot.strategy.base import BaseStrategy
from bot.strategy.breakout import BreakoutStrategy
from bot.strategy.hybrid import HybridStrategy
from bot.strategy.range_trading import RangeStrategy
from bot.strategy.trend_following import TrendFollowingStrategy

logger = logging.getLogger(__name__)


class Regime:
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


def detect_regime(df_1h: pd.DataFrame) -> str:
    """
    Classify market regime from 1h candle data.
    Returns one of: 'trending', 'ranging', 'volatile', 'unknown'
    """
    if len(df_1h) < 30:
        return Regime.UNKNOWN

    close = df_1h["close"]
    high = df_1h["high"]
    low = df_1h["low"]

    try:
        a = adx(high, low, close).iloc[-1]
        current_atr = atr(high, low, close).iloc[-1]
        price = close.iloc[-1]
        atr_pct = (current_atr / price) * 100.0 if price > 0 else 0.0

        # High ATR relative to price = volatile
        if atr_pct > 3.0:
            return Regime.VOLATILE

        # Strong ADX = trending
        if a > 25:
            return Regime.TRENDING

        # Weak ADX = ranging
        if a < 20:
            return Regime.RANGING

        return Regime.UNKNOWN
    except Exception as e:
        logger.warning("Regime detection failed: %s", e)
        return Regime.UNKNOWN


class StrategyRouter:
    """Selects which strategies to run based on detected market regime."""

    def __init__(self) -> None:
        self._hybrid = HybridStrategy()
        self._trend = TrendFollowingStrategy()
        self._range = RangeStrategy()
        self._breakout = BreakoutStrategy()

    def get_strategies(self, regime: str) -> List[BaseStrategy]:
        if regime == Regime.TRENDING:
            return [self._hybrid, self._trend, self._breakout]
        elif regime == Regime.RANGING:
            return [self._hybrid, self._range]
        elif regime == Regime.VOLATILE:
            return [self._hybrid]  # only hybrid in volatile markets
        else:
            return [self._hybrid]

    def position_size_modifier(self, regime: str) -> float:
        """Reduce position sizes in volatile/ranging markets."""
        if regime == Regime.VOLATILE:
            return 0.3
        if regime == Regime.RANGING:
            return 0.5
        return 1.0

    def update_hybrid_params(
        self,
        sentiment_weight: float,
        entry_threshold: float,
        indicator_weights: dict,
    ) -> None:
        """Called by learning optimizer to update hybrid params."""
        self._hybrid.sentiment_weight = sentiment_weight
        self._hybrid.entry_threshold = entry_threshold
        self._hybrid.indicator_weights = indicator_weights
