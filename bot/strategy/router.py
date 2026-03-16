"""Strategy router: regime detection on 4h data, strategy dispatch."""
from __future__ import annotations

from enum import Enum
from typing import List

import pandas as pd

from bot.strategy.base import BaseStrategy
from bot.strategy.orderflow import OrderFlowStrategy
from bot.strategy.funding_contrarian import FundingContrarianStrategy
from bot.strategy.range_trading import RangeStrategy
from bot.strategy.squeeze import SqueezeStrategy


class Regime(str, Enum):
    QUIET = "quiet"
    VOLATILE = "volatile"
    TRENDING = "trending"
    RANGING = "ranging"
    NEUTRAL = "neutral"


def detect_regime(
    df_4h: pd.DataFrame,
    quiet_atr_threshold: float = 1.0,
    volatile_atr_threshold: float = 4.0,
    trending_adx_threshold: float = 25.0,
    ranging_adx_threshold: float = 20.0,
) -> Regime:
    """Detect market regime from 4h candle data.

    Evaluation order (first match wins):
    1. QUIET:    ATR% < quiet_atr_threshold
    2. VOLATILE: ATR% > volatile_atr_threshold
    3. TRENDING: ADX > trending_adx_threshold
    4. RANGING:  ADX < ranging_adx_threshold
    5. NEUTRAL:  everything else
    """
    if df_4h is None or len(df_4h) < 2:
        return Regime.NEUTRAL

    last = df_4h.iloc[-1]
    adx = float(last.get("adx", 20.0))

    close = float(last.get("close", 1.0))
    atr = float(last.get("atr", 0.0))
    atr_pct = (atr / close * 100.0) if close > 0 else 0.0

    if atr_pct < quiet_atr_threshold:
        return Regime.QUIET
    if atr_pct > volatile_atr_threshold:
        return Regime.VOLATILE
    if adx > trending_adx_threshold:
        return Regime.TRENDING
    if adx < ranging_adx_threshold:
        return Regime.RANGING
    return Regime.NEUTRAL


class StrategyRouter:
    def __init__(self) -> None:
        self._orderflow = OrderFlowStrategy()
        self._funding = FundingContrarianStrategy()
        self._range = RangeStrategy()
        self._squeeze = SqueezeStrategy()

    def get_strategies(self, regime: Regime) -> List[BaseStrategy]:
        if regime == Regime.QUIET:
            return []
        elif regime == Regime.VOLATILE:
            return [self._funding, self._squeeze]
        elif regime == Regime.TRENDING:
            return [self._orderflow, self._funding, self._squeeze]
        elif regime == Regime.RANGING:
            return [self._range, self._orderflow, self._funding]
        else:  # NEUTRAL
            return [self._funding, self._orderflow]

    def update_params(self, **kwargs) -> None:
        for strat in [self._orderflow, self._funding, self._range, self._squeeze]:
            for key, val in kwargs.items():
                if hasattr(strat, key):
                    setattr(strat, key, val)
