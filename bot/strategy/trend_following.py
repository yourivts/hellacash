"""EMA crossover + MACD confirmation trend-following strategy."""
from __future__ import annotations

import logging

from bot.indicators.trend import ema, macd, adx
from bot.strategy.base import BaseStrategy, MarketContext, Signal

logger = logging.getLogger(__name__)


class TrendFollowingStrategy(BaseStrategy):
    name = "trend_following"

    def __init__(self, fast_ema: int = 20, slow_ema: int = 50, adx_threshold: float = 25.0) -> None:
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.adx_threshold = adx_threshold

    def generate_signal(self, ctx: MarketContext) -> Signal:
        df = ctx.candles_5m
        if len(df) < self.slow_ema + 5:
            return Signal(ctx.symbol, "NEUTRAL", 0.0, self.name)

        close = df["close"]
        high = df["high"]
        low = df["low"]

        fast = ema(close, self.fast_ema)
        slow = ema(close, self.slow_ema)
        m = macd(close)
        a = adx(high, low, close)

        # EMA crossover
        cross_up = fast.iloc[-1] > slow.iloc[-1] and fast.iloc[-2] <= slow.iloc[-2]
        cross_down = fast.iloc[-1] < slow.iloc[-1] and fast.iloc[-2] >= slow.iloc[-2]
        trending = a.iloc[-1] > self.adx_threshold

        # MACD confirmation
        macd_bull = m["histogram"].iloc[-1] > 0
        macd_bear = m["histogram"].iloc[-1] < 0

        direction = "NEUTRAL"
        strength = 0.0
        if cross_up and macd_bull and trending:
            direction = "LONG"
            strength = min(a.iloc[-1] / 50.0, 1.0)
        elif cross_down and macd_bear and trending:
            direction = "SHORT"
            strength = min(a.iloc[-1] / 50.0, 1.0)

        return Signal(
            symbol=ctx.symbol,
            direction=direction,
            strength=strength,
            strategy_name=self.name,
            technical_score=strength if direction == "LONG" else -strength,
            indicator_snapshot={
                "fast_ema": fast.iloc[-1],
                "slow_ema": slow.iloc[-1],
                "adx": a.iloc[-1],
                "macd_hist": m["histogram"].iloc[-1],
                "confirming_count": 3 if direction != "NEUTRAL" else 0,
            },
        )
