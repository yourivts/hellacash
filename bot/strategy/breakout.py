"""Volume-confirmed price breakout strategy."""
from __future__ import annotations

from bot.indicators.volume import volume_surge_ratio
from bot.strategy.base import BaseStrategy, MarketContext, Signal


class BreakoutStrategy(BaseStrategy):
    name = "breakout"

    def __init__(
        self,
        lookback: int = 20,
        volume_surge_min: float = 2.0,
    ) -> None:
        self.lookback = lookback
        self.volume_surge_min = volume_surge_min

    def generate_signal(self, ctx: MarketContext) -> Signal:
        df = ctx.candles_5m
        if len(df) < self.lookback + 5:
            return Signal(ctx.symbol, "NEUTRAL", 0.0, self.name)

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        recent_high = high.iloc[-(self.lookback + 1):-1].max()
        recent_low = low.iloc[-(self.lookback + 1):-1].min()
        current = close.iloc[-1]
        vsr = volume_surge_ratio(volume).iloc[-1]

        direction = "NEUTRAL"
        strength = 0.0

        if current > recent_high and vsr >= self.volume_surge_min:
            direction = "LONG"
            strength = min(vsr / 4.0, 1.0)
        elif current < recent_low and vsr >= self.volume_surge_min:
            direction = "SHORT"
            strength = min(vsr / 4.0, 1.0)

        return Signal(
            symbol=ctx.symbol,
            direction=direction,
            strength=strength,
            strategy_name=self.name,
            technical_score=strength if direction == "LONG" else -strength,
            indicator_snapshot={
                "recent_high": recent_high,
                "recent_low": recent_low,
                "volume_surge": vsr,
                "confirming_count": 2 if direction != "NEUTRAL" else 0,
            },
        )
