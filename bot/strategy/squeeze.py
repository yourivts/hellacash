"""Volatility Squeeze strategy — BB compression followed by expansion."""
from __future__ import annotations

from bot.strategy.base import BaseStrategy, MarketContext, Signal


class SqueezeStrategy(BaseStrategy):
    name = "squeeze"

    COMPRESSION_THRESHOLD = 0.02
    EXPANSION_THRESHOLD = 0.03
    MIN_VOLUME_SURGE = 1.5

    def __init__(self):
        self.last_tp_distance: float = 0.0

    def evaluate_1h(self, bb_bandwidth_prev, bb_bandwidth, price, bb_upper,
                    bb_lower, volume_surge, ema50_slope) -> tuple[str, float]:
        self.last_tp_distance = 0.0

        if bb_bandwidth_prev >= self.COMPRESSION_THRESHOLD:
            return "NEUTRAL", 0.0
        if bb_bandwidth < self.EXPANSION_THRESHOLD:
            return "NEUTRAL", 0.0
        if volume_surge < self.MIN_VOLUME_SURGE:
            return "NEUTRAL", 0.0

        squeeze_width = bb_upper - bb_lower
        if squeeze_width <= 0:
            return "NEUTRAL", 0.0

        if price > bb_upper:
            direction = "LONG"
            if ema50_slope <= 0:
                return "NEUTRAL", 0.0
        elif price < bb_lower:
            direction = "SHORT"
            if ema50_slope >= 0:
                return "NEUTRAL", 0.0
        else:
            return "NEUTRAL", 0.0

        self.last_tp_distance = squeeze_width * 2.0
        strength = min(0.6 + volume_surge * 0.1, 1.0)
        return direction, strength

    def generate_signal(self, ctx: MarketContext) -> Signal:
        return Signal(symbol=ctx.symbol, direction="NEUTRAL", strength=0.0,
                      strategy_name=self.name, technical_score=0.0)
