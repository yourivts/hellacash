"""Volatility Squeeze strategy stub — full implementation in Task 9."""
from bot.strategy.base import BaseStrategy, MarketContext, Signal


class SqueezeStrategy(BaseStrategy):
    name = "squeeze"

    def generate_signal(self, ctx: MarketContext) -> Signal:
        return Signal(
            symbol=ctx.symbol, direction="NEUTRAL", strength=0.0,
            strategy_name=self.name, technical_score=0.0,
        )
