"""Bollinger Band squeeze + RSI extreme mean-reversion strategy."""
from __future__ import annotations

from bot.indicators.momentum import rsi
from bot.indicators.volatility import bollinger_bands
from bot.strategy.base import BaseStrategy, MarketContext, Signal


class MeanReversionStrategy(BaseStrategy):
    name = "mean_reversion"

    def __init__(
        self,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        bb_period: int = 20,
        bb_std: float = 2.0,
    ) -> None:
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.bb_period = bb_period
        self.bb_std = bb_std

    def generate_signal(self, ctx: MarketContext) -> Signal:
        df = ctx.candles_5m
        if len(df) < self.bb_period + 5:
            return Signal(ctx.symbol, "NEUTRAL", 0.0, self.name)

        close = df["close"]
        r = rsi(close).iloc[-1]
        bb = bollinger_bands(close, self.bb_period, self.bb_std)
        pct_b = bb["pct_b"].iloc[-1]

        direction = "NEUTRAL"
        strength = 0.0

        if r < self.rsi_oversold and pct_b < 0.05:
            direction = "LONG"
            strength = (self.rsi_oversold - r) / self.rsi_oversold
        elif r > self.rsi_overbought and pct_b > 0.95:
            direction = "SHORT"
            strength = (r - self.rsi_overbought) / (100 - self.rsi_overbought)

        strength = min(strength, 1.0)

        return Signal(
            symbol=ctx.symbol,
            direction=direction,
            strength=strength,
            strategy_name=self.name,
            technical_score=strength if direction == "LONG" else -strength,
            indicator_snapshot={
                "rsi": r,
                "bb_pct_b": pct_b,
                "confirming_count": 2 if direction != "NEUTRAL" else 0,
            },
        )
