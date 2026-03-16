"""Funding Rate Contrarian strategy — trade against crowded leverage."""
from __future__ import annotations

import logging
from typing import Any, Dict

from bot.strategy.base import BaseStrategy, MarketContext, Signal

logger = logging.getLogger(__name__)

# Thresholds for extreme funding (8h rate)
EXTREME_POSITIVE = 0.0003  # > 0.03% → crowded longs → short
EXTREME_NEGATIVE = -0.0003  # < -0.03% → crowded shorts → long
MODERATE_POSITIVE = 0.0001
MODERATE_NEGATIVE = -0.0001


class FundingContrarianStrategy(BaseStrategy):
    """
    Contrarian strategy based on Binance perpetual funding rates.

    Logic:
        - Extreme positive funding → market is over-leveraged long → SHORT
        - Extreme negative funding → market is over-leveraged short → LONG
        - Confirms with RSI (avoid fighting strong momentum)
        - Requires RSI to show exhaustion (overbought for shorts, oversold for longs)

    In backtesting: uses RSI extremes + price momentum as proxy for
    funding rate extremes (since historical funding data may not be available).
    """

    name = "funding_contrarian"

    RSI_OVERBOUGHT = 72.0
    RSI_OVERSOLD = 28.0
    RSI_4H_OVERBOUGHT = 60.0
    RSI_4H_OVERSOLD = 40.0

    def evaluate_1h(self, rsi_1h: float, rsi_4h: float, macd_hist: float, macd_hist_prev: float) -> tuple[str, float]:
        """Evaluate a 1h signal using RSI exhaustion and MACD fade confirmation.

        Returns a (direction, strength) tuple where direction is one of
        "LONG", "SHORT", or "NEUTRAL" and strength is in [0.0, 1.0].

        Rules:
            SHORT: rsi_1h > RSI_OVERBOUGHT, rsi_4h > RSI_4H_OVERBOUGHT, MACD histogram fading (hist < prev)
            LONG:  rsi_1h < RSI_OVERSOLD,   rsi_4h < RSI_4H_OVERSOLD,   MACD histogram recovering (hist > prev)
        """
        # SHORT: RSI > 72, 4h RSI > 60, MACD fading (hist < prev)
        if rsi_1h > self.RSI_OVERBOUGHT:
            if rsi_4h <= self.RSI_4H_OVERBOUGHT:
                return "NEUTRAL", 0.0
            if macd_hist >= macd_hist_prev:
                return "NEUTRAL", 0.0
            strength = min((rsi_1h - self.RSI_OVERBOUGHT) / 20.0 + 0.5, 1.0)
            return "SHORT", strength

        # LONG: RSI < 28, 4h RSI < 40, MACD recovering (hist > prev)
        if rsi_1h < self.RSI_OVERSOLD:
            if rsi_4h >= self.RSI_4H_OVERSOLD:
                return "NEUTRAL", 0.0
            if macd_hist <= macd_hist_prev:
                return "NEUTRAL", 0.0
            strength = min((self.RSI_OVERSOLD - rsi_1h) / 20.0 + 0.5, 1.0)
            return "LONG", strength

        return "NEUTRAL", 0.0

    def generate_signal(self, ctx: MarketContext) -> Signal:
        df = ctx.candles_5m
        if len(df) < 50:
            return Signal(ctx.symbol, "NEUTRAL", 0.0, self.name)

        # Live mode: use actual on-chain score (includes funding)
        # The onchain_score already encodes funding as contrarian signal
        onchain = ctx.onchain_score

        direction = "NEUTRAL"
        strength = 0.0
        confirming = 0

        if onchain > 0.3:  # strong contrarian bullish (shorts crowded)
            direction = "LONG"
            strength = min(abs(onchain), 1.0)
            confirming = 2
        elif onchain < -0.3:  # strong contrarian bearish (longs crowded)
            direction = "SHORT"
            strength = min(abs(onchain), 1.0)
            confirming = 2

        return Signal(
            symbol=ctx.symbol,
            direction=direction,
            strength=strength,
            strategy_name=self.name,
            technical_score=onchain,
            indicator_snapshot={"confirming_count": confirming},
        )
