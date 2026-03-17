"""RSI momentum + MACD divergence strategy."""
from __future__ import annotations

from bot.strategy.base import BaseStrategy, MarketContext, Signal


class MomentumStrategy(BaseStrategy):
    """Pure momentum: RSI extremes with MACD confirmation.

    Signals:
        - RSI crossing above oversold + MACD turning up → LONG
        - RSI crossing below overbought + MACD turning down → SHORT
    """

    name = "momentum"

    RSI_OVERSOLD = 30.0
    RSI_OVERBOUGHT = 70.0

    def evaluate_1h(self, rsi_1h: float, rsi_1h_prev: float,
                    macd_hist: float, macd_hist_prev: float,
                    volume_surge: float) -> tuple[str, float]:
        """Evaluate momentum signals from 1h indicators."""
        # RSI crossing out of oversold → bullish momentum
        if rsi_1h_prev <= self.RSI_OVERSOLD and rsi_1h > self.RSI_OVERSOLD:
            if macd_hist > macd_hist_prev:  # MACD turning up
                strength = min(0.5 + volume_surge * 0.1, 1.0)
                return "LONG", strength

        # RSI crossing out of overbought → bearish momentum
        if rsi_1h_prev >= self.RSI_OVERBOUGHT and rsi_1h < self.RSI_OVERBOUGHT:
            if macd_hist < macd_hist_prev:  # MACD turning down
                strength = min(0.5 + volume_surge * 0.1, 1.0)
                return "SHORT", strength

        # Strong momentum continuation (RSI in extreme + MACD confirms)
        if rsi_1h > 60 and macd_hist > 0 and macd_hist > macd_hist_prev:
            if volume_surge >= 1.3:
                return "LONG", min(0.3 + (rsi_1h - 60) / 40.0 * 0.3, 0.7)
        if rsi_1h < 40 and macd_hist < 0 and macd_hist < macd_hist_prev:
            if volume_surge >= 1.3:
                return "SHORT", min(0.3 + (40 - rsi_1h) / 40.0 * 0.3, 0.7)

        return "NEUTRAL", 0.0

    def generate_signal(self, ctx: MarketContext) -> Signal:
        return Signal(ctx.symbol, "NEUTRAL", 0.0, self.name)
