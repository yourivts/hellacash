"""VWAP / EMA mean-reversion strategy."""
from __future__ import annotations

from bot.strategy.base import BaseStrategy, MarketContext, Signal


class MeanReversionStrategy(BaseStrategy):
    """Fade extreme deviations from moving averages.

    Signals:
        - Price far below EMA50 + RSI oversold → LONG (snap-back expected)
        - Price far above EMA50 + RSI overbought → SHORT (snap-back expected)
        - Requires low ADX (not trending) to avoid catching falling knives
    """

    name = "mean_reversion"

    DEVIATION_PCT = 2.0   # min distance from EMA50 to trigger
    RSI_LOW = 35.0
    RSI_HIGH = 65.0
    MAX_ADX = 25.0        # only trade in non-trending markets

    def evaluate_1h(self, price: float, ema50: float,
                    rsi_1h: float, adx_val: float,
                    bb_lower: float, bb_upper: float) -> tuple[str, float]:
        """Evaluate mean-reversion signals from 1h indicators."""
        if adx_val > self.MAX_ADX:
            return "NEUTRAL", 0.0  # trending — don't fade

        if ema50 <= 0:
            return "NEUTRAL", 0.0

        deviation_pct = (price - ema50) / ema50 * 100.0

        # Oversold: price well below EMA50 + near/below BB lower
        if deviation_pct < -self.DEVIATION_PCT and rsi_1h < self.RSI_LOW:
            # Stronger signal the further from the mean
            strength = min(0.4 + abs(deviation_pct) / 10.0, 1.0)
            if price <= bb_lower:
                strength = min(strength + 0.15, 1.0)
            return "LONG", strength

        # Overbought: price well above EMA50 + near/above BB upper
        if deviation_pct > self.DEVIATION_PCT and rsi_1h > self.RSI_HIGH:
            strength = min(0.4 + abs(deviation_pct) / 10.0, 1.0)
            if price >= bb_upper:
                strength = min(strength + 0.15, 1.0)
            return "SHORT", strength

        return "NEUTRAL", 0.0

    def generate_signal(self, ctx: MarketContext) -> Signal:
        return Signal(ctx.symbol, "NEUTRAL", 0.0, self.name)
