"""Order Flow / Absorption strategy — detect hidden buying/selling pressure."""
from __future__ import annotations

import logging
from typing import Any, Dict

from bot.strategy.base import BaseStrategy, MarketContext, Signal

logger = logging.getLogger(__name__)


class OrderFlowStrategy(BaseStrategy):
    """
    Detects absorption patterns using volume and price action.

    Absorption = large volume without proportional price movement, meaning
    big orders are being absorbed by hidden liquidity on the opposite side.

    Signals:
        - High volume + small body + long lower wick → buying absorption → LONG
        - High volume + small body + long upper wick → selling absorption → SHORT

    Combines with OBV divergence and volume surge detection.

    In backtesting: uses OHLCV candle data to approximate orderflow signals:
        - Volume surge ratio (VSR) for volume intensity
        - Candle body ratio (body/range) for absorption detection
        - Wick ratios for rejection patterns
        - OBV divergence for hidden pressure
    """

    name = "orderflow"

    MAX_BODY_RATIO = 0.30
    MIN_WICK_RATIO = 0.40
    MIN_VOLUME_SURGE = 1.5
    RSI_LOW = 30.0
    RSI_HIGH = 70.0

    def evaluate_1h(self, body_ratio: float, wick_lower_ratio: float,
                    wick_upper_ratio: float, volume_surge: float,
                    rsi_1h: float, cmf: float,
                    obv_divergence: float) -> tuple[str, float]:
        """Evaluate absorption signals from a 1-hour candle context.

        Returns a (direction, strength) tuple where direction is one of
        "LONG", "SHORT", or "NEUTRAL".
        """
        if body_ratio > self.MAX_BODY_RATIO:
            return "NEUTRAL", 0.0
        if volume_surge < self.MIN_VOLUME_SURGE:
            return "NEUTRAL", 0.0
        if rsi_1h < self.RSI_LOW or rsi_1h > self.RSI_HIGH:
            return "NEUTRAL", 0.0

        buying_absorption = wick_lower_ratio >= self.MIN_WICK_RATIO
        selling_absorption = wick_upper_ratio >= self.MIN_WICK_RATIO

        if not buying_absorption and not selling_absorption:
            return "NEUTRAL", 0.0

        confirming = 0
        if buying_absorption:
            direction = "LONG"
            if cmf > 0:
                confirming += 1
            if obv_divergence > 0:
                confirming += 1
        else:
            direction = "SHORT"
            if cmf < 0:
                confirming += 1
            if obv_divergence < 0:
                confirming += 1

        strength = min(0.4 + confirming * 0.3, 1.0)
        return direction, strength

    def generate_signal(self, ctx: MarketContext) -> Signal:
        df = ctx.candles_5m
        if len(df) < 30:
            return Signal(ctx.symbol, "NEUTRAL", 0.0, self.name)

        # Live mode also considers orderbook imbalance
        book_score = ctx.orderbook_imbalance

        direction = "NEUTRAL"
        strength = 0.0
        confirming = 0

        if abs(book_score) >= 0.3:
            direction = "LONG" if book_score > 0 else "SHORT"
            strength = min(abs(book_score), 1.0)
            confirming = 2

        return Signal(
            symbol=ctx.symbol,
            direction=direction,
            strength=strength,
            strategy_name=self.name,
            technical_score=book_score,
            indicator_snapshot={"confirming_count": confirming},
        )
