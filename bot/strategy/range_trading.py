"""Range trading strategy -- profits from price oscillating within BB ranges."""
from __future__ import annotations

import logging
from typing import Any, Dict

import pandas as pd

from bot.indicators.momentum import rsi
from bot.indicators.trend import adx
from bot.indicators.volatility import bollinger_bands
from bot.indicators.volume import volume_profile_support
from bot.strategy.base import BaseStrategy, MarketContext, Signal

logger = logging.getLogger(__name__)

# Range confirmation thresholds (1h timeframe)
BB_BANDWIDTH_MAX = 0.10
BB_BANDWIDTH_MIN = 0.02
ADX_MAX = 20
BB_BANDWIDTH_RESET = 0.15
ADX_RESET = 25

# Entry thresholds (5m timeframe)
ENTRY_PROXIMITY_PCT = 0.01
RSI_LONG_THRESHOLD = 40
RSI_SHORT_THRESHOLD = 60
VOLUME_SCORE_THRESHOLD = 0.10

# Bounce limit
MAX_BOUNCES = 3


class RangeStrategy(BaseStrategy):
    """Dedicated range-trading strategy using BB + volume clustering."""

    name = "range"

    def __init__(self) -> None:
        self._bounce_counts: Dict[str, int] = {}

    def get_bounce_count(self, symbol: str) -> int:
        return self._bounce_counts.get(symbol, 0)

    def increment_bounce(self, symbol: str) -> None:
        self._bounce_counts[symbol] = self._bounce_counts.get(symbol, 0) + 1

    def reset_bounces(self, symbol: str) -> None:
        self._bounce_counts[symbol] = 0

    def generate_signal(self, ctx: MarketContext) -> Signal:
        symbol = ctx.symbol
        df_5m = ctx.candles_5m
        df_1h = ctx.candles_1h
        neutral = Signal(symbol, "NEUTRAL", 0.0, self.name,
                         indicator_snapshot=self._empty_snapshot())

        if len(df_5m) < 50 or len(df_1h) < 30:
            return neutral

        # --- 1h range confirmation ---
        close_1h = df_1h["close"]
        high_1h = df_1h["high"]
        low_1h = df_1h["low"]

        bb_1h = bollinger_bands(close_1h)
        bandwidth_1h = bb_1h["bandwidth"].iloc[-1]
        adx_1h = adx(high_1h, low_1h, close_1h).iloc[-1]

        # Check bounce counter reset
        if bandwidth_1h > BB_BANDWIDTH_RESET or adx_1h > ADX_RESET:
            self.reset_bounces(symbol)

        # Range not confirmed
        if bandwidth_1h >= BB_BANDWIDTH_MAX or bandwidth_1h < BB_BANDWIDTH_MIN:
            return neutral
        if adx_1h >= ADX_MAX:
            return neutral

        # Bounce limit reached
        if self.get_bounce_count(symbol) >= MAX_BOUNCES:
            return neutral

        # --- 5m entry trigger ---
        close_5m = df_5m["close"]
        high_5m = df_5m["high"]
        low_5m = df_5m["low"]
        volume_5m = df_5m["volume"]

        bb_5m = bollinger_bands(close_5m)
        rsi_5m = rsi(close_5m).iloc[-1]
        vol_score = volume_profile_support(
            close_5m, volume_5m, lookback=min(200, len(close_5m) - 1),
        ).iloc[-1]

        price = ctx.current_price
        upper = bb_5m["upper"].iloc[-1]
        lower = bb_5m["lower"].iloc[-1]
        mid = bb_5m["mid"].iloc[-1]

        direction = "NEUTRAL"
        confirming = 0

        # LONG: price within 1% of lower BB + RSI < 40 + volume support
        if abs(price - lower) / lower <= ENTRY_PROXIMITY_PCT and rsi_5m < RSI_LONG_THRESHOLD:
            if vol_score > VOLUME_SCORE_THRESHOLD:
                direction = "LONG"
                confirming = 1
                if abs(price - lower) / lower <= 0.005:
                    confirming += 1
                if vol_score > 0.3:
                    confirming += 1

        # SHORT: price within 1% of upper BB + RSI > 60 + volume resistance
        elif abs(price - upper) / upper <= ENTRY_PROXIMITY_PCT and rsi_5m > RSI_SHORT_THRESHOLD:
            if vol_score < -VOLUME_SCORE_THRESHOLD:
                direction = "SHORT"
                confirming = 1
                if abs(price - upper) / upper <= 0.005:
                    confirming += 1
                if vol_score < -0.3:
                    confirming += 1

        if direction == "NEUTRAL":
            return neutral

        strength = min(abs(vol_score) + (1.0 - bandwidth_1h / BB_BANDWIDTH_MAX) * 0.5, 1.0)

        return Signal(
            symbol=symbol,
            direction=direction,
            strength=strength,
            strategy_name=self.name,
            technical_score=strength if direction == "LONG" else -strength,
            indicator_snapshot={
                "range_mid": float(mid),
                "range_upper": float(upper),
                "range_lower": float(lower),
                "bounce_count": self.get_bounce_count(symbol),
                "confirming_count": confirming,
                "rsi": float(rsi_5m),
                "bb_bandwidth": float(bandwidth_1h),
                "vol_profile_score": float(vol_score),
            },
        )

    @staticmethod
    def _empty_snapshot() -> Dict[str, Any]:
        return {
            "range_mid": 0.0,
            "range_upper": 0.0,
            "range_lower": 0.0,
            "bounce_count": 0,
            "confirming_count": 0,
        }
