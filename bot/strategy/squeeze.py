"""Volatility Squeeze strategy — BB compression followed by expansion."""
from __future__ import annotations

from bot.indicators.momentum import rsi
from bot.indicators.trend import ema
from bot.indicators.volatility import bollinger_bands
from bot.indicators.volume import volume_surge_ratio
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
        """Live signal: detect BB squeeze→expansion breakouts."""
        neutral = Signal(symbol=ctx.symbol, direction="NEUTRAL", strength=0.0,
                         strategy_name=self.name, technical_score=0.0)

        df_1h = ctx.candles_1h
        if df_1h is None or len(df_1h) < 30:
            return neutral

        close_1h = df_1h["close"]
        high_1h = df_1h["high"]
        low_1h = df_1h["low"]
        vol_1h = df_1h["volume"]

        bb = bollinger_bands(close_1h)
        bw = bb["bandwidth"]
        if len(bw) < 2:
            return neutral

        bw_prev = float(bw.iloc[-2])
        bw_now = float(bw.iloc[-1])
        bb_upper = float(bb["upper"].iloc[-1])
        bb_lower = float(bb["lower"].iloc[-1])
        price = ctx.current_price

        # Volume surge
        vsr = volume_surge_ratio(vol_1h)
        vol_surge = float(vsr.iloc[-1]) if len(vsr) > 0 else 1.0

        # EMA50 slope over 10 bars (smoothed, not 1-bar noise)
        ema50 = ema(close_1h, 50)
        if len(ema50) >= 10:
            ema50_slope = float(ema50.iloc[-1]) - float(ema50.iloc[-10])
        else:
            ema50_slope = 0.0

        direction, strength = self.evaluate_1h(
            bw_prev, bw_now, price, bb_upper, bb_lower, vol_surge, ema50_slope,
        )

        if direction == "NEUTRAL":
            return neutral

        return Signal(
            symbol=ctx.symbol,
            direction=direction,
            strength=strength,
            strategy_name=self.name,
            technical_score=strength if direction == "LONG" else -strength,
            indicator_snapshot={"confirming_count": 2},
        )
