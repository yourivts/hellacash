"""Composite signal aggregator — combines all indicators into a directional score."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd

from bot.indicators.divergence import rsi_divergence, volume_divergence
from bot.indicators.momentum import rsi, stochastic, cci
from bot.indicators.trend import ema, macd, adx, supertrend
from bot.indicators.volatility import bollinger_bands, atr
from bot.indicators.volume import volume_surge_ratio, cmf

logger = logging.getLogger(__name__)

# Default weights for each indicator signal (tunable by the learning system)
DEFAULT_WEIGHTS: Dict[str, float] = {
    "rsi": 0.15,
    "macd": 0.15,
    "bollinger": 0.10,
    "ema_trend": 0.10,
    "supertrend": 0.10,
    "adx": 0.05,      # trend strength modifier, not direction
    "volume": 0.05,
    "cci": 0.05,
    "rsi_divergence": 0.15,     # leading: RSI divergence
    "volume_divergence": 0.10,  # leading: volume divergence
}


@dataclass
class SignalResult:
    direction: str          # "LONG", "SHORT", "NEUTRAL"
    strength: float         # 0.0 – 1.0
    technical_score: float  # raw score in [-1, +1]
    breakdown: Dict[str, float] = field(default_factory=dict)
    indicator_values: Dict[str, Any] = field(default_factory=dict)
    confirming_count: int = 0


def compute(df: pd.DataFrame, weights: Optional[Dict[str, float]] = None) -> SignalResult:
    """
    Compute a composite directional signal from OHLCV candle data.
    df must have columns: open, high, low, close, volume (chronological order).
    Returns SignalResult.
    """
    if len(df) < 30:
        return SignalResult("NEUTRAL", 0.0, 0.0)

    w = weights or DEFAULT_WEIGHTS
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    scores: Dict[str, float] = {}
    values: Dict[str, Any] = {}

    # ── RSI ──────────────────────────────────────────────────────────────────
    try:
        r = rsi(close).iloc[-1]
        values["rsi"] = r
        if r < 30:
            scores["rsi"] = 1.0   # oversold → bullish
        elif r < 40:
            scores["rsi"] = 0.5
        elif r > 70:
            scores["rsi"] = -1.0  # overbought → bearish
        elif r > 60:
            scores["rsi"] = -0.5
        else:
            scores["rsi"] = 0.0
    except Exception:
        scores["rsi"] = 0.0

    # ── MACD ─────────────────────────────────────────────────────────────────
    try:
        m = macd(close)
        hist_now = m["histogram"].iloc[-1]
        hist_prev = m["histogram"].iloc[-2]
        macd_line = m["macd"].iloc[-1]
        values["macd_histogram"] = hist_now
        values["macd_line"] = macd_line
        if hist_now > 0 and hist_now > hist_prev:
            scores["macd"] = 1.0
        elif hist_now > 0:
            scores["macd"] = 0.4
        elif hist_now < 0 and hist_now < hist_prev:
            scores["macd"] = -1.0
        elif hist_now < 0:
            scores["macd"] = -0.4
        else:
            scores["macd"] = 0.0
    except Exception:
        scores["macd"] = 0.0

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    try:
        bb = bollinger_bands(close)
        pct_b = bb["pct_b"].iloc[-1]
        bw = bb["bandwidth"].iloc[-1]
        values["bb_pct_b"] = pct_b
        values["bb_bandwidth"] = bw
        if pct_b < 0.05:
            scores["bollinger"] = 1.0   # price at/below lower band
        elif pct_b < 0.2:
            scores["bollinger"] = 0.5
        elif pct_b > 0.95:
            scores["bollinger"] = -1.0  # price at/above upper band
        elif pct_b > 0.8:
            scores["bollinger"] = -0.5
        else:
            scores["bollinger"] = 0.0
    except Exception:
        scores["bollinger"] = 0.0

    # ── EMA trend ─────────────────────────────────────────────────────────────
    try:
        e20 = ema(close, 20).iloc[-1]
        e50 = ema(close, 50).iloc[-1]
        e200 = ema(close, 200).iloc[-1] if len(close) >= 200 else e50
        price = close.iloc[-1]
        values["ema20"] = e20
        values["ema50"] = e50
        bullish = sum([price > e20, price > e50, price > e200, e20 > e50])
        scores["ema_trend"] = (bullish - 2) / 2  # maps 0-4 to -1..+1
    except Exception:
        scores["ema_trend"] = 0.0

    # ── Supertrend ────────────────────────────────────────────────────────────
    try:
        st = supertrend(high, low, close).iloc[-1]
        values["supertrend"] = st
        scores["supertrend"] = float(st)  # +1 or -1
    except Exception:
        scores["supertrend"] = 0.0

    # ── ADX (trend strength modifier, not direction) ───────────────────────
    try:
        a = adx(high, low, close).iloc[-1]
        values["adx"] = a
        # ADX doesn't have direction; use as a multiplier
        adx_multiplier = min(a / 25.0, 2.0) if a > 20 else 0.5
        scores["adx"] = 0.0  # no direction contribution
    except Exception:
        adx_multiplier = 1.0

    # ── Volume surge ──────────────────────────────────────────────────────────
    try:
        vsr = volume_surge_ratio(volume).iloc[-1]
        values["volume_surge"] = vsr
        price_change = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]
        if vsr > 1.5:
            scores["volume"] = 1.0 if price_change > 0 else -1.0
        elif vsr > 1.2:
            scores["volume"] = 0.5 if price_change > 0 else -0.5
        else:
            scores["volume"] = 0.0
    except Exception:
        scores["volume"] = 0.0
        adx_multiplier = 1.0

    # ── CCI ───────────────────────────────────────────────────────────────────
    try:
        c = cci(high, low, close).iloc[-1]
        values["cci"] = c
        if c < -100:
            scores["cci"] = 1.0
        elif c < -50:
            scores["cci"] = 0.4
        elif c > 100:
            scores["cci"] = -1.0
        elif c > 50:
            scores["cci"] = -0.4
        else:
            scores["cci"] = 0.0
    except Exception:
        scores["cci"] = 0.0

    # ── RSI divergence (leading) ────────────────────────────────────────────
    try:
        rd = rsi_divergence(close).iloc[-1]
        values["rsi_divergence"] = rd
        scores["rsi_divergence"] = rd  # already -1 to +1
    except Exception:
        scores["rsi_divergence"] = 0.0

    # ── Volume divergence (leading) ──────────────────────────────────────────
    try:
        vd = volume_divergence(close, volume).iloc[-1]
        values["volume_divergence"] = vd
        scores["volume_divergence"] = vd  # already -1 to +1
    except Exception:
        scores["volume_divergence"] = 0.0

    # ── Weighted composite ────────────────────────────────────────────────────
    total_weight = sum(w.get(k, 0) for k in scores if k != "adx")
    raw_score = sum(scores[k] * w.get(k, 0) for k in scores if k != "adx")
    if total_weight > 0:
        raw_score = raw_score / total_weight

    # Apply ADX multiplier for trend-following amplification
    raw_score = max(-1.0, min(1.0, raw_score * adx_multiplier))

    # Count confirming indicators
    confirming = sum(1 for k, v in scores.items() if k != "adx" and v * raw_score > 0)

    direction = "NEUTRAL"
    if raw_score > 0.15:
        direction = "LONG"
    elif raw_score < -0.15:
        direction = "SHORT"

    strength = min(abs(raw_score), 1.0)

    return SignalResult(
        direction=direction,
        strength=strength,
        technical_score=raw_score,
        breakdown=scores,
        indicator_values=values,
        confirming_count=confirming,
    )
