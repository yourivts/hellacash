"""Trend indicators: EMA, MACD, ADX."""
from __future__ import annotations

import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Returns DataFrame with columns: macd, signal, histogram."""
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    return pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "histogram": macd_line - signal_line,
    })


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index — measures trend strength (0-100)."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()

    dm_plus = high.diff().clip(lower=0)
    dm_minus = (-low.diff()).clip(lower=0)
    dm_plus = dm_plus.where(dm_plus > dm_minus, 0)
    dm_minus = dm_minus.where(dm_minus > dm_plus, 0)

    di_plus = 100 * dm_plus.ewm(alpha=1 / period, adjust=False).mean() / atr
    di_minus = 100 * dm_minus.ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, 1e-9)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def supertrend(
    high: pd.Series, low: pd.Series, close: pd.Series,
    period: int = 10, multiplier: float = 3.0
) -> pd.Series:
    """Returns +1 (bullish) or -1 (bearish)."""
    hl2 = (high + low) / 2
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()

    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    trend = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(1, index=close.index, dtype=int)

    for i in range(1, len(close)):
        upper.iloc[i] = min(upper.iloc[i], upper.iloc[i - 1]) if close.iloc[i - 1] > upper.iloc[i - 1] else upper.iloc[i]
        lower.iloc[i] = max(lower.iloc[i], lower.iloc[i - 1]) if close.iloc[i - 1] < lower.iloc[i - 1] else lower.iloc[i]

        if close.iloc[i] > upper.iloc[i - 1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < lower.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]

    return direction.astype(float)
