"""Volatility indicators: Bollinger Bands, ATR, Keltner."""
from __future__ import annotations

import pandas as pd


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def bollinger_bands(
    close: pd.Series, period: int = 20, std_dev: float = 2.0
) -> pd.DataFrame:
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    return pd.DataFrame({
        "upper": mid + std_dev * std,
        "mid": mid,
        "lower": mid - std_dev * std,
        "bandwidth": (2 * std_dev * std) / mid.replace(0, 1e-9),
        "pct_b": (close - (mid - std_dev * std)) / (2 * std_dev * std).replace(0, 1e-9),
    })


def keltner_channels(
    high: pd.Series, low: pd.Series, close: pd.Series,
    period: int = 20, multiplier: float = 1.5
) -> pd.DataFrame:
    mid = close.ewm(span=period, adjust=False).mean()
    a = atr(high, low, close, period)
    return pd.DataFrame({
        "upper": mid + multiplier * a,
        "mid": mid,
        "lower": mid - multiplier * a,
    })
