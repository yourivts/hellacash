"""Volume indicators: OBV, VWAP, CMF."""
from __future__ import annotations

import pandas as pd


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (direction * volume).cumsum()


def vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> pd.Series:
    tp = (high + low + close) / 3
    return (tp * volume).cumsum() / volume.cumsum().replace(0, 1e-9)


def cmf(
    high: pd.Series, low: pd.Series, close: pd.Series,
    volume: pd.Series, period: int = 20
) -> pd.Series:
    mfv = ((close - low) - (high - close)) / (high - low).replace(0, 1e-9) * volume
    return mfv.rolling(period).sum() / volume.rolling(period).sum().replace(0, 1e-9)


def volume_surge_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    """Current volume vs. rolling average — >2.0 means a surge."""
    avg = volume.rolling(period).mean()
    return volume / avg.replace(0, 1e-9)
