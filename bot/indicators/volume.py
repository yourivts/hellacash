"""Volume indicators: OBV, VWAP, CMF."""
from __future__ import annotations

import numpy as np
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


def volume_profile_support(
    close: pd.Series,
    volume: pd.Series,
    lookback: int = 200,
) -> pd.Series:
    """Compute volume profile support/resistance score per bar.

    Returns a score from -1.0 to +1.0:
    - Positive = volume concentrated near current price from below (support)
    - Negative = volume concentrated from above (resistance)
    - Near zero = no clear volume clustering
    """
    n = len(close)
    scores = np.zeros(n, dtype=float)

    close_arr = close.values.astype(float)
    volume_arr = volume.values.astype(float)

    for i in range(lookback, n):
        window_close = close_arr[i - lookback : i]
        window_vol = volume_arr[i - lookback : i]

        price_min = window_close.min()
        price_max = window_close.max()
        price_range = price_max - price_min

        if price_range < 1e-9:
            scores[i] = 0.0
            continue

        total_vol = window_vol.sum()
        if total_vol < 1e-9:
            scores[i] = 0.0
            continue

        lower_threshold = price_min + 0.2 * price_range
        upper_threshold = price_max - 0.2 * price_range

        lower_vol = window_vol[window_close <= lower_threshold].sum()
        upper_vol = window_vol[window_close >= upper_threshold].sum()

        scores[i] = (lower_vol - upper_vol) / total_vol

    return pd.Series(scores, index=close.index, dtype=float)
