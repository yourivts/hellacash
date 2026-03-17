"""Leading indicators: RSI divergence and volume divergence detection."""
from __future__ import annotations

import numpy as np
import pandas as pd

from bot.indicators.momentum import rsi as compute_rsi


def _find_swing_highs(values: np.ndarray, order: int = 5) -> list[int]:
    """Find indices of local maxima (swing highs)."""
    highs = []
    for i in range(order, len(values) - order):
        if all(values[i] >= values[i - j] for j in range(1, order + 1)) and \
           all(values[i] >= values[i + j] for j in range(1, order + 1)):
            highs.append(i)
    return highs


def _find_swing_lows(values: np.ndarray, order: int = 5) -> list[int]:
    """Find indices of local minima (swing lows)."""
    lows = []
    for i in range(order, len(values) - order):
        if all(values[i] <= values[i - j] for j in range(1, order + 1)) and \
           all(values[i] <= values[i + j] for j in range(1, order + 1)):
            lows.append(i)
    return lows


def rsi_divergence(close: pd.Series, period: int = 14, lookback: int = 50) -> pd.Series:
    """
    Detect RSI divergence — a leading reversal signal.

    Returns a Series of scores:
      +1.0 = bullish divergence (price lower low, RSI higher low → reversal UP)
      -1.0 = bearish divergence (price higher high, RSI lower high → reversal DOWN)
       0.0 = no divergence
    """
    rsi_vals = compute_rsi(close, period)
    result = np.zeros(len(close))
    price = close.values
    rsi_arr = rsi_vals.values

    for i in range(lookback, len(close)):
        window_price = price[i - lookback:i + 1]
        window_rsi = rsi_arr[i - lookback:i + 1]

        # Find swing points in the window
        highs = _find_swing_highs(window_price, order=3)
        lows = _find_swing_lows(window_price, order=3)

        # Bearish divergence: last two swing highs — price rising, RSI falling
        if len(highs) >= 2:
            h1, h2 = highs[-2], highs[-1]
            if window_price[h2] > window_price[h1] and window_rsi[h2] < window_rsi[h1]:
                # Only signal if the divergence is recent (within last 10 bars)
                if (lookback - h2) < 10:
                    result[i] = -1.0

        # Bullish divergence: last two swing lows — price falling, RSI rising
        if len(lows) >= 2:
            l1, l2 = lows[-2], lows[-1]
            if window_price[l2] < window_price[l1] and window_rsi[l2] > window_rsi[l1]:
                if (lookback - l2) < 10:
                    result[i] = 1.0

    return pd.Series(result, index=close.index)


def volume_divergence(close: pd.Series, volume: pd.Series, lookback: int = 20) -> pd.Series:
    """
    Detect volume divergence — price trending but volume declining.

    Returns a Series of scores:
      +1.0 = bullish divergence (price falling but volume drying up → selling exhaustion)
      -1.0 = bearish divergence (price rising but volume declining → weakening rally)
       0.0 = no divergence
    """
    result = np.zeros(len(close))
    price = close.values
    vol = volume.values

    for i in range(lookback, len(close)):
        # Price trend over lookback window
        price_change = (price[i] - price[i - lookback]) / price[i - lookback] if price[i - lookback] > 0 else 0

        # Volume trend: compare recent avg volume to earlier avg volume
        half = lookback // 2
        early_vol = np.mean(vol[i - lookback:i - half]) if np.mean(vol[i - lookback:i - half]) > 0 else 1e-9
        recent_vol = np.mean(vol[i - half:i + 1])
        vol_ratio = recent_vol / early_vol

        # Significant price move (>2%) with declining volume (<0.8x)
        if price_change > 0.02 and vol_ratio < 0.8:
            # Price up, volume down → bearish divergence
            strength = min((0.8 - vol_ratio) / 0.4, 1.0)  # scale 0.8→0.4 to 0→1
            result[i] = -strength
        elif price_change < -0.02 and vol_ratio < 0.8:
            # Price down, volume down → bullish divergence (selling exhaustion)
            strength = min((0.8 - vol_ratio) / 0.4, 1.0)
            result[i] = strength

    return pd.Series(result, index=close.index)
