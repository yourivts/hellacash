"""27-feature market feature extractor for RL training and inference."""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from bot.indicators.momentum import rsi
from bot.indicators.trend import adx, ema, macd
from bot.indicators.volatility import atr, bollinger_bands

logger = logging.getLogger(__name__)

N_FEATURES = 27


def extract_features(
    candles_1m: pd.DataFrame,
    btc_candles_1m: pd.DataFrame = None,
) -> np.ndarray:
    """Compute 27 normalized features from a window of 1m candles.
    Resamples 1m to daily/4h/1h internally. All values hard-clipped to [-1, 1].
    Returns float32 array of shape (27,).
    """
    features = np.zeros(N_FEATURES, dtype=np.float32)

    if candles_1m is None or len(candles_1m) < 1440:
        return features

    try:
        close_1m = candles_1m["close"]
        high_1m = candles_1m["high"]
        low_1m = candles_1m["low"]
        vol_1m = candles_1m["volume"]

        df_1h = candles_1m.resample("1h").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()
        df_4h = candles_1m.resample("4h").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()
        df_1d = candles_1m.resample("1D").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()

        if len(df_1d) < 5 or len(df_1h) < 30:
            return features

        price = float(close_1m.iloc[-1])
        if price <= 0:
            return features

        # --- Trend features (0-3) ---
        ema200_d = ema(df_1d["close"], 200)
        ema50_d = ema(df_1d["close"], 50)
        if len(ema200_d.dropna()) > 0:
            features[0] = ((price - float(ema200_d.iloc[-1])) / float(ema200_d.iloc[-1]) * 100.0) / 20.0
        if len(ema50_d.dropna()) > 0:
            features[1] = ((price - float(ema50_d.iloc[-1])) / float(ema50_d.iloc[-1]) * 100.0) / 10.0
        if len(ema50_d.dropna()) >= 10:
            slope = float(ema50_d.iloc[-1]) - float(ema50_d.iloc[-10])
            features[2] = slope / price
        period_return = (price / float(close_1m.iloc[0]) - 1.0) * 100.0
        features[3] = max(min(period_return, 50.0), -50.0) / 50.0

        # --- Volatility features (4-7) ---
        atr_4h = atr(df_4h["high"], df_4h["low"], df_4h["close"])
        if len(atr_4h.dropna()) > 0:
            close_4h = df_4h["close"]
            atr_pct = (atr_4h / close_4h * 100.0).dropna()
            if len(atr_pct) > 0:
                features[4] = float(atr_pct.mean()) / 10.0
                features[5] = float(atr_pct.std()) / 5.0 if len(atr_pct) > 1 else 0.0
                avg = float(atr_pct.mean())
                if avg > 0:
                    features[6] = min(float(atr_pct.iloc[-1]) / avg, 3.0) / 3.0
                features[7] = float(atr_pct.max()) / 20.0

        # --- Regime features (8-11) ---
        adx_4h = adx(df_4h["high"], df_4h["low"], df_4h["close"])
        if len(adx_4h.dropna()) > 0:
            adx_vals = adx_4h.dropna()
            features[8] = float((adx_vals > 25).mean())
            features[9] = float((adx_vals < 20).mean())
            features[10] = float(adx_vals.iloc[-1]) / 50.0
        rsi_1h = rsi(df_1h["close"])
        if len(rsi_1h.dropna()) > 0:
            features[11] = float(rsi_1h.iloc[-1]) / 100.0

        # --- Volume features (12-13) ---
        if len(vol_1m) > 100:
            half = len(vol_1m) // 2
            first_half_vol = float(vol_1m.iloc[:half].mean())
            second_half_vol = float(vol_1m.iloc[half:].mean())
            if first_half_vol > 0:
                vol_trend = (second_half_vol / first_half_vol - 1.0) * 100.0
                features[12] = max(min(vol_trend, 100.0), -100.0) / 100.0
            avg_vol = float(vol_1m.mean())
            if avg_vol > 0:
                features[13] = min(float(vol_1m.iloc[-1]) / avg_vol, 5.0) / 5.0

        # --- Price structure (14-16) ---
        period_high = float(high_1m.max())
        period_low = float(low_1m.min())
        if period_high > 0:
            features[14] = (price - period_high) / period_high * 100.0 / -50.0
        if period_low > 0:
            features[15] = (price - period_low) / period_low * 100.0 / 50.0
        daily_returns = df_1d["close"].pct_change().dropna() * 100.0
        reversals = 0
        prev_dir = 0
        cumulative = 0.0
        for ret in daily_returns:
            if prev_dir == 0:
                prev_dir = 1 if ret > 0 else -1
                cumulative = ret
            elif (ret > 0 and prev_dir > 0) or (ret < 0 and prev_dir < 0):
                cumulative += ret
            else:
                if abs(cumulative) > 5.0:
                    reversals += 1
                prev_dir = 1 if ret > 0 else -1
                cumulative = ret
        features[16] = min(reversals, 20) / 20.0

        # --- Bollinger Band features (17-18) ---
        bb = bollinger_bands(df_1h["close"])
        if "bandwidth" in bb and len(bb["bandwidth"].dropna()) > 0:
            features[17] = min(float(bb["bandwidth"].iloc[-1]) / 0.2, 1.0)
        if "upper" in bb and "lower" in bb:
            upper = float(bb["upper"].iloc[-1])
            lower = float(bb["lower"].iloc[-1])
            if upper > lower:
                pct_b = (price - lower) / (upper - lower)
                features[18] = max(min(pct_b, 1.0), 0.0)

        # --- Momentum features (19-20) ---
        rsi_vals = rsi(df_1h["close"])
        if len(rsi_vals.dropna()) >= 14:
            rsi_roc = float(rsi_vals.iloc[-1]) - float(rsi_vals.iloc[-14])
            features[19] = rsi_roc / 50.0
        macd_data = macd(df_1h["close"])
        if "histogram" in macd_data and len(macd_data["histogram"].dropna()) > 0:
            hist = float(macd_data["histogram"].iloc[-1])
            features[20] = 1.0 if hist > 0 else -1.0

        # --- Candle structure (21-22) ---
        if len(candles_1m) > 100:
            sample = candles_1m.tail(1440)
            bodies = (sample["close"] - sample["open"]).abs()
            upper_wicks = sample["high"] - sample[["open", "close"]].max(axis=1)
            lower_wicks = sample[["open", "close"]].min(axis=1) - sample["low"]
            total_wicks = upper_wicks + lower_wicks
            valid = total_wicks > 0
            if valid.sum() > 0:
                ratios = bodies[valid] / total_wicks[valid]
                features[21] = min(float(ratios.mean()), 2.0) / 2.0
            longer_lower = (lower_wicks > upper_wicks).mean()
            longer_upper = (upper_wicks > lower_wicks).mean()
            features[22] = float(longer_lower - longer_upper)

        # --- Time features (23-24) ---
        last_ts = candles_1m.index[-1]
        features[23] = math.sin(2 * math.pi * last_ts.dayofweek / 7.0)
        features[24] = math.sin(2 * math.pi * last_ts.hour / 24.0)

        # --- BTC correlation (25-26) ---
        if btc_candles_1m is not None and len(btc_candles_1m) > 0:
            try:
                target_daily = close_1m.resample("1D").last().dropna()
                btc_daily = btc_candles_1m["close"].resample("1D").last().dropna()
                aligned = pd.concat([target_daily, btc_daily], axis=1, join="inner")
                if len(aligned) >= 30:
                    corr = aligned.iloc[:, 0].rolling(30).corr(aligned.iloc[:, 1])
                    if len(corr.dropna()) > 0:
                        features[25] = float(corr.iloc[-1])
                btc_ema50 = ema(btc_candles_1m["close"].resample("1D").last().dropna(), 50)
                if len(btc_ema50.dropna()) >= 10:
                    btc_price = float(btc_candles_1m["close"].iloc[-1])
                    if btc_price > 0:
                        btc_slope = float(btc_ema50.iloc[-1]) - float(btc_ema50.iloc[-10])
                        features[26] = btc_slope / btc_price
            except Exception:
                pass

    except Exception as e:
        logger.warning("Feature extraction failed: %s", e)
        return np.zeros(N_FEATURES, dtype=np.float32)

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(features, -1.0, 1.0).astype(np.float32)
