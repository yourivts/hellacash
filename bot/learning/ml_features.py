"""ML feature engineering — tabular features, LSTM sequences, and batch extraction.

This module provides three public functions used to feed the ML signal generator:

  extract_tabular_features  — 65-element float32 vector per candle snapshot
  build_lstm_sequence       — (96, 7) float32 array covering the last 8 hours
  extract_all_features      — batch extraction across an entire candle history
"""
from __future__ import annotations

import logging
import math
from typing import List, Tuple

import numpy as np
import pandas as pd

from bot.indicators.divergence import rsi_divergence, volume_divergence
from bot.indicators.momentum import cci, rsi, stochastic
from bot.indicators.trend import adx, ema, macd, supertrend
from bot.indicators.volatility import atr, bollinger_bands, keltner_channels
from bot.indicators.volume import (
    cmf,
    obv,
    volume_profile_support,
    volume_surge_ratio,
    vwap,
)
from bot.learning.rl_signal_evaluator import get_coin_profile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

N_TABULAR: int = 65
MIN_BARS_5M: int = 1500

LSTM_WINDOW: int = 96
LSTM_CHANNELS: int = 7
LSTM_MIN_BARS: int = LSTM_WINDOW + 20  # need rolling window headroom

SIGNAL_EVERY: int = 6  # extract one sample every 6 five-minute bars (= 30 min)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_RESAMPLE_OHLCV = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def _safe_last(series: pd.Series, default: float = 0.0) -> float:
    """Return the last finite value of a Series, or *default*."""
    try:
        vals = series.dropna()
        if vals.empty:
            return default
        v = float(vals.iloc[-1])
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _pct_distance(price: float, reference: float) -> float:
    """Percentage distance of *price* from *reference* (clamped to ±1)."""
    if reference == 0.0 or not math.isfinite(reference):
        return 0.0
    return max(min((price - reference) / reference, 1.0), -1.0)


def _slope(series: pd.Series, n: int, price: float) -> float:
    """(last - n_bars_ago) / price, clamped to ±1."""
    if len(series) < n + 1 or price == 0.0:
        return 0.0
    v = float(series.iloc[-1]) - float(series.iloc[-n])
    return max(min(v / price, 1.0), -1.0)


def _multi_tf_return(close_5m: pd.Series, bars: int) -> float:
    """log return over *bars* 5-minute candles, clamped to ±1."""
    if len(close_5m) < bars + 1:
        return 0.0
    try:
        ref = float(close_5m.iloc[-bars])
        now = float(close_5m.iloc[-1])
        if ref <= 0:
            return 0.0
        r = math.log(now / ref)
        return max(min(r, 1.0), -1.0)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# extract_tabular_features
# ---------------------------------------------------------------------------

def extract_tabular_features(
    df_5m: pd.DataFrame,
    symbol: str,
    btc_df_5m: pd.DataFrame | None,
    funding_rate: float,
    funding_score: float,
    ob_imbalance: float,
    spread_pct: float,
    bid_ask_wall_ratio: float,
    onchain_composite: float,
    exchange_reserve_trend: float,
    regime_id: int,
    regime_hours: float,
) -> np.ndarray:
    """Extract 65 tabular features from 5-minute candle data.

    Parameters
    ----------
    df_5m               : DataFrame with open/high/low/close/volume, DatetimeIndex
    symbol              : trading pair symbol (e.g. "ETHUSDT")
    btc_df_5m           : BTC 5m candles for cross-asset features (may be None)
    funding_rate        : current perpetual funding rate (0 in backtest)
    funding_score       : aggregated funding signal (0 in backtest)
    ob_imbalance        : order-book bid/ask imbalance (0 in backtest)
    spread_pct          : bid-ask spread as pct of mid-price (0 in backtest)
    bid_ask_wall_ratio  : ratio of bid walls to ask walls (0 in backtest)
    onchain_composite   : on-chain composite signal (0 in backtest)
    exchange_reserve_trend : exchange reserve trend signal (0 in backtest)
    regime_id           : market regime class 0-4
    regime_hours        : hours the bot has been in the current regime

    Returns
    -------
    np.ndarray of shape (65,) and dtype float32.
    All zeros when len(df_5m) < MIN_BARS_5M.
    """
    features = np.zeros(N_TABULAR, dtype=np.float32)

    if df_5m is None or len(df_5m) < MIN_BARS_5M:
        return features

    try:
        # ----------------------------------------------------------------
        # Resample to higher timeframes
        # ----------------------------------------------------------------
        df_1h = df_5m.resample("1h").agg(_RESAMPLE_OHLCV).dropna()
        df_4h = df_5m.resample("4h").agg(_RESAMPLE_OHLCV).dropna()
        df_1d = df_5m.resample("1D").agg(_RESAMPLE_OHLCV).dropna()

        if len(df_1h) < 24 or len(df_4h) < 6 or len(df_1d) < 2:
            return features

        close_5m = df_5m["close"]
        price = float(close_5m.iloc[-1])
        if price <= 0 or not math.isfinite(price):
            return features

        # ================================================================
        # 0-8  Price action
        # ================================================================

        # 0: RSI 1h (scaled 0-1)
        rsi_1h = rsi(df_1h["close"])
        features[0] = float(np.clip(_safe_last(rsi_1h) / 100.0, 0.0, 1.0))

        # 1: RSI 4h (scaled 0-1)
        rsi_4h = rsi(df_4h["close"])
        features[1] = float(np.clip(_safe_last(rsi_4h) / 100.0, 0.0, 1.0))

        # 2: MACD histogram 1h (normalised by price)
        macd_1h = macd(df_1h["close"])
        hist_1h = _safe_last(macd_1h["histogram"])
        features[2] = float(np.clip(hist_1h / price * 100.0, -1.0, 1.0))

        # 3: MACD histogram 4h (normalised by price)
        macd_4h = macd(df_4h["close"])
        hist_4h = _safe_last(macd_4h["histogram"])
        features[3] = float(np.clip(hist_4h / price * 100.0, -1.0, 1.0))

        # 4: MACD histogram sign (1h): +1 / -1
        features[4] = 1.0 if hist_1h > 0 else -1.0

        # 5-6: Stochastic K/D (1h, scaled 0-1)
        stoch_1h = stochastic(df_1h["high"], df_1h["low"], df_1h["close"])
        features[5] = float(np.clip(_safe_last(stoch_1h["k"]) / 100.0, 0.0, 1.0))
        features[6] = float(np.clip(_safe_last(stoch_1h["d"]) / 100.0, 0.0, 1.0))

        # 7: CCI 1h (clamped ±3 → ±1)
        cci_1h = cci(df_1h["high"], df_1h["low"], df_1h["close"])
        features[7] = float(np.clip(_safe_last(cci_1h) / 300.0, -1.0, 1.0))

        # 8: RSI 5m (scaled 0-1)
        rsi_5m = rsi(close_5m)
        features[8] = float(np.clip(_safe_last(rsi_5m) / 100.0, 0.0, 1.0))

        # ================================================================
        # 9-14  Volatility
        # ================================================================

        # 9: ATR% 1h (ATR / close, clamped 0-1)
        atr_1h = atr(df_1h["high"], df_1h["low"], df_1h["close"])
        atr_pct_1h = _safe_last(atr_1h) / _safe_last(df_1h["close"], 1.0)
        features[9] = float(np.clip(atr_pct_1h / 0.05, 0.0, 1.0))  # 5% = 1.0

        # 10: ATR% 4h
        atr_4h = atr(df_4h["high"], df_4h["low"], df_4h["close"])
        atr_pct_4h = _safe_last(atr_4h) / _safe_last(df_4h["close"], 1.0)
        features[10] = float(np.clip(atr_pct_4h / 0.10, 0.0, 1.0))  # 10% = 1.0

        # 11-12: Bollinger bandwidth and %b (1h)
        bb_1h = bollinger_bands(df_1h["close"])
        features[11] = float(np.clip(_safe_last(bb_1h["bandwidth"]) / 0.2, 0.0, 1.0))
        features[12] = float(np.clip(_safe_last(bb_1h["pct_b"]), -0.5, 1.5))  # allow mild OOB

        # 13: Keltner channel position (1h): (close - lower) / (upper - lower)
        kc_1h = keltner_channels(df_1h["high"], df_1h["low"], df_1h["close"])
        kc_upper = _safe_last(kc_1h["upper"])
        kc_lower = _safe_last(kc_1h["lower"])
        kc_range = kc_upper - kc_lower
        if kc_range > 0:
            features[13] = float(np.clip((price - kc_lower) / kc_range, 0.0, 1.0))
        else:
            features[13] = 0.5

        # 14: ATR ratio 1h/4h (current atr_1h vs mean atr_1h)
        atr_1h_mean = atr_1h.dropna().mean()
        if atr_1h_mean > 0:
            features[14] = float(np.clip(_safe_last(atr_1h) / atr_1h_mean, 0.0, 3.0) / 3.0)

        # ================================================================
        # 15-21  Trend
        # ================================================================

        close_1h = df_1h["close"]

        # 15: EMA20 distance (1h)
        ema20_1h = ema(close_1h, 20)
        features[15] = float(np.clip(_pct_distance(price, _safe_last(ema20_1h, price)), -1.0, 1.0))

        # 16: EMA50 distance (1h)
        ema50_1h = ema(close_1h, 50)
        features[16] = float(np.clip(_pct_distance(price, _safe_last(ema50_1h, price)), -1.0, 1.0))

        # 17: EMA200 distance (1h)
        ema200_1h = ema(close_1h, 200)
        features[17] = float(np.clip(_pct_distance(price, _safe_last(ema200_1h, price)), -1.0, 1.0))

        # 18: EMA50 slope 4h (10-bar change / price)
        ema50_4h = ema(df_4h["close"], 50)
        features[18] = float(np.clip(_slope(ema50_4h, 10, price), -1.0, 1.0))

        # 19: EMA50 slope 1d (5-bar change / price)
        if len(df_1d) >= 6:
            ema50_1d = ema(df_1d["close"], 50)
            features[19] = float(np.clip(_slope(ema50_1d, 5, price), -1.0, 1.0))

        # 20: ADX 4h (scaled 0-1)
        adx_4h = adx(df_4h["high"], df_4h["low"], df_4h["close"])
        features[20] = float(np.clip(_safe_last(adx_4h) / 60.0, 0.0, 1.0))

        # 21: Supertrend direction (1h): +1 bullish / -1 bearish
        if len(df_1h) >= 20:
            st_1h = supertrend(df_1h["high"], df_1h["low"], df_1h["close"])
            features[21] = float(np.clip(_safe_last(st_1h), -1.0, 1.0))

        # ================================================================
        # 22-26  Volume
        # ================================================================

        vol_5m = df_5m["volume"]

        # 22: Volume surge ratio (5m, capped at 3)
        vsr_5m = volume_surge_ratio(vol_5m)
        features[22] = float(np.clip(_safe_last(vsr_5m) / 3.0, 0.0, 1.0))

        # 23: OBV trend — slope of OBV over last 20 bars (normalised)
        obv_5m = obv(close_5m, vol_5m)
        obv_arr = obv_5m.dropna()
        if len(obv_arr) >= 20:
            obv_slope = float(obv_arr.iloc[-1]) - float(obv_arr.iloc[-20])
            obv_scale = float(obv_arr.iloc[-20:].abs().max()) + 1e-9
            features[23] = float(np.clip(obv_slope / obv_scale, -1.0, 1.0))

        # 24: CMF 5m (already -1 to 1)
        cmf_5m = cmf(df_5m["high"], df_5m["low"], close_5m, vol_5m)
        features[24] = float(np.clip(_safe_last(cmf_5m), -1.0, 1.0))

        # 25: VWAP distance (5m: % from vwap, clamped ±5%)
        vwap_5m = vwap(df_5m["high"], df_5m["low"], close_5m, vol_5m)
        features[25] = float(np.clip(_pct_distance(price, _safe_last(vwap_5m, price)) / 0.05, -1.0, 1.0))

        # 26: Volume profile support score (-1 to 1)
        vps = volume_profile_support(close_5m, vol_5m)
        features[26] = float(np.clip(_safe_last(vps), -1.0, 1.0))

        # ================================================================
        # 27-28  Divergence
        # ================================================================

        rsi_div = rsi_divergence(close_5m)
        features[27] = float(np.clip(_safe_last(rsi_div), -1.0, 1.0))

        vol_div = volume_divergence(close_5m, vol_5m)
        features[28] = float(np.clip(_safe_last(vol_div), -1.0, 1.0))

        # ================================================================
        # 29-34  Regime (one-hot 5 classes + hours in regime)
        # ================================================================

        rid = int(np.clip(regime_id, 0, 4))
        features[29 + rid] = 1.0
        features[34] = float(np.clip(regime_hours / 168.0, 0.0, 1.0))  # max 1 week

        # ================================================================
        # 35-40  Multi-TF returns (5m bars: 12=1h, 48=4h, 144=12h, 288=24h, 864=72h, 2016=7d)
        # ================================================================

        features[35] = float(_multi_tf_return(close_5m, 12))    # 1h
        features[36] = float(_multi_tf_return(close_5m, 48))    # 4h
        features[37] = float(_multi_tf_return(close_5m, 144))   # 12h
        features[38] = float(_multi_tf_return(close_5m, 288))   # 24h
        features[39] = float(_multi_tf_return(close_5m, 864))   # 72h
        features[40] = float(_multi_tf_return(close_5m, 2016))  # 7d

        # ================================================================
        # 41-44  Candle structure (last 96 bars)
        # ================================================================

        window_96 = df_5m.tail(96)
        o = window_96["open"]
        h = window_96["high"]
        lo = window_96["low"]
        c = window_96["close"]

        body = (c - o).abs()
        candle_range = (h - lo).replace(0, np.nan)
        body_ratio = (body / candle_range).fillna(0.5)
        features[41] = float(np.clip(body_ratio.mean(), 0.0, 1.0))

        upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
        lower_wick = pd.concat([o, c], axis=1).min(axis=1) - lo
        uw_ratio = (upper_wick / candle_range).fillna(0.0)
        lw_ratio = (lower_wick / candle_range).fillna(0.0)
        features[42] = float(np.clip(uw_ratio.mean(), 0.0, 1.0))
        features[43] = float(np.clip(lw_ratio.mean(), 0.0, 1.0))

        # Consecutive green/red count (clamped 0-1 over 20)
        direction = np.where(c.values > o.values, 1, -1)
        streak = 0
        last_dir = direction[-1]
        for d in reversed(direction):
            if d == last_dir:
                streak += 1
            else:
                break
        features[44] = float(np.clip(streak / 20.0, 0.0, 1.0)) * last_dir

        # ================================================================
        # 45-50  Time (cyclic encoding)
        # ================================================================

        last_ts = df_5m.index[-1]
        if hasattr(last_ts, "hour"):
            features[45] = float(math.sin(2 * math.pi * last_ts.hour / 24.0))
            features[46] = float(math.cos(2 * math.pi * last_ts.hour / 24.0))
            features[47] = float(math.sin(2 * math.pi * last_ts.dayofweek / 7.0))
            features[48] = float(math.cos(2 * math.pi * last_ts.dayofweek / 7.0))
            features[49] = float(math.sin(2 * math.pi * (last_ts.month - 1) / 12.0))
            features[50] = float(math.cos(2 * math.pi * (last_ts.month - 1) / 12.0))

        # ================================================================
        # 51-54  Coin markers
        # ================================================================

        profile = get_coin_profile(symbol)
        features[51] = float(profile["cap_tier"])
        features[52] = float(profile["vol_class"])
        features[53] = float(profile["age"])
        features[54] = float(profile["is_btc"])

        # ================================================================
        # 55-59  Funding / order-book (zero-filled in backtest)
        # ================================================================

        features[55] = float(np.clip(funding_rate / 0.01, -1.0, 1.0))   # 1% = max
        features[56] = float(np.clip(funding_score, -1.0, 1.0))
        features[57] = float(np.clip(ob_imbalance, -1.0, 1.0))
        features[58] = float(np.clip(spread_pct / 0.01, 0.0, 1.0))      # 1% = max
        features[59] = float(np.clip(bid_ask_wall_ratio / 5.0, 0.0, 1.0))

        # ================================================================
        # 60-61  On-chain (zero-filled in backtest)
        # ================================================================

        features[60] = float(np.clip(onchain_composite, -1.0, 1.0))
        features[61] = float(np.clip(exchange_reserve_trend, -1.0, 1.0))

        # ================================================================
        # 62-64  Cross-asset (BTC)
        # ================================================================

        if btc_df_5m is not None and len(btc_df_5m) >= 13:
            btc_close = btc_df_5m["close"]
            features[62] = float(_multi_tf_return(btc_close, 12))   # BTC 1h return
            features[63] = float(_multi_tf_return(btc_close, 48))   # BTC 4h return

            # BTC correlation (close 5m, 96-bar window)
            if len(close_5m) >= 96 and len(btc_close) >= 96:
                try:
                    s1 = close_5m.iloc[-96:].reset_index(drop=True)
                    s2 = btc_close.iloc[-96:].reset_index(drop=True)
                    corr = float(s1.corr(s2))
                    features[64] = float(np.clip(corr if math.isfinite(corr) else 0.0, -1.0, 1.0))
                except Exception:
                    features[64] = 0.0

    except Exception as exc:
        logger.warning("extract_tabular_features failed for %s: %s", symbol, exc, exc_info=True)
        return np.zeros(N_TABULAR, dtype=np.float32)

    # Final sanitisation
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features.astype(np.float32)


# ---------------------------------------------------------------------------
# build_lstm_sequence
# ---------------------------------------------------------------------------

def build_lstm_sequence(df_5m: pd.DataFrame) -> np.ndarray:
    """Build a (96, 7) LSTM input sequence from the last 96 five-minute candles.

    Channels
    --------
    0: Normalized close — % change from the first bar in window (clamped ±20%)
    1: Normalized volume — ratio to 20-bar rolling average (clamped 0-5)
    2: High-low range normalised by close (clamped 0-5%)
    3: Body direction — +1 green candle, -1 red candle
    4: RSI 5m scaled 0-1
    5: Close vs EMA20 distance (%, clamped ±5%)
    6: Volume surge ratio (clamped 0-3)

    Returns
    -------
    np.ndarray of shape (LSTM_WINDOW, LSTM_CHANNELS) = (96, 7), dtype float32.
    All zeros when len(df_5m) < LSTM_MIN_BARS.
    """
    result = np.zeros((LSTM_WINDOW, LSTM_CHANNELS), dtype=np.float32)

    if df_5m is None or len(df_5m) < LSTM_MIN_BARS:
        return result

    try:
        # Use the last (LSTM_WINDOW + 20) bars so indicators have warm-up data
        buf = df_5m.iloc[-(LSTM_WINDOW + 20):]
        close = buf["close"]
        high = buf["high"]
        low = buf["low"]
        open_ = buf["open"]
        volume = buf["volume"]

        # ---- indicators over the buffer ----
        rsi_s = rsi(close)
        ema20_s = ema(close, 20)
        vol_avg = volume.rolling(20).mean().replace(0.0, np.nan).fillna(1.0)
        vol_ratio_all = (volume / vol_avg).fillna(1.0)
        vsr_all = volume_surge_ratio(volume)

        # Slice to last LSTM_WINDOW bars
        close_w = close.iloc[-LSTM_WINDOW:].values
        high_w = high.iloc[-LSTM_WINDOW:].values
        low_w = low.iloc[-LSTM_WINDOW:].values
        open_w = open_.iloc[-LSTM_WINDOW:].values
        volume_w = volume.iloc[-LSTM_WINDOW:].values
        rsi_w = rsi_s.iloc[-LSTM_WINDOW:].values
        ema20_w = ema20_s.iloc[-LSTM_WINDOW:].values
        vol_ratio_w = vol_ratio_all.iloc[-LSTM_WINDOW:].values
        vsr_w = vsr_all.iloc[-LSTM_WINDOW:].values

        ref_close = close_w[0] if close_w[0] > 0 else 1.0

        for i in range(LSTM_WINDOW):
            c = float(close_w[i])
            h = float(high_w[i])
            lo = float(low_w[i])
            o = float(open_w[i])

            # 0: normalized close
            result[i, 0] = float(np.clip((c - ref_close) / ref_close, -0.2, 0.2))

            # 1: normalized volume (ratio to 20-bar avg, capped at 5)
            result[i, 1] = float(np.clip(vol_ratio_w[i], 0.0, 5.0))

            # 2: high-low range / close (capped at 0.05 = 5%)
            if c > 0:
                result[i, 2] = float(np.clip((h - lo) / c, 0.0, 0.05))

            # 3: body direction
            result[i, 3] = 1.0 if c >= o else -1.0

            # 4: RSI scaled 0-1
            r = float(rsi_w[i])
            result[i, 4] = float(np.clip(r / 100.0, 0.0, 1.0)) if math.isfinite(r) else 0.5

            # 5: EMA20 distance (%)
            e20 = float(ema20_w[i])
            if e20 > 0 and math.isfinite(e20):
                result[i, 5] = float(np.clip((c - e20) / e20, -0.05, 0.05))

            # 6: volume surge ratio (capped at 3)
            result[i, 6] = float(np.clip(vsr_w[i], 0.0, 3.0))

    except Exception as exc:
        logger.warning("build_lstm_sequence failed: %s", exc, exc_info=True)
        return np.zeros((LSTM_WINDOW, LSTM_CHANNELS), dtype=np.float32)

    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# extract_all_features (batch)
# ---------------------------------------------------------------------------

def extract_all_features(
    df_5m: pd.DataFrame,
    symbol: str,
    btc_df_5m: pd.DataFrame | None,
) -> Tuple[np.ndarray, np.ndarray, List]:
    """Batch feature extraction for training.

    Iterates through *df_5m* at every SIGNAL_EVERY bars and extracts both
    tabular features and an LSTM sequence at each point in time.

    Parameters
    ----------
    df_5m       : full history of 5-minute candles
    symbol      : trading pair symbol
    btc_df_5m   : BTC 5m candle history (may be None)

    Returns
    -------
    Tuple of:
        tabular_array   — shape (N, 65),  dtype float32
        sequence_array  — shape (N, 96, 7), dtype float32
        timestamps_list — list of N pandas Timestamps
    """
    empty = (
        np.empty((0, N_TABULAR), dtype=np.float32),
        np.empty((0, LSTM_WINDOW, LSTM_CHANNELS), dtype=np.float32),
        [],
    )

    if df_5m is None or len(df_5m) < MIN_BARS_5M:
        return empty

    tabular_list: List[np.ndarray] = []
    sequence_list: List[np.ndarray] = []
    timestamps: List = []

    n = len(df_5m)

    for i in range(MIN_BARS_5M, n, SIGNAL_EVERY):
        slice_5m = df_5m.iloc[:i]
        btc_slice = btc_df_5m.iloc[:i] if btc_df_5m is not None else None

        tab = extract_tabular_features(
            slice_5m, symbol, btc_slice,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0,
        )
        seq = build_lstm_sequence(slice_5m)

        tabular_list.append(tab)
        sequence_list.append(seq)
        timestamps.append(df_5m.index[i - 1])

    if not tabular_list:
        return empty

    return (
        np.array(tabular_list, dtype=np.float32),
        np.array(sequence_list, dtype=np.float32),
        timestamps,
    )
