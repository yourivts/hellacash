"""ML feature engineering — tabular features, LSTM sequences, and batch extraction.

This module provides three public functions used to feed the ML signal generator:

  extract_tabular_features  — 90-element float32 vector per candle snapshot
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

N_TABULAR: int = 90
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
    external_data: dict | None = None,
    regime_id: int = 0,
    regime_hours: float = 0.0,
) -> np.ndarray:
    """Extract 90 tabular features from 5-minute candle data.

    Parameters
    ----------
    df_5m          : DataFrame with open/high/low/close/volume, DatetimeIndex
    symbol         : trading pair symbol (e.g. "ETHUSDT")
    btc_df_5m      : BTC 5m candles for cross-asset features (may be None)
    external_data  : dict of external/live market data (funding, OI, on-chain, macro, etc.)
    regime_id      : market regime class 0-4
    regime_hours   : hours the bot has been in the current regime

    Returns
    -------
    np.ndarray of shape (90,) and dtype float32.
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
        features[51] = float(profile["cap_tier"]) / 3.0
        features[52] = float(profile["vol_class"]) / 2.0
        features[53] = float(np.clip(profile["age"] / 16.0, 0.0, 1.0))
        features[54] = float(profile["is_btc"])

        # ================================================================
        # 55-61  External data (live) — zero-filled in backtest
        # ================================================================

        ext = external_data or {}
        features[55] = np.clip(ext.get("funding_rate", 0.0), -1.0, 1.0)
        features[56] = np.clip(ext.get("funding_7d_avg", 0.0), -1.0, 1.0)
        features[57] = np.clip(ext.get("oi_change_24h", 0.0), -1.0, 1.0)
        features[58] = np.clip(ext.get("long_liq_24h", 0.0), -1.0, 1.0)
        features[59] = np.clip(ext.get("short_liq_24h", 0.0), -1.0, 1.0)
        features[60] = np.clip(ext.get("exchange_netflow", 0.0), -1.0, 1.0)
        features[61] = np.clip(ext.get("active_addr_change", 0.0), -1.0, 1.0)

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

        # ================================================================
        # 65-89  Extended external features
        # ================================================================

        features[65] = np.clip(ext.get("fear_greed", 0.0), 0.0, 1.0)
        features[66] = np.clip(ext.get("fear_greed_mom", 0.0), -1.0, 1.0)
        features[67] = np.clip(ext.get("gtrends_bitcoin", 0.0), 0.0, 1.0)
        features[68] = np.clip(ext.get("gtrends_crypto", 0.0), 0.0, 1.0)
        features[69] = np.clip(ext.get("dxy_return", 0.0), -1.0, 1.0)
        features[70] = np.clip(ext.get("sp500_return", 0.0), -1.0, 1.0)
        features[71] = np.clip(ext.get("gold_return", 0.0), -1.0, 1.0)
        features[72] = np.clip(ext.get("vix", 0.0), 0.0, 1.0)
        features[73] = np.clip(ext.get("treasury_10y", 0.0), 0.0, 1.0)
        features[74] = np.clip(ext.get("yield_spread", 0.0), -1.0, 1.0)
        features[75] = np.clip(ext.get("nvt", 0.0), 0.0, 1.0)
        features[76] = np.clip(ext.get("mvrv", 0.0), 0.0, 1.0)
        features[77] = np.clip(ext.get("sopr", 0.0), -1.0, 1.0)
        features[78] = np.clip(ext.get("puell", 0.0), 0.0, 1.0)
        features[79] = np.clip(ext.get("hashrate", 0.0), -1.0, 1.0)
        features[80] = np.clip(ext.get("eth_active_addr", 0.0), -1.0, 1.0)
        features[81] = np.clip(ext.get("stable_supply_change", 0.0), -1.0, 1.0)
        features[82] = np.clip(ext.get("tvl_change", 0.0), -1.0, 1.0)
        features[83] = np.clip(ext.get("oi_change_7d", 0.0), -1.0, 1.0)
        features[84] = np.clip(ext.get("liq_ratio", 0.0), 0.0, 1.0)
        features[85] = np.clip(ext.get("taker_buy_ratio", 0.0), 0.0, 1.0)
        features[86] = np.clip(ext.get("dvol", 0.0), 0.0, 1.0)
        features[87] = np.clip(ext.get("funding_24h_avg", 0.0), -1.0, 1.0)
        features[88] = np.clip(ext.get("btc_dom_change", 0.0), -1.0, 1.0)
        features[89] = np.clip(ext.get("stable_btc_ratio", 0.0), 0.0, 1.0)

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

        # Vectorized channel computation
        # 0: normalized close (% change from first bar)
        result[:, 0] = np.clip((close_w - ref_close) / ref_close, -0.2, 0.2)
        # 1: normalized volume (ratio to 20-bar avg, capped at 5)
        result[:, 1] = np.clip(vol_ratio_w, 0.0, 5.0)
        # 2: high-low range / close (capped at 0.05 = 5%)
        safe_close = np.where(close_w > 0, close_w, 1.0)
        result[:, 2] = np.clip((high_w - low_w) / safe_close, 0.0, 0.05)
        # 3: body direction (+1 green, -1 red)
        result[:, 3] = np.where(close_w >= open_w, 1.0, -1.0)
        # 4: RSI scaled 0-1
        rsi_clean = np.where(np.isfinite(rsi_w), rsi_w, 50.0)
        result[:, 4] = np.clip(rsi_clean / 100.0, 0.0, 1.0)
        # 5: close vs EMA20 distance (%)
        ema_safe = np.where((ema20_w > 0) & np.isfinite(ema20_w), ema20_w, close_w)
        result[:, 5] = np.clip((close_w - ema_safe) / ema_safe, -0.05, 0.05)
        # 6: volume surge ratio (capped at 3)
        result[:, 6] = np.clip(vsr_w, 0.0, 3.0)

    except Exception as exc:
        logger.warning("build_lstm_sequence failed: %s", exc, exc_info=True)
        return np.zeros((LSTM_WINDOW, LSTM_CHANNELS), dtype=np.float32)

    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result.astype(np.float32)


def build_lstm_sequences_batch(
    df_5m: pd.DataFrame, indices: np.ndarray,
) -> np.ndarray:
    """Build (N, 96, 7) LSTM sequences for all indices at once.

    Fully vectorized: precomputes indicators once, then uses advanced indexing
    to build all windows in a single operation. No Python loops.
    """
    n = len(df_5m)
    close = df_5m["close"].values.astype(np.float64)
    high = df_5m["high"].values.astype(np.float64)
    low = df_5m["low"].values.astype(np.float64)
    open_ = df_5m["open"].values.astype(np.float64)
    volume = df_5m["volume"].values.astype(np.float64)

    # Precompute indicators once on full series
    rsi_vals = rsi(df_5m["close"]).values.astype(np.float64)
    rsi_vals = np.where(np.isfinite(rsi_vals), rsi_vals, 50.0)

    ema20_vals = ema(df_5m["close"], 20).values.astype(np.float64)

    vol_avg = df_5m["volume"].rolling(20).mean().replace(0.0, np.nan).fillna(1.0).values.astype(np.float64)
    vol_ratio_arr = np.where(vol_avg > 0, volume / vol_avg, 1.0)

    vsr_vals = volume_surge_ratio(df_5m["volume"]).values.astype(np.float64)

    # Filter valid indices (need LSTM_WINDOW bars behind them)
    valid = indices[(indices >= LSTM_WINDOW - 1) & (indices < n)]
    N = len(valid)

    # Build (N, 96) index matrix: each row is [idx-95, idx-94, ..., idx]
    offsets = np.arange(-(LSTM_WINDOW - 1), 1)  # [-95, -94, ..., 0]
    win_idx = valid[:, None] + offsets[None, :]  # (N, 96)

    # Gather all windows at once via advanced indexing
    c = close[win_idx]       # (N, 96)
    h = high[win_idx]        # (N, 96)
    lo = low[win_idx]        # (N, 96)
    o = open_[win_idx]       # (N, 96)
    vr = vol_ratio_arr[win_idx]
    rs = rsi_vals[win_idx]
    em = ema20_vals[win_idx]
    vs = vsr_vals[win_idx]

    # ref_close = first bar in each window
    ref_close = c[:, 0:1]  # (N, 1) for broadcasting
    ref_close = np.where(ref_close > 0, ref_close, 1.0)

    safe_c = np.where(c > 0, c, 1.0)
    ema_safe = np.where((em > 0) & np.isfinite(em), em, c)

    # Build result: (N, 96, 7) — fully vectorized
    result = np.zeros((N, LSTM_WINDOW, LSTM_CHANNELS), dtype=np.float32)
    result[:, :, 0] = np.clip((c - ref_close) / ref_close, -0.2, 0.2)
    result[:, :, 1] = np.clip(vr, 0.0, 5.0)
    result[:, :, 2] = np.clip((h - lo) / safe_c, 0.0, 0.05)
    result[:, :, 3] = np.where(c >= o, 1.0, -1.0)
    result[:, :, 4] = np.clip(rs / 100.0, 0.0, 1.0)
    result[:, :, 5] = np.clip((c - ema_safe) / ema_safe, -0.05, 0.05)
    result[:, :, 6] = np.clip(vs, 0.0, 3.0)

    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result


# ---------------------------------------------------------------------------
# Batch precomputation (compute all indicators once, then index)
# ---------------------------------------------------------------------------

def _precompute_indicators(
    df_5m: pd.DataFrame, btc_df_5m: pd.DataFrame | None,
) -> dict:
    """Compute all technical indicators once on the full DataFrame.

    Returns a dict of numpy arrays, all aligned to df_5m's integer index.
    """
    n = len(df_5m)
    idx_5m = df_5m.index
    close_5m = df_5m["close"]
    vol_5m = df_5m["volume"]

    # Resample ONCE
    df_1h = df_5m.resample("1h").agg(_RESAMPLE_OHLCV).dropna()
    df_4h = df_5m.resample("4h").agg(_RESAMPLE_OHLCV).dropna()
    df_1d = df_5m.resample("1D").agg(_RESAMPLE_OHLCV).dropna()

    def ffill(series):
        """Forward-fill a resampled Series to 5m resolution."""
        return series.reindex(idx_5m, method="ffill").values.astype(np.float64)

    pc: dict = {}

    # ---- 1h indicators ----
    pc["rsi_1h"] = ffill(rsi(df_1h["close"]))

    macd_1h_d = macd(df_1h["close"])
    pc["macd_hist_1h"] = ffill(macd_1h_d["histogram"])

    stoch_1h_d = stochastic(df_1h["high"], df_1h["low"], df_1h["close"])
    pc["stoch_k_1h"] = ffill(stoch_1h_d["k"])
    pc["stoch_d_1h"] = ffill(stoch_1h_d["d"])

    pc["cci_1h"] = ffill(cci(df_1h["high"], df_1h["low"], df_1h["close"]))

    atr_1h_s = atr(df_1h["high"], df_1h["low"], df_1h["close"])
    pc["atr_1h"] = ffill(atr_1h_s)
    pc["close_1h"] = ffill(df_1h["close"])
    pc["atr_1h_exp_mean"] = ffill(atr_1h_s.expanding().mean())

    bb_1h_d = bollinger_bands(df_1h["close"])
    pc["bb_bandwidth_1h"] = ffill(bb_1h_d["bandwidth"])
    pc["bb_pct_b_1h"] = ffill(bb_1h_d["pct_b"])

    kc_1h_d = keltner_channels(df_1h["high"], df_1h["low"], df_1h["close"])
    pc["kc_upper_1h"] = ffill(kc_1h_d["upper"])
    pc["kc_lower_1h"] = ffill(kc_1h_d["lower"])

    pc["ema20_1h"] = ffill(ema(df_1h["close"], 20))
    pc["ema50_1h"] = ffill(ema(df_1h["close"], 50))
    pc["ema200_1h"] = ffill(ema(df_1h["close"], 200))

    if len(df_1h) >= 20:
        pc["supertrend_1h"] = ffill(supertrend(df_1h["high"], df_1h["low"], df_1h["close"]))
    else:
        pc["supertrend_1h"] = np.zeros(n)

    # ---- 4h indicators ----
    pc["rsi_4h"] = ffill(rsi(df_4h["close"]))

    macd_4h_d = macd(df_4h["close"])
    pc["macd_hist_4h"] = ffill(macd_4h_d["histogram"])

    atr_4h_s = atr(df_4h["high"], df_4h["low"], df_4h["close"])
    pc["atr_4h"] = ffill(atr_4h_s)
    pc["close_4h"] = ffill(df_4h["close"])

    # EMA50 slope (10-bar diff in 4h timeframe, then ffill)
    ema50_4h_s = ema(df_4h["close"], 50)
    pc["ema50_4h_diff10"] = ffill(ema50_4h_s - ema50_4h_s.shift(10))

    pc["adx_4h"] = ffill(adx(df_4h["high"], df_4h["low"], df_4h["close"]))

    # ---- 1d indicators ----
    if len(df_1d) >= 56:
        ema50_1d_s = ema(df_1d["close"], 50)
        pc["ema50_1d_diff5"] = ffill(ema50_1d_s - ema50_1d_s.shift(5))
    else:
        pc["ema50_1d_diff5"] = np.zeros(n)

    # ---- 5m indicators ----
    pc["rsi_5m"] = rsi(close_5m).values.astype(np.float64)
    pc["vol_surge_5m"] = volume_surge_ratio(vol_5m).values.astype(np.float64)

    obv_5m_s = obv(close_5m, vol_5m)
    obv_slope = obv_5m_s - obv_5m_s.shift(20)
    obv_abs_max = obv_5m_s.abs().rolling(20).max().fillna(1.0) + 1e-9
    pc["obv_slope_norm"] = (obv_slope / obv_abs_max).values.astype(np.float64)

    pc["cmf_5m"] = cmf(df_5m["high"], df_5m["low"], close_5m, vol_5m).values.astype(np.float64)

    # Rolling VWAP (2000-bar window to match original windowed behavior)
    tp = (df_5m["high"] + df_5m["low"] + close_5m) / 3
    vwap_num = (tp * vol_5m).rolling(2000, min_periods=1).sum()
    vwap_den = vol_5m.rolling(2000, min_periods=1).sum().replace(0, 1e-9)
    pc["vwap_5m"] = (vwap_num / vwap_den).values.astype(np.float64)

    pc["vps_5m"] = volume_profile_support(close_5m, vol_5m).values.astype(np.float64)
    pc["rsi_div_5m"] = rsi_divergence(close_5m).values.astype(np.float64)
    pc["vol_div_5m"] = volume_divergence(close_5m, vol_5m).values.astype(np.float64)

    pc["close_5m"] = close_5m.values.astype(np.float64)

    # Multi-TF returns (log return over shifted bars)
    for shift_bars, label in [
        (12, "1h"), (48, "4h"), (144, "12h"),
        (288, "24h"), (864, "72h"), (2016, "7d"),
    ]:
        shifted = close_5m.shift(shift_bars)
        log_ret = np.where(
            shifted.values > 0,
            np.log(close_5m.values / np.where(shifted.values > 0, shifted.values, 1.0)),
            0.0,
        )
        pc[f"return_{label}"] = np.clip(log_ret, -1.0, 1.0)

    # Candle structure (rolling 96-bar means)
    o, h, lo, c = df_5m["open"], df_5m["high"], df_5m["low"], close_5m
    body = (c - o).abs()
    candle_range = (h - lo).replace(0, np.nan)
    pc["body_ratio_96"] = (body / candle_range).fillna(0.5).rolling(96).mean().values.astype(np.float64)
    upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_wick = pd.concat([o, c], axis=1).min(axis=1) - lo
    pc["uw_ratio_96"] = (upper_wick / candle_range).fillna(0.0).rolling(96).mean().values.astype(np.float64)
    pc["lw_ratio_96"] = (lower_wick / candle_range).fillna(0.0).rolling(96).mean().values.astype(np.float64)

    # Consecutive green/red streak (single O(n) pass)
    direction = np.where(c.values > o.values, 1, -1)
    streak = np.ones(n, dtype=np.int32)
    for i in range(1, n):
        if direction[i] == direction[i - 1]:
            streak[i] = streak[i - 1] + 1
        else:
            streak[i] = 1
    pc["streak_feature"] = (np.clip(streak / 20.0, 0.0, 1.0) * direction).astype(np.float64)

    # Time features
    pc["hour"] = idx_5m.hour.values.astype(np.float64)
    pc["dayofweek"] = idx_5m.dayofweek.values.astype(np.float64)
    pc["month"] = idx_5m.month.values.astype(np.float64)

    # BTC cross-asset
    if btc_df_5m is not None and len(btc_df_5m) >= 49:
        btc_close = btc_df_5m["close"]
        # Compute on BTC's index, then align to main pair's timestamps
        btc_shifted_12 = btc_close.shift(12)
        btc_shifted_48 = btc_close.shift(48)
        btc_ret_1h = np.where(
            btc_shifted_12.values > 0,
            np.log(btc_close.values / np.where(btc_shifted_12.values > 0, btc_shifted_12.values, 1.0)),
            0.0,
        )
        btc_ret_4h = np.where(
            btc_shifted_48.values > 0,
            np.log(btc_close.values / np.where(btc_shifted_48.values > 0, btc_shifted_48.values, 1.0)),
            0.0,
        )
        btc_ret_1h_s = pd.Series(btc_ret_1h, index=btc_df_5m.index)
        btc_ret_4h_s = pd.Series(btc_ret_4h, index=btc_df_5m.index)
        pc["btc_return_1h"] = np.clip(btc_ret_1h_s.reindex(idx_5m, method="ffill").values, -1.0, 1.0)
        pc["btc_return_4h"] = np.clip(btc_ret_4h_s.reindex(idx_5m, method="ffill").values, -1.0, 1.0)

        # Rolling 96-bar correlation
        btc_aligned = btc_close.reindex(idx_5m, method="ffill")
        corr_96 = close_5m.rolling(96).corr(btc_aligned)
        pc["btc_corr_96"] = corr_96.values.astype(np.float64)
    else:
        pc["btc_return_1h"] = np.zeros(n)
        pc["btc_return_4h"] = np.zeros(n)
        pc["btc_corr_96"] = np.zeros(n)

    return pc


def _batch_extract_tabular(
    df_5m: pd.DataFrame,
    symbol: str,
    btc_df_5m: pd.DataFrame | None,
    sample_indices: np.ndarray,
    external_features: np.ndarray | None = None,
) -> np.ndarray:
    """Extract 90 tabular features at multiple sample points using precomputed indicators.

    Fully vectorized: indicators are computed once on the full DataFrame, then
    features for all sample points are assembled via numpy fancy-indexing.

    Parameters
    ----------
    df_5m              : full 5m DataFrame
    symbol             : trading pair symbol
    btc_df_5m          : BTC 5m DataFrame (may be None)
    sample_indices     : integer indices into df_5m (0-based)
    external_features  : optional (len(df_5m), 32) array of external data for training

    Returns
    -------
    np.ndarray of shape (len(sample_indices), 90), dtype float32.
    """
    logger.info("Precomputing indicators on %d bars for %s...", len(df_5m), symbol)
    pc = _precompute_indicators(df_5m, btc_df_5m)
    logger.info("Precomputation done, extracting features at %d sample points", len(sample_indices))

    idx = sample_indices
    ns = len(idx)
    features = np.zeros((ns, N_TABULAR), dtype=np.float64)

    price = pc["close_5m"][idx]
    safe_price = np.where(price > 0, price, 1.0)
    valid = (price > 0) & np.isfinite(price)

    def g(arr):
        """Get precomputed values at sample indices."""
        return arr[idx]

    # ---- 0-8: Price action ----
    features[:, 0] = np.clip(g(pc["rsi_1h"]) / 100.0, 0.0, 1.0)
    features[:, 1] = np.clip(g(pc["rsi_4h"]) / 100.0, 0.0, 1.0)
    features[:, 2] = np.clip(g(pc["macd_hist_1h"]) / safe_price * 100.0, -1.0, 1.0)
    features[:, 3] = np.clip(g(pc["macd_hist_4h"]) / safe_price * 100.0, -1.0, 1.0)
    features[:, 4] = np.where(g(pc["macd_hist_1h"]) > 0, 1.0, -1.0)
    features[:, 5] = np.clip(g(pc["stoch_k_1h"]) / 100.0, 0.0, 1.0)
    features[:, 6] = np.clip(g(pc["stoch_d_1h"]) / 100.0, 0.0, 1.0)
    features[:, 7] = np.clip(g(pc["cci_1h"]) / 300.0, -1.0, 1.0)
    features[:, 8] = np.clip(g(pc["rsi_5m"]) / 100.0, 0.0, 1.0)

    # ---- 9-14: Volatility ----
    close_1h = np.where(g(pc["close_1h"]) > 0, g(pc["close_1h"]), 1.0)
    close_4h = np.where(g(pc["close_4h"]) > 0, g(pc["close_4h"]), 1.0)
    features[:, 9] = np.clip(g(pc["atr_1h"]) / close_1h / 0.05, 0.0, 1.0)
    features[:, 10] = np.clip(g(pc["atr_4h"]) / close_4h / 0.10, 0.0, 1.0)
    features[:, 11] = np.clip(g(pc["bb_bandwidth_1h"]) / 0.2, 0.0, 1.0)
    features[:, 12] = np.clip(g(pc["bb_pct_b_1h"]), -0.5, 1.5)

    kc_upper = g(pc["kc_upper_1h"])
    kc_lower = g(pc["kc_lower_1h"])
    kc_range = kc_upper - kc_lower
    features[:, 13] = np.where(kc_range > 0, np.clip((price - kc_lower) / kc_range, 0.0, 1.0), 0.5)

    atr_exp = np.where(g(pc["atr_1h_exp_mean"]) > 0, g(pc["atr_1h_exp_mean"]), 1.0)
    features[:, 14] = np.clip(g(pc["atr_1h"]) / atr_exp, 0.0, 3.0) / 3.0

    # ---- 15-21: Trend ----
    ema20 = g(pc["ema20_1h"])
    ema20_s = np.where((ema20 > 0) & np.isfinite(ema20), ema20, price)
    features[:, 15] = np.clip((price - ema20_s) / np.where(ema20_s > 0, ema20_s, 1.0), -1.0, 1.0)

    ema50 = g(pc["ema50_1h"])
    ema50_s = np.where((ema50 > 0) & np.isfinite(ema50), ema50, price)
    features[:, 16] = np.clip((price - ema50_s) / np.where(ema50_s > 0, ema50_s, 1.0), -1.0, 1.0)

    ema200 = g(pc["ema200_1h"])
    ema200_s = np.where((ema200 > 0) & np.isfinite(ema200), ema200, price)
    features[:, 17] = np.clip((price - ema200_s) / np.where(ema200_s > 0, ema200_s, 1.0), -1.0, 1.0)

    features[:, 18] = np.clip(g(pc["ema50_4h_diff10"]) / safe_price, -1.0, 1.0)
    features[:, 19] = np.clip(g(pc["ema50_1d_diff5"]) / safe_price, -1.0, 1.0)
    features[:, 20] = np.clip(g(pc["adx_4h"]) / 60.0, 0.0, 1.0)
    features[:, 21] = np.clip(g(pc["supertrend_1h"]), -1.0, 1.0)

    # ---- 22-26: Volume ----
    features[:, 22] = np.clip(g(pc["vol_surge_5m"]) / 3.0, 0.0, 1.0)
    features[:, 23] = np.clip(g(pc["obv_slope_norm"]), -1.0, 1.0)
    features[:, 24] = np.clip(g(pc["cmf_5m"]), -1.0, 1.0)

    vwap_val = g(pc["vwap_5m"])
    vwap_s = np.where(vwap_val > 0, vwap_val, price)
    features[:, 25] = np.clip(
        np.where(vwap_s > 0, (price - vwap_s) / vwap_s, 0.0) / 0.05, -1.0, 1.0,
    )
    features[:, 26] = np.clip(g(pc["vps_5m"]), -1.0, 1.0)

    # ---- 27-28: Divergence ----
    features[:, 27] = np.clip(g(pc["rsi_div_5m"]), -1.0, 1.0)
    features[:, 28] = np.clip(g(pc["vol_div_5m"]), -1.0, 1.0)

    # ---- 29-34: Regime (zero in batch — regime_id=0, regime_hours=0) ----
    features[:, 29] = 1.0  # one-hot for regime_id=0

    # ---- 35-40: Multi-TF returns ----
    features[:, 35] = g(pc["return_1h"])
    features[:, 36] = g(pc["return_4h"])
    features[:, 37] = g(pc["return_12h"])
    features[:, 38] = g(pc["return_24h"])
    features[:, 39] = g(pc["return_72h"])
    features[:, 40] = g(pc["return_7d"])

    # ---- 41-44: Candle structure ----
    features[:, 41] = np.clip(g(pc["body_ratio_96"]), 0.0, 1.0)
    features[:, 42] = np.clip(g(pc["uw_ratio_96"]), 0.0, 1.0)
    features[:, 43] = np.clip(g(pc["lw_ratio_96"]), 0.0, 1.0)
    features[:, 44] = g(pc["streak_feature"])

    # ---- 45-50: Time (cyclic) ----
    hour = g(pc["hour"])
    dow = g(pc["dayofweek"])
    month = g(pc["month"])
    features[:, 45] = np.sin(2 * np.pi * hour / 24.0)
    features[:, 46] = np.cos(2 * np.pi * hour / 24.0)
    features[:, 47] = np.sin(2 * np.pi * dow / 7.0)
    features[:, 48] = np.cos(2 * np.pi * dow / 7.0)
    features[:, 49] = np.sin(2 * np.pi * (month - 1) / 12.0)
    features[:, 50] = np.cos(2 * np.pi * (month - 1) / 12.0)

    # ---- 51-54: Coin markers (constant per symbol) ----
    profile = get_coin_profile(symbol)
    features[:, 51] = float(profile["cap_tier"]) / 3.0
    features[:, 52] = float(profile["vol_class"]) / 2.0
    features[:, 53] = float(np.clip(profile["age"] / 16.0, 0.0, 1.0))
    features[:, 54] = float(profile["is_btc"])

    # ---- 55-61: External data (populated from external_features during training) ----
    # ---- 65-89: Extended external features ----
    if external_features is not None:
        # external_features shape: (len(df_5m), 32) — 7 cols for indices 55-61 + 25 cols for 65-89
        ext_at_samples = external_features[idx]  # (ns, 32)
        features[:, 55:62] = ext_at_samples[:, :7]   # funding, OI, liq, netflow, active_addr
        features[:, 65:90] = ext_at_samples[:, 7:32]  # 25 extended features

    # ---- 62-64: Cross-asset (BTC) ----
    features[:, 62] = g(pc["btc_return_1h"])
    features[:, 63] = g(pc["btc_return_4h"])
    features[:, 64] = np.clip(g(pc["btc_corr_96"]), -1.0, 1.0)

    # Zero out invalid prices
    features[~valid] = 0.0

    # Final sanitisation
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features.astype(np.float32)


# ---------------------------------------------------------------------------
# extract_all_features (batch)
# ---------------------------------------------------------------------------

def extract_all_features(
    df_5m: pd.DataFrame,
    symbol: str,
    btc_df_5m: pd.DataFrame | None,
    external_features: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, List]:
    """Batch feature extraction for training.

    Computes all indicators once on the full DataFrame, then extracts tabular
    features via vectorized indexing and LSTM sequences per sample point.

    Parameters
    ----------
    df_5m              : full history of 5-minute candles
    symbol             : trading pair symbol
    btc_df_5m          : BTC 5m candle history (may be None)
    external_features  : optional (len(df_5m), 32) array of external data for training

    Returns
    -------
    Tuple of:
        tabular_array   — shape (N, 90),  dtype float32
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

    n = len(df_5m)

    # Determine sample indices (every SIGNAL_EVERY bars, starting at MIN_BARS_5M)
    sample_indices = np.arange(MIN_BARS_5M, n, SIGNAL_EVERY)
    if len(sample_indices) == 0:
        return empty

    # Vectorized tabular feature extraction (indicators computed once)
    tabular = _batch_extract_tabular(df_5m, symbol, btc_df_5m, sample_indices, external_features)

    # Batch LSTM sequence builder (vectorized, indicators computed once)
    sequences = build_lstm_sequences_batch(df_5m, sample_indices)

    timestamps = [df_5m.index[i - 1] for i in sample_indices]

    return (
        tabular,
        sequences,
        timestamps,
    )
