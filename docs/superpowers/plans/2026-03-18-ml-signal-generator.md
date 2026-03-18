# ML Signal Generator Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace 8 hand-coded trading strategies with a stacked LSTM+XGBoost ensemble that predicts directional probabilities at 6 time horizons, gated by the existing RL Signal Evaluator.

**Architecture:** LSTM encodes 96-bar price sequences into 16-dim embeddings. These embeddings are concatenated with 65 tabular features (indicators, regime, time, coin markers) to form 81-dim input for 12 XGBoost binary classifiers (up/down × 6 horizons). The RL Signal Evaluator gates ML signals and adjusts execution parameters via online PPO.

**Tech Stack:** Python, XGBoost, PyTorch (LSTM), numpy, pandas, existing indicator library

**Spec:** `docs/superpowers/specs/2026-03-18-ml-signal-generator-design.md`

---

## File Structure

### New files

| File | Responsibility |
|------|---------------|
| `bot/learning/ml_features.py` | Compute 65 tabular features + 96×7 LSTM sequences from candle DataFrames |
| `bot/learning/lstm_embedder.py` | LSTM model definition, self-supervised training, embedding extraction, save/load |
| `bot/learning/ml_signal_generator.py` | Load trained LSTM + XGBoost models, expose `predict()` → 12 probabilities |
| `scripts/train_ml_signals.py` | End-to-end training pipeline: fetch data → features → train LSTM → train XGBoost → save |
| `tests/test_ml_features.py` | Tests for tabular feature extraction and LSTM sequence building |
| `tests/test_lstm_embedder.py` | Tests for LSTM model, training, embedding extraction |
| `tests/test_ml_signal_generator.py` | Tests for prediction interface, model loading |
| `tests/test_train_ml_signals.py` | Tests for training pipeline components |

### Modified files

| File | Changes |
|------|---------|
| `bot/learning/rl_signal_evaluator.py` | OBS_DIM 32→30, new `build_ml_signal_obs()`, remove `STRATEGY_IDS` and old `build_signal_obs()` |
| `bot/backtest/engine.py` | Add `ml_signal_generator` param, new ML signal path alongside existing strategy path |
| `bot/learning/gpu_backtest_kernel.py` | Update `SIGNAL_EVERY` from 12 to 6 |
| `bot/learning/gpu_vec_env.py` | Update `_OBS_DIM` from 32 to 30 |
| `bot/learning/rl_environment.py` | Update `obs_dim` from 32 to 30 |
| `bot/trading_loop.py` | Use `MLSignalGenerator` when available instead of `StrategyRouter` |
| `bot/main.py` | Wire `MLSignalGenerator` into live loop |
| `scripts/pretrain_signal_evaluator.py` | Update to use ML signal observations |
| `scripts/rl_vs_champion.py` | Add ML signal generator test mode |

---

## Chunk 1: Feature Engineering

### Task 1: Tabular Feature Extraction (65 features)

**Files:**
- Create: `bot/learning/ml_features.py`
- Create: `tests/test_ml_features.py`
- Reference: `bot/indicators/momentum.py`, `bot/indicators/trend.py`, `bot/indicators/volatility.py`, `bot/indicators/volume.py`, `bot/indicators/divergence.py`
- Reference: `bot/learning/rl_signal_evaluator.py:49-65` (COIN_PROFILES)

This task creates the tabular feature extraction function. It takes a 5m candle DataFrame and produces a 65-dim feature vector at each SIGNAL_EVERY interval.

- [ ] **Step 1: Write test for tabular feature shape and bounds**

```python
# tests/test_ml_features.py
"""Tests for ML feature engineering."""
import numpy as np
import pandas as pd
import pytest


def _make_5m_candles(bars: int = 2000) -> pd.DataFrame:
    """Generate synthetic 5m candle data for testing."""
    np.random.seed(42)
    timestamps = pd.date_range("2024-01-01", periods=bars, freq="5min")
    base = 50000.0
    returns = np.random.randn(bars) * 0.001
    close = base * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.randn(bars)) * 0.002)
    low = close * (1 - np.abs(np.random.randn(bars)) * 0.002)
    open_ = close * (1 + np.random.randn(bars) * 0.001)
    volume = np.random.exponential(100, bars)
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume,
    }, index=timestamps)


class TestExtractTabularFeatures:
    def test_output_shape_is_65(self):
        from bot.learning.ml_features import extract_tabular_features
        df = _make_5m_candles(2000)
        features = extract_tabular_features(df, symbol="BTC-EUR")
        assert features.shape == (65,), f"Expected 65 features, got {features.shape}"

    def test_output_dtype_float32(self):
        from bot.learning.ml_features import extract_tabular_features
        df = _make_5m_candles(2000)
        features = extract_tabular_features(df, symbol="BTC-EUR")
        assert features.dtype == np.float32

    def test_features_bounded(self):
        from bot.learning.ml_features import extract_tabular_features
        df = _make_5m_candles(2000)
        features = extract_tabular_features(df, symbol="BTC-EUR")
        assert np.all(np.isfinite(features)), "Features contain NaN or Inf"

    def test_different_symbols_change_coin_features(self):
        from bot.learning.ml_features import extract_tabular_features
        df = _make_5m_candles(2000)
        btc = extract_tabular_features(df, symbol="BTC-EUR")
        eth = extract_tabular_features(df, symbol="ETH-EUR")
        # Coin marker features (last 4) should differ
        assert not np.array_equal(btc[-4:], eth[-4:])

    def test_insufficient_data_returns_zeros(self):
        from bot.learning.ml_features import extract_tabular_features
        df = _make_5m_candles(50)  # too few bars
        features = extract_tabular_features(df, symbol="BTC-EUR")
        assert features.shape == (65,)
        assert np.all(features == 0.0)

    def test_zero_filled_live_features(self):
        from bot.learning.ml_features import extract_tabular_features
        df = _make_5m_candles(2000)
        features = extract_tabular_features(df, symbol="BTC-EUR")
        # Funding/OB features (indices 55-59) should be zero without live data
        assert np.all(features[55:60] == 0.0)
        # On-chain features (indices 60-61) should be zero
        assert np.all(features[60:62] == 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ml_features.py::TestExtractTabularFeatures -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bot.learning.ml_features'`

- [ ] **Step 3: Implement tabular feature extraction**

Create `bot/learning/ml_features.py` with `extract_tabular_features()`. The function resamples 5m candles to 1h/4h/1d, computes all indicators, and assembles the 65-dim vector.

Feature index layout:
- 0-8: Price action (RSI 1h/4h, MACD hist 1h/4h, MACD sign, Stoch K/D, CCI)
- 9-14: Volatility (ATR% 1h/4h, BB bandwidth, BB %b, Keltner position, ATR ratio)
- 15-21: Trend (EMA20/50/200 distances, EMA50 slope 4h/1d, ADX 4h, Supertrend)
- 22-26: Volume (volume surge, OBV trend, CMF, VWAP dist, volume profile)
- 27-28: Divergence (RSI divergence, volume divergence)
- 29-34: Regime (one-hot 5 regimes + hours in regime)
- 35-40: Multi-TF returns (1h, 4h, 12h, 24h, 72h, 7d)
- 41-44: Candle structure (body ratio, upper/lower wick, consecutive green/red)
- 45-50: Time (hour sin/cos, dow sin/cos, month sin/cos)
- 51-54: Coin markers (cap tier, vol class, age, is_BTC)
- 55-59: Funding/OB (zero-filled in backtest)
- 60-61: On-chain (zero-filled in backtest)
- 62-64: Cross-asset (BTC returns 1h/4h, BTC correlation)

```python
"""ML feature engineering: tabular features and LSTM sequences."""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd

from bot.indicators.momentum import rsi, stochastic, cci
from bot.indicators.trend import ema, macd, adx, supertrend
from bot.indicators.volatility import atr, bollinger_bands, keltner_channels
from bot.indicators.volume import cmf, obv, vwap, volume_surge_ratio, volume_profile_support
from bot.indicators.divergence import rsi_divergence, volume_divergence

# Re-use coin profiles from rl_signal_evaluator
from bot.learning.rl_signal_evaluator import get_coin_profile

N_TABULAR = 65
MIN_BARS_5M = 1500  # minimum 5m bars needed (~5 days)

# Regime detection thresholds (must match engine.py)
_QUIET_ATR = 1.0
_VOLATILE_ATR = 4.0
_TRENDING_ADX = 24
_RANGING_ADX = 20


def _resample(df_5m: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample 5m OHLCV to a higher timeframe."""
    return df_5m.resample(rule).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()


def _safe(series: pd.Series, idx: int, default: float = 0.0) -> float:
    """Safely get a value from a series by integer position."""
    if idx < 0 or idx >= len(series):
        return default
    val = series.iloc[idx]
    return default if (val != val) else float(val)  # NaN check


def extract_tabular_features(
    df_5m: pd.DataFrame,
    symbol: str = "BTC-EUR",
    btc_df_5m: Optional[pd.DataFrame] = None,
    funding_rate: float = 0.0,
    funding_score: float = 0.0,
    ob_imbalance: float = 0.0,
    spread_pct: float = 0.0,
    bid_ask_wall_ratio: float = 0.0,
    onchain_composite: float = 0.0,
    exchange_reserve_trend: float = 0.0,
    regime_id: int = -1,
    regime_hours: float = 0.0,
) -> np.ndarray:
    """Extract 65 tabular features from 5m candle data.

    Args:
        df_5m: DataFrame with columns [open, high, low, close, volume], DatetimeIndex, 5m freq
        symbol: Trading pair for coin marker lookup
        btc_df_5m: BTC candles for cross-asset features (optional)
        funding_rate..exchange_reserve_trend: Live data (zero in backtests)
        regime_id: Current regime (0-4), -1 = auto-detect
        regime_hours: Hours in current regime

    Returns:
        np.ndarray of shape (65,) dtype float32. All zeros if insufficient data.
    """
    out = np.zeros(N_TABULAR, dtype=np.float32)

    if len(df_5m) < MIN_BARS_5M:
        return out

    # Resample to higher timeframes
    df_1h = _resample(df_5m, "1h")
    df_4h = _resample(df_5m, "4h")
    df_1d = _resample(df_5m, "1D")

    if len(df_1h) < 30 or len(df_4h) < 10 or len(df_1d) < 5:
        return out

    c1h, h1h, l1h, v1h = df_1h["close"], df_1h["high"], df_1h["low"], df_1h["volume"]
    c4h, h4h, l4h = df_4h["close"], df_4h["high"], df_4h["low"]
    c1d = df_1d["close"]
    c5m = df_5m["close"]

    # ── Price action (9 features, indices 0-8) ──
    rsi_1h = rsi(c1h, 14)
    rsi_4h = rsi(c4h, 14)
    macd_1h = macd(c1h)
    macd_4h = macd(c4h)
    stoch = stochastic(h1h, l1h, c1h)
    cci_1h = cci(h1h, l1h, c1h)

    out[0] = np.clip(_safe(rsi_1h, -1, 50) / 100.0, 0, 1)
    out[1] = np.clip(_safe(rsi_4h, -1, 50) / 100.0, 0, 1)
    out[2] = np.clip(_safe(macd_1h["histogram"], -1) / (abs(_safe(c1h, -1, 1)) * 0.01 + 1e-9), -1, 1)
    out[3] = np.clip(_safe(macd_4h["histogram"], -1) / (abs(_safe(c4h, -1, 1)) * 0.01 + 1e-9), -1, 1)
    out[4] = np.sign(_safe(macd_1h["histogram"], -1))
    out[5] = np.clip(_safe(stoch["k"], -1, 50) / 100.0, 0, 1)
    out[6] = np.clip(_safe(stoch["d"], -1, 50) / 100.0, 0, 1)
    out[7] = np.clip(_safe(cci_1h, -1) / 200.0, -1, 1)
    # Index 8: spare slot in price action → use RSI 5m for extra granularity
    rsi_5m = rsi(c5m, 14)
    out[8] = np.clip(_safe(rsi_5m, -1, 50) / 100.0, 0, 1)

    # ── Volatility (6 features, indices 9-14) ──
    atr_1h = atr(h1h, l1h, c1h)
    atr_4h = atr(h4h, l4h, c4h)
    bb = bollinger_bands(c1h)
    kc = keltner_channels(h1h, l1h, c1h)
    price = _safe(c1h, -1, 1)
    atr_1h_pct = _safe(atr_1h, -1) / price * 100.0 if price > 0 else 0
    atr_4h_pct = _safe(atr_4h, -1) / _safe(c4h, -1, 1) * 100.0

    out[9] = np.clip(atr_1h_pct / 10.0, 0, 1)
    out[10] = np.clip(atr_4h_pct / 10.0, 0, 1)
    out[11] = np.clip(_safe(bb["bandwidth"], -1) / 0.2, 0, 1)
    out[12] = np.clip(_safe(bb["pct_b"], -1, 0.5), 0, 1)
    # Keltner position: where is close relative to KC channel
    kc_upper = _safe(kc["upper"], -1, price)
    kc_lower = _safe(kc["lower"], -1, price)
    kc_range = kc_upper - kc_lower if kc_upper != kc_lower else 1e-9
    out[13] = np.clip((price - kc_lower) / kc_range, 0, 1)
    # ATR ratio: current / 20-bar median
    atr_median = float(atr_1h.iloc[-20:].median()) if len(atr_1h) >= 20 else _safe(atr_1h, -1, 1)
    out[14] = np.clip(_safe(atr_1h, -1) / (atr_median + 1e-9), 0, 3) / 3.0

    # ── Trend (7 features, indices 15-21) ──
    ema20 = ema(c1h, 20)
    ema50 = ema(c1h, 50)
    ema200 = ema(c1d, 200)
    ema50_4h = ema(c4h, 50)
    ema50_1d = ema(c1d, 50)
    adx_4h = adx(h4h, l4h, c4h)
    st = supertrend(h1h, l1h, c1h)

    out[15] = np.clip((price - _safe(ema20, -1, price)) / price * 100.0 / 5.0, -1, 1)
    out[16] = np.clip((price - _safe(ema50, -1, price)) / price * 100.0 / 10.0, -1, 1)
    out[17] = np.clip((_safe(c1d, -1, price) - _safe(ema200, -1, price)) / price * 100.0 / 20.0, -1, 1)
    # EMA50 slopes
    if len(ema50_4h) >= 4:
        slope_4h = (_safe(ema50_4h, -1) - _safe(ema50_4h, -4)) / (price + 1e-9) * 100.0
        out[18] = np.clip(slope_4h / 5.0, -1, 1)
    if len(ema50_1d) >= 4:
        slope_1d = (_safe(ema50_1d, -1) - _safe(ema50_1d, -4)) / (price + 1e-9) * 100.0
        out[19] = np.clip(slope_1d / 10.0, -1, 1)
    out[20] = np.clip(_safe(adx_4h, -1, 20) / 50.0, 0, 1)
    out[21] = _safe(st, -1, 1)  # +1 or -1

    # ── Volume (5 features, indices 22-26) ──
    vsr = volume_surge_ratio(v1h)
    obv_val = obv(c1h, v1h)
    cmf_val = cmf(h1h, l1h, c1h, v1h)
    vwap_val = vwap(h1h, l1h, c1h, v1h)
    vp = volume_profile_support(c1h, v1h)

    out[22] = np.clip(_safe(vsr, -1, 1) / 5.0, 0, 1)
    # OBV trend: slope of last 20 bars normalized
    if len(obv_val) >= 20:
        obv_slope = (_safe(obv_val, -1) - _safe(obv_val, -20)) / (abs(_safe(obv_val, -1)) + 1e-9)
        out[23] = np.clip(obv_slope, -1, 1)
    out[24] = np.clip(_safe(cmf_val, -1), -1, 1)
    # VWAP distance %
    vwap_v = _safe(vwap_val, -1, price)
    out[25] = np.clip((price - vwap_v) / price * 100.0 / 5.0, -1, 1) if price > 0 else 0
    out[26] = np.clip(_safe(vp, -1), -1, 1)

    # ── Divergence (2 features, indices 27-28) ──
    rsi_div = rsi_divergence(c1h)
    vol_div = volume_divergence(c1h, v1h)
    out[27] = _safe(rsi_div, -1)
    out[28] = _safe(vol_div, -1)

    # ── Regime (6 features, indices 29-34) ──
    if regime_id < 0:
        # Auto-detect from 4h data
        _atr_pct = atr_4h_pct
        _adx = _safe(adx_4h, -1, 20)
        if _atr_pct < _QUIET_ATR:
            regime_id = 0
        elif _atr_pct > _VOLATILE_ATR:
            regime_id = 1
        elif _adx > _TRENDING_ADX:
            regime_id = 2
        elif _adx < _RANGING_ADX:
            regime_id = 3
        else:
            regime_id = 4
    # One-hot encode (5 slots)
    if 0 <= regime_id <= 4:
        out[29 + regime_id] = 1.0
    out[34] = np.clip(regime_hours / 48.0, 0, 1)

    # ── Multi-TF returns (6 features, indices 35-40) ──
    if len(c5m) >= 2:
        curr = float(c5m.iloc[-1])
        for j, bars_back in enumerate([12, 48, 144, 288, 864, 2016]):
            if len(c5m) > bars_back:
                past = float(c5m.iloc[-bars_back - 1])
                ret = (curr - past) / past if past > 0 else 0
                out[35 + j] = np.clip(ret / 0.1, -1, 1)  # normalize ±10%

    # ── Candle structure (4 features, indices 41-44) ──
    last_candle = df_5m.iloc[-1]
    body = abs(last_candle["close"] - last_candle["open"])
    full_range = last_candle["high"] - last_candle["low"]
    out[41] = np.clip(body / (full_range + 1e-9), 0, 1)
    upper_wick = last_candle["high"] - max(last_candle["open"], last_candle["close"])
    lower_wick = min(last_candle["open"], last_candle["close"]) - last_candle["low"]
    out[42] = np.clip(upper_wick / (full_range + 1e-9), 0, 1)
    out[43] = np.clip(lower_wick / (full_range + 1e-9), 0, 1)
    # Consecutive green/red count
    greens = 0
    for k in range(1, min(11, len(df_5m))):
        row = df_5m.iloc[-k]
        if row["close"] >= row["open"]:
            greens += 1
        else:
            break
    reds = 0
    for k in range(1, min(11, len(df_5m))):
        row = df_5m.iloc[-k]
        if row["close"] < row["open"]:
            reds += 1
        else:
            break
    out[44] = np.clip((greens - reds) / 10.0, -1, 1)

    # ── Time (6 features, indices 45-50) ──
    ts = df_5m.index[-1]
    hour = ts.hour + ts.minute / 60.0
    out[45] = math.sin(2 * math.pi * hour / 24.0)
    out[46] = math.cos(2 * math.pi * hour / 24.0)
    dow = ts.dayofweek
    out[47] = math.sin(2 * math.pi * dow / 7.0)
    out[48] = math.cos(2 * math.pi * dow / 7.0)
    month = ts.month - 1
    out[49] = math.sin(2 * math.pi * month / 12.0)
    out[50] = math.cos(2 * math.pi * month / 12.0)

    # ── Coin markers (4 features, indices 51-54) ──
    coin = get_coin_profile(symbol)
    out[51] = coin["cap_tier"] / 3.0
    out[52] = coin["vol_class"] / 2.0
    out[53] = np.clip(coin["age"] / 16.0, 0, 1)
    out[54] = coin["is_btc"]

    # ── Funding/OB (5 features, indices 55-59) — zero-filled in backtest ──
    out[55] = np.clip(funding_rate / 0.01, -1, 1)
    out[56] = np.clip(funding_score, -1, 1)
    out[57] = np.clip(ob_imbalance, -1, 1)
    out[58] = np.clip(spread_pct / 0.5, 0, 1)
    out[59] = np.clip(bid_ask_wall_ratio / 5.0, 0, 1)

    # ── On-chain (2 features, indices 60-61) — zero-filled in backtest ──
    out[60] = np.clip(onchain_composite, -1, 1)
    out[61] = np.clip(exchange_reserve_trend, -1, 1)

    # ── Cross-asset (3 features, indices 62-64) ──
    if btc_df_5m is not None and len(btc_df_5m) >= 288:
        btc_c = btc_df_5m["close"]
        btc_curr = float(btc_c.iloc[-1])
        # BTC 1h return
        if len(btc_c) > 12:
            btc_1h = float(btc_c.iloc[-13])
            out[62] = np.clip((btc_curr - btc_1h) / btc_1h / 0.05, -1, 1)
        # BTC 4h return
        if len(btc_c) > 48:
            btc_4h = float(btc_c.iloc[-49])
            out[63] = np.clip((btc_curr - btc_4h) / btc_4h / 0.1, -1, 1)
        # BTC correlation (30-day rolling)
        if len(c5m) >= 8640 and len(btc_c) >= 8640:
            corr = c5m.iloc[-8640:].corr(btc_c.iloc[-8640:])
            out[64] = np.clip(corr if corr == corr else 0, -1, 1)

    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ml_features.py::TestExtractTabularFeatures -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add bot/learning/ml_features.py tests/test_ml_features.py
git commit -m "feat: add tabular feature extraction (65 features) for ML signal generator"
```

---

### Task 2: LSTM Sequence Builder

**Files:**
- Modify: `bot/learning/ml_features.py`
- Modify: `tests/test_ml_features.py`

This task adds the function to build 96×7 LSTM input sequences from 5m candle data.

- [ ] **Step 1: Write tests for LSTM sequence building**

```python
# Append to tests/test_ml_features.py

class TestBuildLSTMSequence:
    def test_output_shape(self):
        from bot.learning.ml_features import build_lstm_sequence
        df = _make_5m_candles(2000)
        seq = build_lstm_sequence(df)
        assert seq.shape == (96, 7), f"Expected (96, 7), got {seq.shape}"

    def test_output_dtype(self):
        from bot.learning.ml_features import build_lstm_sequence
        df = _make_5m_candles(2000)
        seq = build_lstm_sequence(df)
        assert seq.dtype == np.float32

    def test_values_finite(self):
        from bot.learning.ml_features import build_lstm_sequence
        df = _make_5m_candles(2000)
        seq = build_lstm_sequence(df)
        assert np.all(np.isfinite(seq)), "Sequence contains NaN or Inf"

    def test_insufficient_data_returns_zeros(self):
        from bot.learning.ml_features import build_lstm_sequence
        df = _make_5m_candles(50)  # Need at least 96+20 bars
        seq = build_lstm_sequence(df)
        assert seq.shape == (96, 7)
        assert np.all(seq == 0.0)

    def test_close_channel_is_normalized(self):
        from bot.learning.ml_features import build_lstm_sequence
        df = _make_5m_candles(2000)
        seq = build_lstm_sequence(df)
        # Channel 0 = normalized close (% change from first bar in window)
        # First bar should be ~0 (reference point)
        assert abs(seq[0, 0]) < 0.01

    def test_body_direction_channel(self):
        from bot.learning.ml_features import build_lstm_sequence
        df = _make_5m_candles(2000)
        seq = build_lstm_sequence(df)
        # Channel 3 = body direction, should be +1 or -1
        assert np.all(np.isin(seq[:, 3], [-1.0, 1.0]))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ml_features.py::TestBuildLSTMSequence -v`
Expected: FAIL with `ImportError: cannot import name 'build_lstm_sequence'`

- [ ] **Step 3: Implement LSTM sequence builder**

Add to `bot/learning/ml_features.py`:

```python
LSTM_WINDOW = 96  # 96 bars × 5m = 8 hours
LSTM_CHANNELS = 7
LSTM_MIN_BARS = LSTM_WINDOW + 20  # need extra for EMA warmup


def build_lstm_sequence(df_5m: pd.DataFrame) -> np.ndarray:
    """Build a (96, 7) LSTM input sequence from the last 96 five-minute candles.

    Channels:
        0: Normalized close (% change from first bar in window)
        1: Normalized volume (ratio to 20-bar rolling average)
        2: High-low range (normalized by close)
        3: Body direction (+1 green, -1 red)
        4: RSI (5m, scaled 0-1)
        5: Close vs EMA20 distance (%)
        6: Volume surge ratio

    Returns:
        np.ndarray of shape (96, 7) dtype float32. All zeros if insufficient data.
    """
    seq = np.zeros((LSTM_WINDOW, LSTM_CHANNELS), dtype=np.float32)

    if len(df_5m) < LSTM_MIN_BARS:
        return seq

    # Take the last LSTM_WINDOW bars
    window = df_5m.iloc[-LSTM_WINDOW:]
    close = window["close"].values.astype(float)
    high = window["high"].values.astype(float)
    low = window["low"].values.astype(float)
    open_ = window["open"].values.astype(float)
    volume = window["volume"].values.astype(float)

    # Channel 0: Normalized close (% change from first bar)
    ref_price = close[0] if close[0] > 0 else 1.0
    seq[:, 0] = (close - ref_price) / ref_price

    # Channel 1: Normalized volume (ratio to 20-bar average)
    # Use a wider window for the rolling average
    vol_extended = df_5m["volume"].iloc[-(LSTM_WINDOW + 20):].values.astype(float)
    vol_avg = np.convolve(vol_extended, np.ones(20) / 20, mode="valid")
    # vol_avg has length = len(vol_extended) - 19 = LSTM_WINDOW + 1
    # We need the last LSTM_WINDOW values
    vol_avg = vol_avg[-LSTM_WINDOW:]
    seq[:, 1] = np.clip(volume / (vol_avg + 1e-9), 0, 10) / 10.0

    # Channel 2: High-low range normalized by close
    seq[:, 2] = (high - low) / (close + 1e-9)

    # Channel 3: Body direction (+1 green, -1 red)
    seq[:, 3] = np.where(close >= open_, 1.0, -1.0)

    # Channel 4: RSI (5m) scaled 0-1
    rsi_series = rsi(df_5m["close"], 14)
    rsi_vals = rsi_series.iloc[-LSTM_WINDOW:].values.astype(float)
    seq[:, 4] = np.clip(np.nan_to_num(rsi_vals, nan=50.0) / 100.0, 0, 1)

    # Channel 5: Close vs EMA20 distance (%)
    ema20 = ema(df_5m["close"], 20)
    ema20_vals = ema20.iloc[-LSTM_WINDOW:].values.astype(float)
    seq[:, 5] = np.clip((close - ema20_vals) / (ema20_vals + 1e-9) * 100.0 / 5.0, -1, 1)

    # Channel 6: Volume surge ratio
    vsr = volume_surge_ratio(df_5m["volume"])
    vsr_vals = vsr.iloc[-LSTM_WINDOW:].values.astype(float)
    seq[:, 6] = np.clip(np.nan_to_num(vsr_vals, nan=1.0) / 5.0, 0, 1)

    # Final NaN cleanup
    np.nan_to_num(seq, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)

    return seq
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ml_features.py -v`
Expected: All 12 tests PASS (6 tabular + 6 LSTM)

- [ ] **Step 5: Commit**

```bash
git add bot/learning/ml_features.py tests/test_ml_features.py
git commit -m "feat: add LSTM sequence builder (96×7) for ML signal generator"
```

---

### Task 3: Batch Feature Extraction for Training

**Files:**
- Modify: `bot/learning/ml_features.py`
- Modify: `tests/test_ml_features.py`

This task adds a function to extract features for all SIGNAL_EVERY intervals in a candle dataset — needed by the training pipeline.

- [ ] **Step 1: Write tests for batch extraction**

```python
# Append to tests/test_ml_features.py

class TestExtractAllFeatures:
    def test_returns_aligned_arrays(self):
        from bot.learning.ml_features import extract_all_features
        df = _make_5m_candles(2000)
        tabular, sequences, timestamps = extract_all_features(df, symbol="BTC-EUR")
        assert tabular.ndim == 2
        assert sequences.ndim == 3
        assert tabular.shape[0] == sequences.shape[0] == len(timestamps)
        assert tabular.shape[1] == 65
        assert sequences.shape[1:] == (96, 7)

    def test_signal_every_spacing(self):
        from bot.learning.ml_features import extract_all_features, SIGNAL_EVERY
        df = _make_5m_candles(2000)
        _, _, timestamps = extract_all_features(df, symbol="BTC-EUR")
        if len(timestamps) >= 2:
            # Timestamps should be SIGNAL_EVERY * 5 minutes apart
            delta = (timestamps[1] - timestamps[0]).total_seconds()
            assert delta == SIGNAL_EVERY * 5 * 60

    def test_insufficient_data(self):
        from bot.learning.ml_features import extract_all_features
        df = _make_5m_candles(50)
        tabular, sequences, timestamps = extract_all_features(df, symbol="BTC-EUR")
        assert len(tabular) == 0
        assert len(sequences) == 0
        assert len(timestamps) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ml_features.py::TestExtractAllFeatures -v`
Expected: FAIL with `ImportError: cannot import name 'extract_all_features'`

- [ ] **Step 3: Implement batch extraction**

Add to `bot/learning/ml_features.py`:

```python
SIGNAL_EVERY = 6  # canonical: evaluate every 6 bars (30 minutes on 5m candles)


def extract_all_features(
    df_5m: pd.DataFrame,
    symbol: str = "BTC-EUR",
    btc_df_5m: Optional[pd.DataFrame] = None,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Extract features at every SIGNAL_EVERY interval for training.

    Returns:
        tabular: (N, 65) array of tabular features
        sequences: (N, 96, 7) array of LSTM sequences
        timestamps: list of N datetime timestamps
    """
    n = len(df_5m)
    if n < MIN_BARS_5M:
        return np.zeros((0, N_TABULAR), dtype=np.float32), \
               np.zeros((0, LSTM_WINDOW, LSTM_CHANNELS), dtype=np.float32), []

    tab_list = []
    seq_list = []
    ts_list = []

    # Start after enough warmup for all indicators
    start = max(MIN_BARS_5M, LSTM_MIN_BARS)
    for i in range(start, n, SIGNAL_EVERY):
        window = df_5m.iloc[:i + 1]
        btc_window = btc_df_5m.iloc[:i + 1] if btc_df_5m is not None else None

        tab = extract_tabular_features(window, symbol=symbol, btc_df_5m=btc_window)
        seq = build_lstm_sequence(window)

        tab_list.append(tab)
        seq_list.append(seq)
        ts_list.append(df_5m.index[i])

    if not tab_list:
        return np.zeros((0, N_TABULAR), dtype=np.float32), \
               np.zeros((0, LSTM_WINDOW, LSTM_CHANNELS), dtype=np.float32), []

    return np.stack(tab_list), np.stack(seq_list), ts_list
```

- [ ] **Step 4: Run all tests**

Run: `python -m pytest tests/test_ml_features.py -v`
Expected: All 15 tests PASS

- [ ] **Step 5: Commit**

```bash
git add bot/learning/ml_features.py tests/test_ml_features.py
git commit -m "feat: add batch feature extraction for ML training pipeline"
```

---

## Chunk 2: LSTM Embedder and ML Signal Generator

### Task 4: LSTM Embedder — Model Architecture and Inference

**Files:**
- Create: `bot/learning/lstm_embedder.py`
- Create: `tests/test_lstm_embedder.py`

The LSTM encodes 96×7 price sequences into 16-dim embeddings. Trained via self-supervised next-bar prediction (predict next 12 bars' OHLCV). After training, the prediction head is discarded and only the embedding layer is used.

- [ ] **Step 1: Write tests for LSTM model**

```python
# tests/test_lstm_embedder.py
"""Tests for LSTM embedder model."""
import numpy as np
import pytest
import torch


class TestLSTMEmbedder:
    def test_embedding_shape(self):
        from bot.learning.lstm_embedder import LSTMEmbedder
        model = LSTMEmbedder()
        x = torch.randn(4, 96, 7)  # batch=4, seq=96, channels=7
        emb = model.embed(x)
        assert emb.shape == (4, 16), f"Expected (4, 16), got {emb.shape}"

    def test_prediction_shape(self):
        from bot.learning.lstm_embedder import LSTMEmbedder
        model = LSTMEmbedder()
        x = torch.randn(4, 96, 7)
        pred = model.predict_next(x)
        assert pred.shape == (4, 60), f"Expected (4, 60), got {pred.shape}"

    def test_single_sample(self):
        from bot.learning.lstm_embedder import LSTMEmbedder
        model = LSTMEmbedder()
        x = torch.randn(1, 96, 7)
        emb = model.embed(x)
        assert emb.shape == (1, 16)

    def test_embed_deterministic(self):
        from bot.learning.lstm_embedder import LSTMEmbedder
        model = LSTMEmbedder()
        model.eval()
        x = torch.randn(2, 96, 7)
        with torch.no_grad():
            e1 = model.embed(x)
            e2 = model.embed(x)
        assert torch.allclose(e1, e2)

    def test_save_load_roundtrip(self, tmp_path):
        from bot.learning.lstm_embedder import LSTMEmbedder
        model = LSTMEmbedder()
        model.eval()
        x = torch.randn(2, 96, 7)
        with torch.no_grad():
            emb_before = model.embed(x)

        path = str(tmp_path / "lstm.pt")
        model.save(path)

        model2 = LSTMEmbedder()
        model2.load(path)
        model2.eval()
        with torch.no_grad():
            emb_after = model2.embed(x)

        assert torch.allclose(emb_before, emb_after, atol=1e-6)


class TestLSTMEmbedderNumpy:
    def test_embed_numpy(self):
        from bot.learning.lstm_embedder import LSTMEmbedder
        model = LSTMEmbedder()
        model.eval()
        arr = np.random.randn(96, 7).astype(np.float32)
        emb = model.embed_numpy(arr)
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (16,)
        assert emb.dtype == np.float32
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_lstm_embedder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bot.learning.lstm_embedder'`

- [ ] **Step 3: Implement LSTM embedder**

```python
# bot/learning/lstm_embedder.py
"""LSTM embedder: encode price sequences into fixed-dim embeddings.

Architecture:
    Input: (batch, 96, 7)
    → LSTM(7 → 32, 2 layers, dropout=0.2)
    → last hidden → Linear(32 → 16) + ReLU
    → embedding (batch, 16)

Self-supervised training objective:
    Predict next 12 bars' normalized OHLCV (5 channels × 12 = 60 outputs).
    Prediction head: Linear(32 → 60), discarded after training.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

SEQ_LEN = 96
INPUT_DIM = 7
HIDDEN_DIM = 32
NUM_LAYERS = 2
DROPOUT = 0.2
EMBED_DIM = 16
PREDICT_BARS = 12
PREDICT_CHANNELS = 5  # OHLCV
PREDICT_DIM = PREDICT_BARS * PREDICT_CHANNELS  # 60


class LSTMEmbedder(nn.Module):
    """LSTM sequence encoder with self-supervised prediction head."""

    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            batch_first=True,
        )
        # Embedding projection
        self.embed_proj = nn.Sequential(
            nn.Linear(HIDDEN_DIM, EMBED_DIM),
            nn.ReLU(),
        )
        # Self-supervised prediction head (discarded after training)
        self.predict_head = nn.Linear(HIDDEN_DIM, PREDICT_DIM)

    def _lstm_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Run LSTM, return last hidden state. Shape: (batch, HIDDEN_DIM)."""
        # x: (batch, seq_len, input_dim)
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers, batch, hidden_dim) — take last layer
        return h_n[-1]

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Get 16-dim embedding. Input: (batch, 96, 7) → Output: (batch, 16)."""
        h = self._lstm_hidden(x)
        return self.embed_proj(h)

    def predict_next(self, x: torch.Tensor) -> torch.Tensor:
        """Predict next 12 bars OHLCV. Input: (batch, 96, 7) → Output: (batch, 60)."""
        h = self._lstm_hidden(x)
        return self.predict_head(h)

    def embed_numpy(self, seq: np.ndarray) -> np.ndarray:
        """Convenience: single numpy sequence (96, 7) → embedding (16,)."""
        self.eval()
        x = torch.from_numpy(seq).unsqueeze(0).float()
        with torch.no_grad():
            emb = self.embed(x)
        return emb.squeeze(0).numpy()

    def save(self, path: str) -> None:
        """Save model weights."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        logger.info("LSTM embedder saved to %s", path)

    def load(self, path: str) -> None:
        """Load model weights."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state)
        logger.info("LSTM embedder loaded from %s", path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_lstm_embedder.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add bot/learning/lstm_embedder.py tests/test_lstm_embedder.py
git commit -m "feat: add LSTM embedder model with self-supervised prediction head"
```

---

### Task 5: LSTM Training Function

**Files:**
- Modify: `bot/learning/lstm_embedder.py`
- Modify: `tests/test_lstm_embedder.py`

Add the training loop for self-supervised next-bar prediction.

- [ ] **Step 1: Write tests for LSTM training**

```python
# Append to tests/test_lstm_embedder.py

class TestLSTMTraining:
    def _make_training_data(self, n_samples=200):
        """Create synthetic (sequence, target) pairs."""
        np.random.seed(42)
        sequences = np.random.randn(n_samples, 96, 7).astype(np.float32)
        # Targets: next 12 bars OHLCV (60 values)
        targets = np.random.randn(n_samples, 60).astype(np.float32) * 0.01
        return sequences, targets

    def test_train_reduces_loss(self):
        from bot.learning.lstm_embedder import LSTMEmbedder, train_lstm
        model = LSTMEmbedder()
        seqs, targets = self._make_training_data(200)
        history = train_lstm(model, seqs, targets, epochs=5, batch_size=32, lr=1e-3)
        assert history[-1] < history[0], "Loss should decrease over training"

    def test_train_returns_loss_history(self):
        from bot.learning.lstm_embedder import LSTMEmbedder, train_lstm
        model = LSTMEmbedder()
        seqs, targets = self._make_training_data(100)
        history = train_lstm(model, seqs, targets, epochs=3, batch_size=32, lr=1e-3)
        assert len(history) == 3
        assert all(isinstance(v, float) for v in history)

    def test_train_with_validation(self):
        from bot.learning.lstm_embedder import LSTMEmbedder, train_lstm
        model = LSTMEmbedder()
        seqs, targets = self._make_training_data(200)
        history = train_lstm(
            model, seqs[:160], targets[:160],
            val_seqs=seqs[160:], val_targets=targets[160:],
            epochs=3, batch_size=32, lr=1e-3,
        )
        assert len(history) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_lstm_embedder.py::TestLSTMTraining -v`
Expected: FAIL with `ImportError: cannot import name 'train_lstm'`

- [ ] **Step 3: Implement training function**

Add to `bot/learning/lstm_embedder.py`:

```python
def train_lstm(
    model: LSTMEmbedder,
    sequences: np.ndarray,
    targets: np.ndarray,
    val_seqs: np.ndarray | None = None,
    val_targets: np.ndarray | None = None,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> list[float]:
    """Train the LSTM via self-supervised next-bar prediction.

    Args:
        sequences: (N, 96, 7) input sequences
        targets: (N, 60) normalized OHLCV of next 12 bars
        val_seqs/val_targets: optional validation set
        epochs: number of training epochs
        batch_size: mini-batch size
        lr: learning rate

    Returns:
        List of per-epoch average training loss values.
    """
    device = torch.device("cpu")
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X = torch.from_numpy(sequences).float().to(device)
    Y = torch.from_numpy(targets).float().to(device)
    n = len(X)

    loss_history = []

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        batches = 0

        for start in range(0, n - batch_size + 1, batch_size):
            idx = perm[start:start + batch_size]
            x_batch = X[idx]
            y_batch = Y[idx]

            optimizer.zero_grad()
            pred = model.predict_next(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / max(batches, 1)
        loss_history.append(avg_loss)

        # Validation
        if val_seqs is not None and val_targets is not None:
            model.eval()
            with torch.no_grad():
                vX = torch.from_numpy(val_seqs).float().to(device)
                vY = torch.from_numpy(val_targets).float().to(device)
                val_pred = model.predict_next(vX)
                val_loss = criterion(val_pred, vY).item()
            logger.info(
                "LSTM epoch %d/%d — train_loss=%.6f, val_loss=%.6f",
                epoch + 1, epochs, avg_loss, val_loss,
            )
        else:
            logger.info("LSTM epoch %d/%d — train_loss=%.6f", epoch + 1, epochs, avg_loss)

    return loss_history
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_lstm_embedder.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add bot/learning/lstm_embedder.py tests/test_lstm_embedder.py
git commit -m "feat: add LSTM self-supervised training function"
```

---

### Task 6: ML Signal Generator — Prediction Interface

**Files:**
- Create: `bot/learning/ml_signal_generator.py`
- Create: `tests/test_ml_signal_generator.py`
- Reference: `bot/learning/lstm_embedder.py`
- Reference: `bot/learning/ml_features.py`

The `MLSignalGenerator` loads trained LSTM + 12 XGBoost models and exposes a `predict()` method that returns 12 probabilities + a recommended direction.

- [ ] **Step 1: Write tests for ML Signal Generator**

```python
# tests/test_ml_signal_generator.py
"""Tests for ML signal generator prediction interface."""
import json
import numpy as np
import pandas as pd
import pytest


def _make_5m_candles(bars: int = 2000) -> pd.DataFrame:
    np.random.seed(42)
    timestamps = pd.date_range("2024-01-01", periods=bars, freq="5min")
    base = 50000.0
    returns = np.random.randn(bars) * 0.001
    close = base * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.randn(bars)) * 0.002)
    low = close * (1 - np.abs(np.random.randn(bars)) * 0.002)
    open_ = close * (1 + np.random.randn(bars) * 0.001)
    volume = np.random.exponential(100, bars)
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume,
    }, index=timestamps)


class TestMLSignalGenerator:
    def test_predict_returns_correct_structure(self, tmp_path):
        from bot.learning.ml_signal_generator import MLSignalGenerator
        gen = self._make_generator(tmp_path)
        df = _make_5m_candles(2000)
        result = gen.predict(df, symbol="BTC-EUR")
        assert "probabilities" in result
        assert "direction" in result
        assert "net_up" in result
        assert "net_down" in result
        assert len(result["probabilities"]) == 12

    def test_predict_probabilities_bounded(self, tmp_path):
        from bot.learning.ml_signal_generator import MLSignalGenerator
        gen = self._make_generator(tmp_path)
        df = _make_5m_candles(2000)
        result = gen.predict(df, symbol="BTC-EUR")
        for p in result["probabilities"]:
            assert 0.0 <= p <= 1.0, f"Probability {p} out of bounds"

    def test_predict_direction_valid(self, tmp_path):
        from bot.learning.ml_signal_generator import MLSignalGenerator
        gen = self._make_generator(tmp_path)
        df = _make_5m_candles(2000)
        result = gen.predict(df, symbol="BTC-EUR")
        assert result["direction"] in ("LONG", "SHORT", None)

    def test_predict_insufficient_data(self, tmp_path):
        from bot.learning.ml_signal_generator import MLSignalGenerator
        gen = self._make_generator(tmp_path)
        df = _make_5m_candles(50)
        result = gen.predict(df, symbol="BTC-EUR")
        assert result["direction"] is None
        assert all(p == 0.5 for p in result["probabilities"])

    def test_models_not_found_raises(self):
        from bot.learning.ml_signal_generator import MLSignalGenerator
        with pytest.raises(FileNotFoundError):
            MLSignalGenerator(model_dir="/nonexistent/path")

    def _make_generator(self, tmp_path):
        """Create a generator with dummy models for testing."""
        from bot.learning.ml_signal_generator import MLSignalGenerator
        from bot.learning.lstm_embedder import LSTMEmbedder
        import xgboost as xgb

        model_dir = tmp_path / "ml_signals"
        model_dir.mkdir()

        # Save dummy LSTM
        lstm = LSTMEmbedder()
        lstm.save(str(model_dir / "lstm.pt"))

        # Save 12 dummy XGBoost models
        horizons = ["30m", "1h", "4h", "12h", "24h", "72h"]
        directions = ["up", "down"]
        for h in horizons:
            for d in directions:
                # Create a simple XGBoost model trained on random data
                X = np.random.randn(100, 81).astype(np.float32)
                y = np.random.randint(0, 2, 100)
                model = xgb.XGBClassifier(
                    n_estimators=5, max_depth=2, use_label_encoder=False,
                    eval_metric="logloss",
                )
                model.fit(X, y)
                model.save_model(str(model_dir / f"xgb_{h}_{d}.json"))

        # Save feature config
        config = {"n_tabular": 65, "n_lstm_embed": 16, "horizons": horizons}
        (model_dir / "feature_config.json").write_text(json.dumps(config))

        return MLSignalGenerator(model_dir=str(model_dir))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ml_signal_generator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bot.learning.ml_signal_generator'`

- [ ] **Step 3: Implement MLSignalGenerator**

```python
# bot/learning/ml_signal_generator.py
"""ML Signal Generator: load trained models and predict directional probabilities.

Loads:
    - LSTM embedder (lstm.pt)
    - 12 XGBoost classifiers (xgb_{horizon}_{direction}.json)
    - Feature config (feature_config.json)

Exposes:
    predict(df_5m, symbol) → dict with probabilities, direction, net_up, net_down
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import xgboost as xgb

from bot.learning.lstm_embedder import LSTMEmbedder
from bot.learning.ml_features import extract_tabular_features, build_lstm_sequence

logger = logging.getLogger(__name__)

HORIZONS = ["30m", "1h", "4h", "12h", "24h", "72h"]
DIRECTIONS = ["up", "down"]
MIN_SIGNAL_PROB = 0.4  # minimum average probability to generate a direction signal


class MLSignalGenerator:
    """Load trained LSTM + XGBoost models and generate trading signals."""

    def __init__(self, model_dir: str = "models/ml_signals") -> None:
        self._model_dir = Path(model_dir)
        if not self._model_dir.exists():
            raise FileNotFoundError(
                f"ML model directory not found: {model_dir}. "
                f"Run scripts/train_ml_signals.py first."
            )

        # Load LSTM
        lstm_path = self._model_dir / "lstm.pt"
        if not lstm_path.exists():
            raise FileNotFoundError(f"LSTM model not found: {lstm_path}")
        self._lstm = LSTMEmbedder()
        self._lstm.load(str(lstm_path))
        self._lstm.eval()

        # Load XGBoost models
        self._xgb_models: Dict[str, xgb.XGBClassifier] = {}
        for h in HORIZONS:
            for d in DIRECTIONS:
                key = f"{h}_{d}"
                path = self._model_dir / f"xgb_{key}.json"
                if not path.exists():
                    raise FileNotFoundError(f"XGBoost model not found: {path}")
                model = xgb.XGBClassifier()
                model.load_model(str(path))
                self._xgb_models[key] = model

        # Load feature config
        config_path = self._model_dir / "feature_config.json"
        if config_path.exists():
            with open(config_path) as f:
                self._config = json.load(f)
        else:
            self._config = {"n_tabular": 65, "n_lstm_embed": 16, "horizons": HORIZONS}

        logger.info(
            "MLSignalGenerator loaded: LSTM + %d XGBoost models from %s",
            len(self._xgb_models), model_dir,
        )

    def predict(
        self,
        df_5m: "pd.DataFrame",
        symbol: str = "BTC-EUR",
        btc_df_5m: Optional["pd.DataFrame"] = None,
        **live_features,
    ) -> Dict[str, Any]:
        """Generate predictions for the current market state.

        Returns:
            dict with keys:
                probabilities: list of 12 floats [up_30m, down_30m, up_1h, down_1h, ...]
                direction: "LONG", "SHORT", or None (no signal)
                net_up: float (mean of up probabilities across horizons)
                net_down: float (mean of down probabilities across horizons)
        """
        default = {
            "probabilities": [0.5] * 12,
            "direction": None,
            "net_up": 0.5,
            "net_down": 0.5,
        }

        # Extract features
        tabular = extract_tabular_features(
            df_5m, symbol=symbol, btc_df_5m=btc_df_5m, **live_features,
        )
        seq = build_lstm_sequence(df_5m)

        # Check for insufficient data (all zeros)
        if np.all(tabular == 0) or np.all(seq == 0):
            return default

        # Get LSTM embedding
        embedding = self._lstm.embed_numpy(seq)  # (16,)

        # Concatenate: 65 tabular + 16 embedding = 81 features
        combined = np.concatenate([tabular, embedding]).reshape(1, -1)

        # Run all 12 XGBoost models
        probabilities = []
        for h in HORIZONS:
            for d in DIRECTIONS:
                key = f"{h}_{d}"
                model = self._xgb_models[key]
                prob = float(model.predict_proba(combined)[0, 1])
                probabilities.append(prob)

        # Compute net direction
        # probabilities layout: [up_30m, down_30m, up_1h, down_1h, ...]
        up_probs = [probabilities[i] for i in range(0, 12, 2)]  # indices 0,2,4,6,8,10
        down_probs = [probabilities[i] for i in range(1, 12, 2)]  # indices 1,3,5,7,9,11
        net_up = float(np.mean(up_probs))
        net_down = float(np.mean(down_probs))

        # Direction decision
        direction = None
        if max(net_up, net_down) >= MIN_SIGNAL_PROB:
            direction = "LONG" if net_up > net_down else "SHORT"

        return {
            "probabilities": probabilities,
            "direction": direction,
            "net_up": net_up,
            "net_down": net_down,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ml_signal_generator.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add bot/learning/ml_signal_generator.py tests/test_ml_signal_generator.py
git commit -m "feat: add ML signal generator prediction interface"
```

---

### Task 7: Training Pipeline Script

**Files:**
- Create: `scripts/train_ml_signals.py`
- Create: `tests/test_train_ml_signals.py`
- Reference: `bot/learning/ml_features.py`, `bot/learning/lstm_embedder.py`

End-to-end training: fetch candles → extract features → compute labels → train LSTM → generate embeddings → train 12 XGBoost models → save.

- [ ] **Step 1: Write tests for label generation and training helpers**

```python
# tests/test_train_ml_signals.py
"""Tests for ML signal training pipeline."""
import numpy as np
import pandas as pd
import pytest


def _make_5m_candles(bars: int = 3000) -> pd.DataFrame:
    np.random.seed(42)
    timestamps = pd.date_range("2024-01-01", periods=bars, freq="5min")
    base = 50000.0
    returns = np.random.randn(bars) * 0.001
    close = base * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.randn(bars)) * 0.002)
    low = close * (1 - np.abs(np.random.randn(bars)) * 0.002)
    open_ = close * (1 + np.random.randn(bars) * 0.001)
    volume = np.random.exponential(100, bars)
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume,
    }, index=timestamps)


class TestComputeLabels:
    def test_label_shape(self):
        # Import with sys.path manipulation (scripts/ is not a package)
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from scripts.train_ml_signals import compute_labels
        df = _make_5m_candles(3000)
        labels = compute_labels(df)
        # Should have 12 columns: up/down for each of 6 horizons
        assert labels.shape[1] == 12
        assert labels.shape[0] == len(df)

    def test_labels_binary(self):
        from scripts.train_ml_signals import compute_labels
        df = _make_5m_candles(3000)
        labels = compute_labels(df)
        assert set(np.unique(labels[~np.isnan(labels)])).issubset({0.0, 1.0})

    def test_future_bars_have_nan(self):
        from scripts.train_ml_signals import compute_labels
        df = _make_5m_candles(3000)
        labels = compute_labels(df)
        # Last 864 bars (72h horizon) cannot have labels
        assert np.all(np.isnan(labels[-864:, -2:]))


class TestBuildTrainingTargets:
    def test_lstm_targets_shape(self):
        from scripts.train_ml_signals import build_lstm_targets
        df = _make_5m_candles(3000)
        targets = build_lstm_targets(df)
        # 12 bars × 5 channels = 60 per sample
        assert targets.shape[1] == 60

    def test_targets_aligned_with_candles(self):
        from scripts.train_ml_signals import build_lstm_targets
        df = _make_5m_candles(3000)
        targets = build_lstm_targets(df)
        # Should have one target per bar (except last 12)
        assert targets.shape[0] == len(df) - 12
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_train_ml_signals.py -v`
Expected: FAIL with `ModuleNotFoundError` (the `sys.path` insert in the test handles the import path; the error is because the module doesn't exist yet)

- [ ] **Step 3: Implement training pipeline**

```python
# scripts/train_ml_signals.py
"""Train ML signal generator models: LSTM + 12 XGBoost classifiers.

Usage:
    python scripts/train_ml_signals.py

Pipeline:
    1. Fetch 5 years of 5m candles per pair
    2. Compute labels (horizon-scaled thresholds)
    3. Train LSTM (self-supervised next-bar prediction)
    4. Generate embeddings
    5. Train 12 XGBoost models (up/down × 6 horizons)
    6. Save all models to models/ml_signals/
"""
import sys, os, json, time
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from bot.exchange.bitvavo_client import BitvavoClient
from bot.learning.lstm_embedder import LSTMEmbedder, train_lstm
from bot.learning.ml_features import (
    extract_all_features, N_TABULAR, LSTM_WINDOW, LSTM_CHANNELS,
)

PAIRS = ["BTC-EUR", "ETH-EUR", "XRP-EUR"]
YEARS = 5
MODEL_DIR = "models/ml_signals"

# Horizon definitions: (name, bars_ahead, threshold_pct)
HORIZONS = [
    ("30m",  6,    0.15),
    ("1h",   12,   0.3),
    ("4h",   48,   0.8),
    ("12h",  144,  1.5),
    ("24h",  288,  2.5),
    ("72h",  864,  4.0),
]

# LSTM training params
LSTM_EPOCHS = 50
LSTM_BATCH = 256
LSTM_LR = 1e-3
LSTM_PREDICT_BARS = 12

# XGBoost params
XGB_PARAMS = {
    "objective": "binary:logistic",
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "eval_metric": "logloss",
    "use_label_encoder": False,
}


def compute_labels(df_5m: pd.DataFrame) -> np.ndarray:
    """Compute binary labels for all horizons (vectorized).

    Returns:
        (N, 12) array: columns are [up_30m, down_30m, up_1h, down_1h, ...]
        NaN where future data is unavailable.
    """
    close = df_5m["close"].values.astype(np.float64)
    n = len(close)
    labels = np.full((n, 12), np.nan, dtype=np.float32)

    for h_idx, (name, bars, threshold) in enumerate(HORIZONS):
        threshold_frac = threshold / 100.0
        # Vectorized: compute future returns for all valid bars at once
        valid = n - bars
        if valid <= 0:
            continue
        future_close = close[bars:bars + valid]
        current_close = close[:valid]
        future_return = (future_close - current_close) / (current_close + 1e-12)
        labels[:valid, h_idx * 2] = (future_return > threshold_frac).astype(np.float32)
        labels[:valid, h_idx * 2 + 1] = (future_return < -threshold_frac).astype(np.float32)

    return labels


def build_lstm_targets(df_5m: pd.DataFrame) -> np.ndarray:
    """Build LSTM prediction targets: next 12 bars normalized OHLCV (vectorized).

    Returns:
        (N-12, 60) array where each row is 12 bars × 5 channels (OHLCV).
    """
    ohlcv = df_5m[["open", "high", "low", "close", "volume"]].values.astype(np.float64)
    n = len(ohlcv)
    n_targets = n - LSTM_PREDICT_BARS
    targets = np.zeros((n_targets, LSTM_PREDICT_BARS * 5), dtype=np.float32)

    ref_close = ohlcv[:n_targets, 3]  # reference close for each sample
    ref_vol = np.where(ohlcv[:n_targets, 4] > 0, ohlcv[:n_targets, 4], 1.0)

    for step in range(LSTM_PREDICT_BARS):
        future = ohlcv[step + 1:step + 1 + n_targets]
        # OHLC as % change from reference close
        for ch in range(4):
            targets[:, step * 5 + ch] = (future[:, ch] - ref_close) / (ref_close + 1e-12)
        # Volume ratio
        targets[:, step * 5 + 4] = future[:, 4] / (ref_vol + 1e-12)

    return targets


def main():
    print("=" * 80)
    print("  ML SIGNAL GENERATOR TRAINING PIPELINE")
    print("=" * 80)

    model_dir = Path(MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Fetch candles ──
    all_dfs = {}
    client = BitvavoClient(api_key="", api_secret="", paper_trading=True)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=YEARS * 365)

    for pair in PAIRS:
        print(f"\n  Fetching {pair}...", end="", flush=True)
        candles = client.get_candles_range(pair, interval="5m", start=start, end=end)
        if not candles or len(candles) < 10000:
            print(f" skipped (insufficient data: {len(candles) if candles else 0})")
            continue
        df = pd.DataFrame([{
            "open": c.open, "high": c.high, "low": c.low,
            "close": c.close, "volume": c.volume,
        } for c in candles], index=[c.timestamp for c in candles])
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        print(f" {len(df):,} candles")
        all_dfs[pair] = df
        time.sleep(1)

    if not all_dfs:
        print("  ERROR: No data fetched. Exiting.")
        sys.exit(1)

    # ── Step 2: Build LSTM targets and train LSTM ──
    print("\n  Building LSTM training data...")
    all_seqs = []
    all_lstm_targets = []
    for pair, df in all_dfs.items():
        # Build sequences for every bar that has enough history
        from bot.learning.ml_features import build_lstm_sequence, LSTM_MIN_BARS
        targets = build_lstm_targets(df)
        for i in range(LSTM_MIN_BARS, len(df) - LSTM_PREDICT_BARS):
            seq = build_lstm_sequence(df.iloc[:i + 1])
            if not np.all(seq == 0):
                all_seqs.append(seq)
                all_lstm_targets.append(targets[i])
        print(f"    {pair}: {len(all_seqs)} sequences so far")

    seqs_arr = np.stack(all_seqs)
    targets_arr = np.stack(all_lstm_targets)
    print(f"  Total LSTM samples: {len(seqs_arr):,}")

    # Split: last 6 months for validation
    val_cutoff = int(len(seqs_arr) * 0.88)  # ~last 6 months of 5 years
    train_seqs, val_seqs = seqs_arr[:val_cutoff], seqs_arr[val_cutoff:]
    train_targets, val_targets = targets_arr[:val_cutoff], targets_arr[val_cutoff:]

    print(f"  LSTM train: {len(train_seqs):,}, val: {len(val_seqs):,}")
    lstm = LSTMEmbedder()
    print("  Training LSTM...")
    t0 = time.time()
    train_lstm(
        lstm, train_seqs, train_targets,
        val_seqs=val_seqs, val_targets=val_targets,
        epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH, lr=LSTM_LR,
    )
    print(f"  LSTM trained in {time.time() - t0:.0f}s")
    lstm.save(str(model_dir / "lstm.pt"))

    # ── Step 3: Extract features + embeddings for XGBoost ──
    print("\n  Extracting tabular features + LSTM embeddings...")
    lstm.eval()
    all_tabular = []
    all_labels = []
    all_embeddings = []

    btc_df = all_dfs.get("BTC-EUR")

    for pair, df in all_dfs.items():
        print(f"    {pair}: extracting features...", end="", flush=True)
        tabular, sequences, timestamps = extract_all_features(
            df, symbol=pair, btc_df_5m=btc_df if pair != "BTC-EUR" else None,
        )
        if len(tabular) == 0:
            print(" skipped (no features)")
            continue

        # Compute labels at the same timestamps
        labels = compute_labels(df)
        # Map timestamps to bar indices
        ts_indices = [df.index.get_loc(ts) for ts in timestamps]
        label_rows = labels[ts_indices]

        # Generate embeddings
        embeddings = []
        for seq in sequences:
            emb = lstm.embed_numpy(seq)
            embeddings.append(emb)
        embeddings = np.stack(embeddings)

        all_tabular.append(tabular)
        all_labels.append(label_rows)
        all_embeddings.append(embeddings)
        print(f" {len(tabular):,} samples")

    X_tab = np.vstack(all_tabular)
    y_all = np.vstack(all_labels)
    X_emb = np.vstack(all_embeddings)
    X = np.hstack([X_tab, X_emb])  # (N, 81)
    print(f"  Total XGBoost samples: {len(X):,}, features: {X.shape[1]}")

    # ── Step 4: Walk-forward split and train XGBoost ──
    # Date-based split per pair: years 1-3 train, year 4 validate, year 5 test
    # Since data is concatenated per-pair in order, use timestamps for splitting
    all_timestamps_flat = []
    for pair, df in all_dfs.items():
        _, _, timestamps = extract_all_features(
            df, symbol=pair, btc_df_5m=btc_df if pair != "BTC-EUR" else None,
        )
        all_timestamps_flat.extend(timestamps)
    all_ts = pd.DatetimeIndex(all_timestamps_flat)
    total_span = (all_ts.max() - all_ts.min()).days
    train_cutoff = all_ts.min() + timedelta(days=int(total_span * 0.6))
    val_cutoff = all_ts.min() + timedelta(days=int(total_span * 0.8))

    train_mask = all_ts < train_cutoff
    val_mask = (all_ts >= train_cutoff) & (all_ts < val_cutoff)
    test_mask = all_ts >= val_cutoff

    print(f"  Walk-forward split: train={train_mask.sum():,}, "
          f"val={val_mask.sum():,}, test={test_mask.sum():,}")

    importances = {}
    for h_idx, (h_name, bars, threshold) in enumerate(HORIZONS):
        for d_idx, d_name in enumerate(["up", "down"]):
            col = h_idx * 2 + d_idx
            y = y_all[:, col]

            # Filter out NaN labels within each split
            train_valid = train_mask & ~np.isnan(y)
            val_valid = val_mask & ~np.isnan(y)
            test_valid = test_mask & ~np.isnan(y)

            X_train, y_train = X[train_valid], y[train_valid]
            X_val, y_val = X[val_valid], y[val_valid]
            X_test, y_test = X[test_valid], y[test_valid]

            model = xgb.XGBClassifier(**XGB_PARAMS)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False,
            )

            # Test accuracy
            test_pred = model.predict(X_test)
            acc = float(np.mean(test_pred == y_test))
            pos_rate = float(y_test.mean())

            # Feature importance check
            fi = model.feature_importances_
            max_fi = float(fi.max())
            top_feat = int(fi.argmax())
            importances[f"{h_name}_{d_name}"] = fi.tolist()

            # Validation: accuracy > 52%, no feature > 30%
            status = "OK"
            if acc < 0.52:
                status = "WARN: acc < 52%"
            if max_fi > 0.30:
                status = f"WARN: feature {top_feat} dominates ({max_fi:.1%})"

            print(f"    xgb_{h_name}_{d_name}: test_acc={acc:.3f}, "
                  f"pos_rate={pos_rate:.3f}, top_feat={top_feat}({max_fi:.1%}) [{status}]")

            model.save_model(str(model_dir / f"xgb_{h_name}_{d_name}.json"))

    # Save feature importances
    (model_dir / "feature_importances.json").write_text(json.dumps(importances, indent=2))

    # ── Step 5: Save config ──
    config = {
        "n_tabular": N_TABULAR,
        "n_lstm_embed": 16,
        "horizons": [h[0] for h in HORIZONS],
        "thresholds": {h[0]: h[2] for h in HORIZONS},
        "pairs_trained": list(all_dfs.keys()),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    (model_dir / "feature_config.json").write_text(json.dumps(config, indent=2))

    print(f"\n  All models saved to {MODEL_DIR}/")
    print("  Done!")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_train_ml_signals.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/train_ml_signals.py tests/test_train_ml_signals.py
git commit -m "feat: add ML signal training pipeline (LSTM + 12 XGBoost models)"
```

---

## Chunk 3: RL Signal Evaluator Update and Engine Integration

### Task 8: Update RL Signal Evaluator Observation Space

**Files:**
- Modify: `bot/learning/rl_signal_evaluator.py:24-163`
- Create: `tests/test_rl_signal_evaluator_ml.py`

Update OBS_DIM from 32 to 30, add new `build_ml_signal_obs()` function that accepts ML predictions + portfolio state, remove `STRATEGY_IDS` and the old `build_signal_obs()`.

**Important:** Keep `COIN_PROFILES`, `get_coin_profile()`, `EvalResult`, and the `RLSignalEvaluator` class unchanged (they're used by multiple consumers). Only the observation building and dimension change.

- [ ] **Step 1: Write tests for new observation builder**

```python
# tests/test_rl_signal_evaluator_ml.py
"""Tests for ML-updated RL signal evaluator observation space."""
import numpy as np
import pytest


class TestBuildMLSignalObs:
    def test_output_shape_is_30(self):
        from bot.learning.rl_signal_evaluator import build_ml_signal_obs
        obs = build_ml_signal_obs(
            probabilities=[0.6, 0.3, 0.55, 0.35, 0.7, 0.2, 0.65, 0.25, 0.5, 0.4, 0.45, 0.55],
        )
        assert obs.shape == (30,), f"Expected (30,), got {obs.shape}"

    def test_output_dtype(self):
        from bot.learning.rl_signal_evaluator import build_ml_signal_obs
        obs = build_ml_signal_obs(
            probabilities=[0.5] * 12,
        )
        assert obs.dtype == np.float32

    def test_probabilities_in_obs(self):
        from bot.learning.rl_signal_evaluator import build_ml_signal_obs
        probs = [0.8, 0.1, 0.7, 0.2, 0.6, 0.3, 0.5, 0.4, 0.9, 0.05, 0.75, 0.15]
        obs = build_ml_signal_obs(probabilities=probs)
        # First 12 elements should be the probabilities (clipped 0-1)
        for i in range(12):
            assert abs(obs[i] - probs[i]) < 0.01

    def test_cross_horizon_features(self):
        from bot.learning.rl_signal_evaluator import build_ml_signal_obs
        probs = [0.9, 0.1, 0.7, 0.2, 0.6, 0.3, 0.5, 0.4, 0.8, 0.05, 0.75, 0.15]
        obs = build_ml_signal_obs(probabilities=probs)
        # Index 12: max_prob, 13: min_prob, 14: horizon_agreement, 15: trend_alignment
        assert obs[12] == pytest.approx(0.9, abs=0.01)   # max prob
        assert obs[13] == pytest.approx(0.05, abs=0.01)  # min prob

    def test_coin_markers_in_obs(self):
        from bot.learning.rl_signal_evaluator import build_ml_signal_obs
        obs_btc = build_ml_signal_obs(probabilities=[0.5] * 12, symbol="BTC-EUR")
        obs_eth = build_ml_signal_obs(probabilities=[0.5] * 12, symbol="ETH-EUR")
        # Last 4 elements are coin markers — BTC and ETH differ
        assert not np.array_equal(obs_btc[-4:], obs_eth[-4:])

    def test_obs_dim_constant_is_30(self):
        from bot.learning.rl_signal_evaluator import OBS_DIM
        assert OBS_DIM == 30


class TestEvaluatorWithNewObs:
    def test_evaluate_with_30_dim_obs(self):
        from bot.learning.rl_signal_evaluator import RLSignalEvaluator, build_ml_signal_obs
        evaluator = RLSignalEvaluator()
        obs = build_ml_signal_obs(probabilities=[0.6, 0.3] * 6)
        result = evaluator.evaluate(obs)
        assert 0 <= result.confidence <= 1
        assert 0.5 <= result.atr_multiplier <= 4.0
        assert 1.0 <= result.rr_ratio <= 5.0

    def test_save_load_with_new_dims(self, tmp_path):
        from bot.learning.rl_signal_evaluator import RLSignalEvaluator, build_ml_signal_obs
        evaluator = RLSignalEvaluator()
        obs = build_ml_signal_obs(probabilities=[0.7, 0.2] * 6)
        result1 = evaluator.evaluate(obs, deterministic=True)

        path = str(tmp_path / "test_model.npz")
        evaluator.save(path)

        evaluator2 = RLSignalEvaluator()
        evaluator2.load(path)
        result2 = evaluator2.evaluate(obs, deterministic=True)

        assert abs(result1.confidence - result2.confidence) < 1e-5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_rl_signal_evaluator_ml.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_ml_signal_obs'`

- [ ] **Step 3: Update rl_signal_evaluator.py**

Make these changes to `bot/learning/rl_signal_evaluator.py`:

1. Change `OBS_DIM = 32` → `OBS_DIM = 30` (line 24)
2. Remove `STRATEGY_IDS` dict (lines 37-40)
3. Keep `build_signal_obs()` as-is (the old strategy-based engine path still uses it until Task 10 adds the ML path). Add `# LEGACY: will be removed once ML signal path is fully wired` comment above it
4. Add new `build_ml_signal_obs()` function:

```python
def build_ml_signal_obs(
    # ML predictions (12 probabilities)
    probabilities: list[float] | None = None,
    # Portfolio state (10 features)
    equity_ratio: float = 1.0,
    drawdown_pct: float = 0.0,
    open_pos_ratio: float = 0.0,
    win_rate_recent: float = 0.5,
    avg_pnl_recent: float = 0.0,
    bars_since_trade: int = 100,
    balance_ratio: float = 1.0,
    recent_loss_streak: int = 0,
    total_trades: int = 0,
    recent_sharpe: float = 0.0,
    # Coin markers
    symbol: str = "BTC-EUR",
) -> np.ndarray:
    """Build a 30-dim observation vector from ML predictions + portfolio state.

    Layout:
        0-11:  ML predictions (prob_up/down × 6 horizons)
        12-15: Cross-horizon features (max_prob, min_prob, horizon_agreement, trend_alignment)
        16-25: Portfolio state
        26-29: Coin markers
    """
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    probs = probabilities if probabilities is not None else [0.5] * 12

    # ML predictions (12 features, indices 0-11)
    for i in range(min(12, len(probs))):
        obs[i] = np.clip(probs[i], 0.0, 1.0)

    # Cross-horizon features (4 features, indices 12-15)
    obs[12] = max(probs)                              # max_prob
    obs[13] = min(probs)                              # min_prob
    # Horizon agreement: how many horizons agree on direction
    up_probs = [probs[i] for i in range(0, 12, 2)]
    down_probs = [probs[i] for i in range(1, 12, 2)]
    up_votes = sum(1 for p in up_probs if p > 0.5)
    down_votes = sum(1 for p in down_probs if p > 0.5)
    obs[14] = max(up_votes, down_votes) / 6.0         # horizon_agreement (0-1)
    # Trend alignment: net direction strength
    net_up = np.mean(up_probs)
    net_down = np.mean(down_probs)
    obs[15] = np.clip(net_up - net_down, -1.0, 1.0)   # trend_alignment

    # Portfolio state (10 features, indices 16-25)
    obs[16] = np.clip(equity_ratio, 0.0, 3.0) / 3.0
    obs[17] = np.clip(drawdown_pct / 20.0, 0.0, 1.0)
    obs[18] = np.clip(open_pos_ratio, 0.0, 1.0)
    obs[19] = np.clip(win_rate_recent, 0.0, 1.0)
    obs[20] = np.clip(avg_pnl_recent / 5.0, -1.0, 1.0)
    obs[21] = np.clip(bars_since_trade / 500.0, 0.0, 1.0)
    obs[22] = np.clip(balance_ratio, 0.0, 3.0) / 3.0
    obs[23] = np.clip(recent_loss_streak / 10.0, 0.0, 1.0)
    obs[24] = np.clip(total_trades / 500.0, 0.0, 1.0)
    obs[25] = np.clip(recent_sharpe / 3.0, -1.0, 1.0)

    # Coin markers (4 features, indices 26-29)
    coin = get_coin_profile(symbol)
    obs[26] = coin["cap_tier"] / 3.0
    obs[27] = coin["vol_class"] / 2.0
    obs[28] = np.clip(coin["age"] / 16.0, 0.0, 1.0)
    obs[29] = coin["is_btc"]

    return obs
```

5. Update `_init_weights()` — the `OBS_DIM` change automatically flows through since it uses the constant.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_rl_signal_evaluator_ml.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Run existing RL tests to check for regressions**

Run: `python -m pytest tests/ -k "rl" -v`
Expected: Check for failures. If existing tests use `build_signal_obs`, they'll need updating to use `_build_signal_obs_legacy` or the new `build_ml_signal_obs`. Fix any failures.

- [ ] **Step 6: Commit**

```bash
git add bot/learning/rl_signal_evaluator.py tests/test_rl_signal_evaluator_ml.py
git commit -m "feat: update RL signal evaluator to 30-dim ML observation space"
```

---

### Task 9: Update GPU and RL Environment Dimensions

**Files:**
- Modify: `bot/learning/gpu_backtest_kernel.py:185` — `SIGNAL_EVERY = 12` → `SIGNAL_EVERY = 6`
- Modify: `bot/learning/gpu_vec_env.py` — `_OBS_DIM = 32` → `_OBS_DIM = 30`
- Modify: `bot/learning/rl_environment.py` — update obs_dim references from 32 to 30

These are small constant changes to keep all engine implementations consistent.

- [ ] **Step 1: Update GPU backtest kernel SIGNAL_EVERY**

In `bot/learning/gpu_backtest_kernel.py`, find `SIGNAL_EVERY = 12` and change to `SIGNAL_EVERY = 6`.

- [ ] **Step 2: Update GPU vec env OBS_DIM**

In `bot/learning/gpu_vec_env.py`, find `_OBS_DIM = 32` and change to `_OBS_DIM = 30`. Also update the observation building code to match the new 30-dim layout (remove strategy ID slot, adjust indices for portfolio state and coin markers).

- [ ] **Step 3: Update RL environment obs_dim**

In `bot/learning/rl_environment.py`, find any hardcoded `32` references for obs_dim and change to `30`. The observation space shape should reference the `OBS_DIM` constant from `rl_signal_evaluator.py` instead of a hardcoded value.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/ -k "gpu or rl_env or backtest" -v`
Expected: All tests PASS (or skip if GPU not available)

- [ ] **Step 5: Commit**

```bash
git add bot/learning/gpu_backtest_kernel.py bot/learning/gpu_vec_env.py bot/learning/rl_environment.py
git commit -m "fix: update SIGNAL_EVERY to 6 and OBS_DIM to 30 across all engine implementations"
```

---

### Task 10: Backtest Engine ML Signal Integration

**Files:**
- Modify: `bot/backtest/engine.py:113-128,569-695`
- Create: `tests/test_engine_ml_signals.py`

Add `ml_signal_generator` parameter to `BacktestEngine`. When set, the engine uses ML predictions instead of the strategy router for signal generation.

- [ ] **Step 1: Write tests for ML signal path in engine**

```python
# tests/test_engine_ml_signals.py
"""Tests for backtest engine ML signal integration."""
import json
import numpy as np
import pandas as pd
import pytest


def _make_5m_candles(bars: int = 3000) -> list:
    np.random.seed(42)
    timestamps = pd.date_range("2024-01-01", periods=bars, freq="5min")
    base = 50000.0
    returns = np.random.randn(bars) * 0.001
    close = base * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.randn(bars)) * 0.002)
    low = close * (1 - np.abs(np.random.randn(bars)) * 0.002)
    open_ = close * (1 + np.random.randn(bars) * 0.001)
    volume = np.random.exponential(100, bars)
    candles = []
    for i in range(bars):
        candles.append({
            "timestamp": str(timestamps[i]),
            "open": float(open_[i]), "high": float(high[i]),
            "low": float(low[i]), "close": float(close[i]),
            "volume": float(volume[i]),
        })
    return candles


class MockMLSignalGenerator:
    """Mock ML signal generator for testing."""
    def __init__(self, direction="LONG", prob=0.6):
        self._direction = direction
        self._prob = prob
        self.predict_count = 0

    def predict(self, df_5m, symbol="BTC-EUR", **kwargs):
        self.predict_count += 1
        up = self._prob if self._direction == "LONG" else 0.3
        down = self._prob if self._direction == "SHORT" else 0.3
        return {
            "probabilities": [up, down] * 6,
            "direction": self._direction,
            "net_up": up,
            "net_down": down,
        }


class TestEngineMlSignalPath:
    def test_engine_accepts_ml_signal_generator(self):
        from bot.backtest.engine import BacktestEngine
        candles = _make_5m_candles(3000)
        gen = MockMLSignalGenerator()
        engine = BacktestEngine(
            candles, ml_signal_generator=gen,
            initial_capital=10_000,
        )
        result = engine.run()
        assert result is not None
        assert gen.predict_count > 0

    def test_ml_path_generates_trades(self):
        from bot.backtest.engine import BacktestEngine
        candles = _make_5m_candles(3000)
        gen = MockMLSignalGenerator(direction="LONG", prob=0.7)
        engine = BacktestEngine(
            candles, ml_signal_generator=gen,
            initial_capital=10_000,
        )
        result = engine.run()
        assert result.total_trades > 0

    def test_ml_path_with_signal_evaluator(self):
        from bot.backtest.engine import BacktestEngine
        from bot.learning.rl_signal_evaluator import RLSignalEvaluator
        candles = _make_5m_candles(3000)
        gen = MockMLSignalGenerator(direction="LONG", prob=0.7)
        evaluator = RLSignalEvaluator()
        engine = BacktestEngine(
            candles, ml_signal_generator=gen,
            signal_evaluator=evaluator,
            online_learning=True,
            initial_capital=10_000,
        )
        result = engine.run()
        assert result is not None

    def test_ml_path_no_direction_skips_signal(self):
        from bot.backtest.engine import BacktestEngine
        candles = _make_5m_candles(3000)
        gen = MockMLSignalGenerator(direction=None, prob=0.3)
        gen._direction = None
        engine = BacktestEngine(
            candles, ml_signal_generator=gen,
            initial_capital=10_000,
        )
        result = engine.run()
        # No direction = no trades
        assert result.total_trades == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_engine_ml_signals.py -v`
Expected: FAIL (engine doesn't accept `ml_signal_generator` parameter yet)

- [ ] **Step 3: Modify BacktestEngine**

Add `ml_signal_generator` parameter to `__init__()`:

```python
# In __init__ signature, add:
ml_signal_generator: Optional[Any] = None,

# In __init__ body, add:
self._ml_signal_generator = ml_signal_generator
```

In the `run()` method, after the QUIET regime skip and before strategy dispatch (around line 531), add the ML signal path:

```python
# --- ML signal path (replaces strategy router when ml_signal_generator is set) ---
if self._ml_signal_generator is not None:
    # Build DataFrame window for ML prediction
    ml_df = df.iloc[max(0, abs_i - 2000):abs_i + 1]
    ml_result = self._ml_signal_generator.predict(
        ml_df, symbol=symbol,
    )
    direction = ml_result["direction"]
    if direction is None:
        equity_curve.append(current_equity)
        continue

    self._signals_generated += 1

    # RL Signal Evaluator: gate on ML predictions
    rl_eval_result = None
    use_rl_atr = self._atr_multiplier
    use_rl_rr = self._rr_ratio
    if self._signal_evaluator is not None:
        from bot.learning.rl_signal_evaluator import build_ml_signal_obs
        _eq = self._equity(current_price)
        _eq_ratio = _eq / self.initial_capital if self.initial_capital > 0 else 1.0
        _dd_pct = ((self.peak_balance - _eq) / self.peak_balance * 100.0
                   if self.peak_balance > 0 else 0.0)
        _wr, _aw, _al = self._trade_stats()
        _avg_pnl = (_aw - _al) * 100.0 if self.closed_trades else 0.0

        obs = build_ml_signal_obs(
            probabilities=ml_result["probabilities"],
            equity_ratio=_eq_ratio, drawdown_pct=_dd_pct,
            open_pos_ratio=len(self.positions) / max(self.max_open, 1),
            win_rate_recent=_wr, avg_pnl_recent=_avg_pnl,
            bars_since_trade=i - self._last_trade_close_bar_for_rl,
            balance_ratio=self.balance / self.initial_capital if self.initial_capital > 0 else 1.0,
            recent_loss_streak=self._loss_streak,
            total_trades=len(self.closed_trades),
            recent_sharpe=0.0,
            symbol=symbol,
        )
        rl_eval_result = self._signal_evaluator.evaluate(obs)
        if not rl_eval_result.take_trade:
            equity_curve.append(current_equity)
            continue
        use_rl_atr = rl_eval_result.atr_multiplier
        use_rl_rr = rl_eval_result.rr_ratio

    # --- Fill rate model (same as existing, lines 647-654) ---
    limit_price = current_price
    if i + 1 < n:
        if direction == "LONG" and lows[i + 1] > limit_price:
            equity_curve.append(current_equity)
            continue
        elif direction == "SHORT" and highs[i + 1] < limit_price:
            equity_curve.append(current_equity)
            continue

    self._signals_filled += 1

    # --- Stop calculation (same as existing, lines 658-670) ---
    window_1h = df_1h.iloc[max(0, h1_idx - 100): h1_idx + 1]
    _effective_atr_mult = use_rl_atr
    _effective_rr = use_rl_rr
    sl, tp = initial_stops(
        current_price, window_1h, direction=direction,
        atr_multiplier=_effective_atr_mult,
        rr_ratio=_effective_rr,
        total_fee_pct=(MAKER_FEE_PCT + TAKER_FEE_PCT) * 100.0,
    )

    # --- Fee gate (same as existing, lines 672-683) ---
    tp_distance_pct = abs(tp - current_price) / current_price * 100.0 if current_price > 0 else 0.0
    size_eur_est = self._compute_position_size(1.0, current_price, atr_pct=atr_pct)
    fee_result = check_fee_gate(
        position_size=size_eur_est,
        tp_distance_pct=tp_distance_pct,
        is_short=(direction == "SHORT"),
        min_profit_multiple=self._strategy_params.get("min_profit_multiple", 3.0),
    )
    if not fee_result.approved:
        equity_curve.append(current_equity)
        continue

    if atr_pct > 0:
        self._atr_pct_history.append(atr_pct)
        if len(self._atr_pct_history) > 200:
            self._atr_pct_history = self._atr_pct_history[-200:]

    # Build a synthetic Signal object for _open_position
    ml_signal = Signal(
        direction=direction,
        strength=max(ml_result["net_up"], ml_result["net_down"]),
        strategy_name="ml_signal",
    )
    self._open_position(
        ml_signal, current_price, current_time,
        window_1h, 1.0, bar_index=i,
        atr_pct=atr_pct,
        regime=regime,
        rl_eval=rl_eval_result,
    )

    equity_curve.append(current_equity)
    continue

# --- existing strategy-based signal path (unchanged, used when ml_signal_generator is None) ---
strategies = self._router.get_strategies(regime)
# ... rest of existing code ...
```

The key change: when `ml_signal_generator` is set, the engine calls `predict()` instead of running strategies through the router. The RL evaluator receives the new 30-dim observation. All existing risk management (fee gate, position sizing, drawdown scaling, stop management) stays the same.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_engine_ml_signals.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Run full engine test suite**

Run: `python -m pytest tests/test_backtest_engine_overhaul.py -v`
Expected: All existing tests still PASS (old strategy path unchanged)

- [ ] **Step 6: Commit**

```bash
git add bot/backtest/engine.py tests/test_engine_ml_signals.py
git commit -m "feat: add ML signal generator path to backtest engine"
```

---

## Chunk 4: Live Integration and Script Updates

### Task 11: Wire MLSignalGenerator into Trading Loop and Main

**Files:**
- Modify: `bot/trading_loop.py`
- Modify: `bot/main.py`

When trained ML models exist, the live trading loop uses `MLSignalGenerator.predict()` instead of the strategy router. The change is in the signal generation path only — all risk management, order management, and online learning stay the same.

- [ ] **Step 1: Modify trading_loop.py**

In `TradingLoop.__init__()`, add `ml_signal_generator` parameter:

```python
def __init__(self, ..., ml_signal_generator=None, ...):
    ...
    self._ml_signal_generator = ml_signal_generator
```

In the signal generation handler (the candle processing path), add ML signal path before the strategy router call:

```python
# When ML signal generator is available, use it instead of strategy router
if self._ml_signal_generator is not None:
    ml_result = self._ml_signal_generator.predict(
        df_5m_window, symbol=symbol,
        btc_df_5m=btc_window,
        funding_rate=funding_rate,
        funding_score=funding_score,
        ob_imbalance=ob_imbalance,
        spread_pct=spread_pct,
        bid_ask_wall_ratio=bid_ask_wall_ratio,
        onchain_composite=onchain_composite,
        exchange_reserve_trend=exchange_reserve_trend,
    )
    direction = ml_result["direction"]
    if direction is None:
        return  # no signal

    # Build RL observation from ML predictions
    from bot.learning.rl_signal_evaluator import build_ml_signal_obs
    equity = self._portfolio.total_equity()
    initial = self._portfolio.initial_capital
    obs = build_ml_signal_obs(
        probabilities=ml_result["probabilities"],
        equity_ratio=equity / initial if initial > 0 else 1.0,
        drawdown_pct=self._portfolio.current_drawdown_pct(),
        open_pos_ratio=len(self._portfolio.open_positions) / max(self._max_positions, 1),
        win_rate_recent=self._trade_analyzer.recent_win_rate(),
        avg_pnl_recent=self._trade_analyzer.recent_avg_pnl() * 100.0,
        bars_since_trade=self._bars_since_last_trade,
        balance_ratio=self._portfolio.balance / initial if initial > 0 else 1.0,
        recent_loss_streak=self._loss_streak,
        total_trades=self._trade_analyzer.total_trades,
        recent_sharpe=self._trade_analyzer.recent_sharpe(),
        symbol=symbol,
    )
    # Evaluate with signal_evaluator and open position (same pattern as existing code)
    rl_eval_result = self._signal_evaluator.evaluate(obs)
    if not rl_eval_result.take_trade:
        return
    # ... open position using existing order manager with RL-adjusted params
    return

# --- existing strategy router path (only used if ml_signal_generator was not set) ---
```

- [ ] **Step 2: Modify main.py**

In the bot startup code, conditionally load `MLSignalGenerator`:

```python
# In _get_ml_signal_generator() or startup:
from pathlib import Path
_ml_signal_generator = None

def _get_ml_signal_generator():
    """Load ML signal generator. Raises if models not found (per spec: no fallback)."""
    global _ml_signal_generator
    if _ml_signal_generator is None:
        model_dir = "models/ml_signals"
        if Path(model_dir).exists() and (Path(model_dir) / "lstm.pt").exists():
            from bot.learning.ml_signal_generator import MLSignalGenerator
            _ml_signal_generator = MLSignalGenerator(model_dir=model_dir)
            logger.info("ML Signal Generator loaded from %s", model_dir)
        else:
            raise FileNotFoundError(
                f"ML models not found at {model_dir}. "
                f"Run 'python scripts/train_ml_signals.py' first. "
                f"The bot requires trained ML models to start."
            )
    return _ml_signal_generator
```

Pass it to `TradingLoop`:

```python
trading_loop = TradingLoop(
    ...,
    ml_signal_generator=_get_ml_signal_generator(),
    ...,
)
```

- [ ] **Step 3: Test the wiring**

Run: `python -c "from bot.main import _get_ml_signal_generator; print(_get_ml_signal_generator())"`
Expected: `None` (no trained models yet) or the MLSignalGenerator instance if models exist

- [ ] **Step 4: Commit**

```bash
git add bot/trading_loop.py bot/main.py
git commit -m "feat: wire ML signal generator into live trading loop"
```

---

### Task 12: Update Pretrain Signal Evaluator Script

**Files:**
- Modify: `scripts/pretrain_signal_evaluator.py`

Update the pre-training script to use ML signal observations (`build_ml_signal_obs`) when an `MLSignalGenerator` is available, falling back to the legacy `_build_signal_obs_legacy` when not.

- [ ] **Step 1: Update collect_training_data**

Replace the observation building in `_ObsCollector` to use `build_ml_signal_obs` when `ml_signal_generator` is passed:

```python
class _ObsCollector(RLSignalEvaluator):
    def __init__(self, ml_signal_generator=None):
        super().__init__()
        self._ml_gen = ml_signal_generator
        self.experiences = []

    def evaluate(self, obs, deterministic=False):
        result = super().evaluate(obs, deterministic=False)
        result.take_trade = True
        return result

    def record_outcome(self, eval_result, reward):
        if eval_result.obs is not None:
            self.experiences.append((
                eval_result.obs.copy(),
                eval_result.action.copy(),
                eval_result.log_prob,
                reward,
            ))
        return False
```

Update `collect_training_data()` to accept and pass through `ml_signal_generator`:

```python
def collect_training_data(candles, pair, ml_signal_generator=None):
    collector = _ObsCollector(ml_signal_generator=ml_signal_generator)
    engine = BacktestEngine(
        candles, initial_capital=CAPITAL, max_open_positions=10,
        ml_signal_generator=ml_signal_generator,
        signal_evaluator=collector,
        online_learning=True,
    )
    engine.run()
    return collector.experiences
```

Update `main()` to optionally load MLSignalGenerator:

```python
# In main(), after fetching candles:
ml_gen = None
model_dir = "models/ml_signals"
if Path(model_dir).exists() and (Path(model_dir) / "lstm.pt").exists():
    from bot.learning.ml_signal_generator import MLSignalGenerator
    ml_gen = MLSignalGenerator(model_dir=model_dir)
    print(f"  Using ML Signal Generator for pre-training")
else:
    print(f"  ML models not found — using strategy-based signals for pre-training")

# Pass ml_gen to collect_training_data
experiences = collect_training_data(candles, pair, ml_signal_generator=ml_gen)
```

- [ ] **Step 2: Verify script imports and new code paths**

Run: `python -c "import sys; sys.path.insert(0,'.'); from scripts.pretrain_signal_evaluator import collect_training_data, _ObsCollector; print('Import OK, ObsCollector accepts ml_signal_generator:', 'ml_signal_generator' in _ObsCollector.__init__.__code__.co_varnames if hasattr(_ObsCollector.__init__, '__code__') else 'N/A')"`
Expected: `Import OK` with confirmation that `_ObsCollector.__init__` accepts `ml_signal_generator`

- [ ] **Step 3: Commit**

```bash
git add scripts/pretrain_signal_evaluator.py
git commit -m "feat: update pretrain script for ML signal observations"
```

---

### Task 13: Add ML Test Mode to rl_vs_champion Script

**Files:**
- Modify: `scripts/rl_vs_champion.py`

Add a test mode that uses `MLSignalGenerator` alongside the RL Signal Evaluator. This allows comparing ML-based signals against the champion defaults.

- [ ] **Step 1: Add ML test to the script**

After the existing test modes (champion defaults, RL online combined, RL per-strategy), add:

```python
# --- ML Signal Generator test ---
ml_gen = None
model_dir = "models/ml_signals"
if Path(model_dir).exists() and (Path(model_dir) / "lstm.pt").exists():
    from bot.learning.ml_signal_generator import MLSignalGenerator
    ml_gen = MLSignalGenerator(model_dir=model_dir)

    print(f"\n{'='*100}")
    print(f"  ML SIGNAL GENERATOR TEST")
    print(f"{'='*100}")

    for pair in PAIRS:
        client = BitvavoClient(api_key="", api_secret="", paper_trading=True)
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=DAYS)
        candles = client.get_candles_range(pair, interval="5m", start=start_dt, end=end_dt)
        if not candles or len(candles) < 1000:
            print(f"  {pair}: skipped (insufficient data)")
            continue

        # ML without RL evaluator
        engine_ml = BacktestEngine(
            candles, initial_capital=CAPITAL,
            ml_signal_generator=ml_gen,
        )
        bt_ml = engine_ml.run()
        print(f"  {pair} ML Only:    {bt_ml.total_trades} trades, "
              f"EUR {bt_ml.total_pnl:+.2f}, Sharpe={bt_ml.sharpe_ratio:+.2f}")

        # ML with RL evaluator + online learning
        evaluator = RLSignalEvaluator()
        engine_ml_rl = BacktestEngine(
            candles, initial_capital=CAPITAL,
            ml_signal_generator=ml_gen,
            signal_evaluator=evaluator,
            online_learning=True,
        )
        bt_ml_rl = engine_ml_rl.run()
        print(f"  {pair} ML + RL:    {bt_ml_rl.total_trades} trades, "
              f"EUR {bt_ml_rl.total_pnl:+.2f}, Sharpe={bt_ml_rl.sharpe_ratio:+.2f}")
else:
    print(f"\n  ML models not found at {model_dir} — skipping ML test")
```

- [ ] **Step 2: Commit**

```bash
git add scripts/rl_vs_champion.py
git commit -m "feat: add ML signal generator test mode to rl_vs_champion script"
```

---

### Task 14: Add xgboost Dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add xgboost to requirements.txt**

Add `xgboost>=2.0.0` to the ML section of requirements.txt (after `stable-baselines3`). Also add explicit `torch>=2.0` if not already present.

- [ ] **Step 2: Install**

Run: `pip install xgboost>=2.0.0`
Expected: Successfully installed

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "deps: add xgboost and explicit torch dependency"
```
