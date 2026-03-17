"""Gymnasium environment for PPO-based trading parameter optimization."""
from __future__ import annotations

import logging
import math
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import gymnasium
import numpy as np
import pandas as pd
from gymnasium import spaces

from bot.backtest.engine import BacktestEngine, BacktestResult
from bot.learning.feature_extractor import extract_features

logger = logging.getLogger(__name__)

# Default parameter table (fallback)
_DEFAULT_PARAMS: List[Tuple[str, float, float, bool]] = [
    ("atr_multiplier",          1.0,   6.0, False),
    ("rr_ratio",                1.0,   5.0, False),
    ("base_risk_pct",           1.0,   5.0, False),
    ("min_profit_multiple",     1.0,   4.0, False),
    ("max_hold_hours",         12.0, 168.0, True),
    ("quiet_atr_threshold",     0.3,   2.0, False),
    ("regime_adx_threshold",   15.0,  35.0, False),
    ("ranging_adx_threshold",  15.0,  30.0, False),
    ("signal_strength_min",     0.1,   0.5, False),
    ("tf_weight_1h",            0.3,   1.0, False),
    ("tf_weight_4h",            0.0,   1.0, False),
    ("tf_weight_1d",            0.0,   1.0, False),
    ("max_concurrent_positions", 1.0,  5.0, True),
    ("confidence_size_scaling",  0.0,  1.5, False),
    ("ema200_filter_pct",       0.0,   5.0, False),
    ("volatile_atr_threshold",  2.0,   6.0, False),
    ("consecutive_confirms",    1.0,   2.0, True),
    ("confluence_boost",        1.0,   1.5, False),
    ("drawdown_scale_pct",      2.0,  10.0, False),
    ("max_position_pct",        0.1,   0.4, False),
    ("trail_activation_mult",   0.5,   2.0, False),
]

# Per-strategy parameter ranges — tighter ranges tuned to each strategy's nature
STRATEGY_PARAM_TABLES: Dict[str, List[Tuple[str, float, float, bool]]] = {
    "orderflow": [
        ("atr_multiplier",          1.5,   4.0, False),  # Trend-following
        ("rr_ratio",                1.5,   4.0, False),  # Higher RR for momentum
        ("base_risk_pct",           1.0,   3.0, False),
        ("min_profit_multiple",     1.0,   3.0, False),
        ("max_hold_hours",         24.0, 168.0, True),   # Longer holds for order flow
        ("quiet_atr_threshold",     0.3,   1.5, False),
        ("regime_adx_threshold",   15.0,  30.0, False),
        ("ranging_adx_threshold",  15.0,  25.0, False),
        ("signal_strength_min",     0.1,   0.4, False),
        ("tf_weight_1h",            0.3,   1.0, False),
        ("tf_weight_4h",            0.2,   1.0, False),
        ("tf_weight_1d",            0.0,   0.8, False),
        ("max_concurrent_positions", 1.0,  3.0, True),
        ("confidence_size_scaling",  0.0,  1.5, False),
        ("ema200_filter_pct",       0.0,   3.0, False),
        ("volatile_atr_threshold",  2.0,   5.0, False),
        ("consecutive_confirms",    1.0,   2.0, True),
        ("confluence_boost",        1.0,   1.5, False),
        ("drawdown_scale_pct",      3.0,   8.0, False),
        ("max_position_pct",        0.1,   0.3, False),
        ("trail_activation_mult",   0.5,   2.0, False),
    ],
    "range": [
        ("atr_multiplier",          1.0,   3.0, False),  # Tighter stops for mean-rev
        ("rr_ratio",                1.0,   3.0, False),  # Lower RR, higher win rate
        ("base_risk_pct",           1.0,   3.0, False),
        ("min_profit_multiple",     1.0,   2.5, False),
        ("max_hold_hours",          6.0,  72.0, True),   # Shorter holds for ranging
        ("quiet_atr_threshold",     0.3,   1.5, False),
        ("regime_adx_threshold",   15.0,  30.0, False),
        ("ranging_adx_threshold",  12.0,  25.0, False),  # Lower to detect ranges better
        ("signal_strength_min",     0.1,   0.4, False),
        ("tf_weight_1h",            0.5,   1.0, False),  # Higher 1h weight for range
        ("tf_weight_4h",            0.0,   0.8, False),
        ("tf_weight_1d",            0.0,   0.5, False),
        ("max_concurrent_positions", 1.0,  4.0, True),
        ("confidence_size_scaling",  0.0,  1.0, False),
        ("ema200_filter_pct",       0.0,   3.0, False),
        ("volatile_atr_threshold",  2.0,   4.0, False),
        ("consecutive_confirms",    1.0,   2.0, True),
        ("confluence_boost",        1.0,   1.3, False),
        ("drawdown_scale_pct",      2.0,   8.0, False),
        ("max_position_pct",        0.1,   0.3, False),
        ("trail_activation_mult",   0.5,   1.5, False),
        ("range_max_hold_hours",    6.0,  72.0, True),   # Extra param for range
    ],
    "squeeze": [
        ("atr_multiplier",          2.0,   6.0, False),  # Wider stops for breakouts
        ("rr_ratio",                2.0,   5.0, False),  # Higher RR for breakouts
        ("base_risk_pct",           1.0,   4.0, False),
        ("min_profit_multiple",     1.5,   4.0, False),
        ("max_hold_hours",         24.0, 168.0, True),   # Longer for breakout follow-through
        ("quiet_atr_threshold",     0.3,   1.2, False),  # Lower = detect squeeze better
        ("regime_adx_threshold",   15.0,  35.0, False),
        ("ranging_adx_threshold",  15.0,  30.0, False),
        ("signal_strength_min",     0.15,  0.5, False),  # Higher min for squeeze confidence
        ("tf_weight_1h",            0.3,   1.0, False),
        ("tf_weight_4h",            0.2,   1.0, False),
        ("tf_weight_1d",            0.0,   1.0, False),
        ("max_concurrent_positions", 1.0,  3.0, True),
        ("confidence_size_scaling",  0.2,  1.5, False),
        ("ema200_filter_pct",       0.0,   5.0, False),
        ("volatile_atr_threshold",  2.5,   6.0, False),  # Higher for squeeze detection
        ("consecutive_confirms",    1.0,   2.0, True),
        ("confluence_boost",        1.0,   1.5, False),
        ("drawdown_scale_pct",      3.0,  10.0, False),
        ("max_position_pct",        0.1,   0.4, False),  # Allow larger for high-conviction
        ("trail_activation_mult",   0.8,   2.0, False),
    ],
    "funding_contrarian": [
        ("atr_multiplier",          1.5,   4.0, False),
        ("rr_ratio",                1.5,   4.0, False),
        ("base_risk_pct",           1.0,   3.0, False),
        ("min_profit_multiple",     1.0,   3.0, False),
        ("max_hold_hours",         12.0, 120.0, True),   # Medium duration
        ("quiet_atr_threshold",     0.3,   1.5, False),
        ("regime_adx_threshold",   15.0,  30.0, False),
        ("ranging_adx_threshold",  15.0,  25.0, False),
        ("signal_strength_min",     0.1,   0.4, False),
        ("tf_weight_1h",            0.3,   1.0, False),
        ("tf_weight_4h",            0.0,   1.0, False),
        ("tf_weight_1d",            0.0,   0.8, False),
        ("max_concurrent_positions", 1.0,  4.0, True),
        ("confidence_size_scaling",  0.0,  1.5, False),
        ("ema200_filter_pct",       0.0,   3.0, False),
        ("volatile_atr_threshold",  2.0,   5.0, False),
        ("consecutive_confirms",    1.0,   2.0, True),
        ("confluence_boost",        1.0,   1.5, False),
        ("drawdown_scale_pct",      2.0,   8.0, False),
        ("max_position_pct",        0.1,   0.3, False),
        ("trail_activation_mult",   0.5,   2.0, False),
    ],
}


def get_param_table(strategy: str) -> List[Tuple[str, float, float, bool]]:
    """Get the parameter table for a specific strategy."""
    return STRATEGY_PARAM_TABLES.get(strategy, _DEFAULT_PARAMS)


def action_to_params(action: np.ndarray, strategy: str) -> Dict[str, Any]:
    """Map [-1, 1] action array to parameter dict."""
    table = get_param_table(strategy)

    params = {}
    for i, (name, lo, hi, is_int) in enumerate(table):
        if i >= len(action):
            break
        # Linear map: [-1, 1] -> [lo, hi]
        val = (action[i] + 1.0) / 2.0 * (hi - lo) + lo
        if is_int:
            val = int(round(val))
        params[name] = val
    return params


class TradingParamEnv(gymnasium.Env):
    """Single-step RL environment for trading parameter optimization.

    reset(): pick random symbol + 90-day window, compute features
    step(action): map to params, backtest, return reward
    """

    metadata = {"render_modes": []}

    # Shared disk cache directory — set by RLTrainer before spawning workers
    _signal_cache_dir: Optional[str] = None
    # Class-level shared RAM cache (avoids 4x duplication in DummyVecEnv)
    _shared_1m: Dict[str, pd.DataFrame] = {}
    _shared_5m: Dict[str, pd.DataFrame] = {}
    _shared_signals: Dict[str, Any] = {}

    def __init__(self, candle_store, symbols: List[str], strategy: str,
                 window_days: int = 90, validation_mode: bool = False,
                 n_segments: int = 3):
        super().__init__()
        self._candle_store = candle_store
        self._symbols = symbols
        self._strategy = strategy
        self._window_days = window_days
        self._validation_mode = validation_mode
        self._n_segments = n_segments

        # Observation: 27 market features + 5 performance features (multi-step)
        obs_dim = 32 if n_segments > 1 else 27
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32,
        )
        param_table = get_param_table(strategy)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(param_table),), dtype=np.float32,
        )

        # Pre-compute available windows per symbol
        self._windows: List[Tuple[str, datetime, datetime]] = []
        self._build_window_list()

        # Preload ALL candle data into RAM (one DB read per symbol, zero during training)
        self._preloaded_1m: Dict[str, pd.DataFrame] = {}
        self._preloaded_5m: Dict[str, pd.DataFrame] = {}
        # Pre-computed indicator signals per symbol (full history)
        self._preloaded_signals: Dict[str, Any] = {}
        self._preload_candles()

        self._current_obs: Optional[np.ndarray] = None
        self._current_candles_5m = None
        self._current_symbol: Optional[str] = None

        # Multi-step state
        self._segment_idx: int = 0
        self._segment_boundaries: List[Tuple[datetime, datetime]] = []
        self._cumulative_pnl: float = 0.0
        self._cumulative_trades: int = 0
        self._cumulative_signals: int = 0
        self._max_drawdown: float = 0.0

    def _build_window_list(self) -> None:
        """Build list of (symbol, start, end) windows from available data."""
        step = timedelta(days=14)
        window = timedelta(days=self._window_days)
        for sym in self._symbols:
            date_range = self._candle_store.get_date_range(sym)
            if date_range is None:
                continue
            min_ts, max_ts = date_range
            if not hasattr(min_ts, 'tzinfo') or min_ts.tzinfo is None:
                min_ts = min_ts.replace(tzinfo=timezone.utc)
                max_ts = max_ts.replace(tzinfo=timezone.utc)
            total_days = (max_ts - min_ts).days
            if total_days < self._window_days:
                continue
            # Split: first 90% for training, last 10% for validation
            split_point = min_ts + timedelta(days=int(total_days * 0.9))
            if self._validation_mode:
                start = split_point
                while start + window <= max_ts:
                    self._windows.append((sym, start, start + window))
                    start += step
            else:
                start = min_ts
                while start + window <= split_point:
                    self._windows.append((sym, start, start + window))
                    start += step

    def _preload_candles(self) -> None:
        """Load all candle data into RAM — one DB read per symbol, zero during training.

        Uses class-level shared dicts so DummyVecEnv instances share one copy.
        If a disk cache directory is set, loads from .npz files instead of DB.
        """
        unique_symbols = set(w[0] for w in self._windows)
        if "BTC-EUR" not in unique_symbols and any(s != "BTC-EUR" for s in unique_symbols):
            unique_symbols.add("BTC-EUR")

        cache_dir = TradingParamEnv._signal_cache_dir
        cls = TradingParamEnv

        for sym in sorted(unique_symbols):
            # ── Reuse shared RAM data if already loaded by another env ──
            if sym in cls._shared_1m:
                self._preloaded_1m[sym] = cls._shared_1m[sym]
                self._preloaded_5m[sym] = cls._shared_5m[sym]
                self._preloaded_signals[sym] = cls._shared_signals[sym]
                continue

            # ── Try loading from disk cache first ─────────────────────
            if cache_dir:
                cache_file = os.path.join(cache_dir, f"{sym.replace('-', '_')}.npz")
                if os.path.exists(cache_file):
                    try:
                        self._load_from_cache(sym, cache_file)
                        # Store in shared cache for other envs
                        cls._shared_1m[sym] = self._preloaded_1m[sym]
                        cls._shared_5m[sym] = self._preloaded_5m[sym]
                        cls._shared_signals[sym] = self._preloaded_signals[sym]
                        continue
                    except Exception as exc:
                        logger.warning("Cache load failed for %s: %s — falling back to DB", sym, exc)

            # ── Original path: load from DB + compute ─────────────────
            self._preload_symbol_from_db(sym, cache_dir)
            # Store in shared cache
            if sym in self._preloaded_1m:
                cls._shared_1m[sym] = self._preloaded_1m[sym]
                cls._shared_5m[sym] = self._preloaded_5m[sym]
                cls._shared_signals[sym] = self._preloaded_signals[sym]

        total_1m = sum(len(df) for df in self._preloaded_1m.values())
        mem_mb = sum(df.memory_usage(deep=True).sum() for df in self._preloaded_1m.values()) / 1024 / 1024
        logger.info("Preloaded %d symbols (%d 1m candles, %.0f MB) into RAM",
                     len(self._preloaded_1m), total_1m, mem_mb)

    def _load_from_cache(self, sym: str, cache_file: str) -> None:
        """Load precomputed signals + candle data from a .npz cache file."""
        data = np.load(cache_file, allow_pickle=False)

        # Reconstruct 1m DataFrame (pandas 2.0+ stores as datetime64[us])
        idx_1m = pd.to_datetime(data["ts_1m"], unit="us", utc=True)
        df_1m = pd.DataFrame({
            "open": data["ohlcv_1m"][:, 0],
            "high": data["ohlcv_1m"][:, 1],
            "low": data["ohlcv_1m"][:, 2],
            "close": data["ohlcv_1m"][:, 3],
            "volume": data["ohlcv_1m"][:, 4],
        }, index=idx_1m).astype(np.float32)
        self._preloaded_1m[sym] = df_1m

        # Reconstruct 5m DataFrame
        idx_5m = pd.to_datetime(data["ts_5m"], unit="us", utc=True)
        df_5m = pd.DataFrame({
            "open": data["ohlcv_5m"][:, 0],
            "high": data["ohlcv_5m"][:, 1],
            "low": data["ohlcv_5m"][:, 2],
            "close": data["ohlcv_5m"][:, 3],
            "volume": data["ohlcv_5m"][:, 4],
        }, index=idx_5m).astype(np.float32)
        self._preloaded_5m[sym] = df_5m

        # Reconstruct resampled DataFrames (1h, 4h, 1d)
        def _rebuild_df(prefix):
            idx = pd.to_datetime(data[f"ts_{prefix}"], unit="us", utc=True)
            ohlcv = data[f"ohlcv_{prefix}"]
            return pd.DataFrame({
                "open": ohlcv[:, 0], "high": ohlcv[:, 1],
                "low": ohlcv[:, 2], "close": ohlcv[:, 3],
                "volume": ohlcv[:, 4],
            }, index=idx).astype(np.float32)

        df_1h = _rebuild_df("1h")
        df_4h = _rebuild_df("4h")
        df_1d = _rebuild_df("1d")

        # Reconstruct signal dicts
        signal_keys = [k for k in data.files if k.startswith("sig_")]
        signals = {"5m": {}, "1h": {}, "4h": {}, "1d": {}}
        for k in signal_keys:
            # Format: sig_{tf}_{name}  e.g. sig_1h_rsi
            parts = k.split("_", 2)  # ['sig', '1h', 'rsi']
            tf = parts[1]
            name = parts[2]
            signals[tf][name] = data[k]

        self._preloaded_signals[sym] = {
            "df_1h": df_1h, "df_4h": df_4h, "df_1d": df_1d,
            "signals_5m": signals["5m"], "signals_1h": signals["1h"],
            "signals_4h": signals["4h"], "signals_1d": signals["1d"],
            "regime_adx_4h": data["regime_adx_4h"],
            "regime_atr_4h": data["regime_atr_4h"],
            "regime_close_4h": data["regime_close_4h"],
            "regime_atr_1h": data["regime_atr_1h"],
            "regime_close_1h": data["regime_close_1h"],
            "atr_1h_series": data["atr_1h_series"],
            "5m_timestamps": idx_5m, "1h_timestamps": df_1h.index,
            "4h_timestamps": df_4h.index, "1d_timestamps": df_1d.index,
        }
        logger.debug("Loaded %s from cache (%d 5m bars)", sym, len(df_5m))

    def _preload_symbol_from_db(self, sym: str, cache_dir: Optional[str]) -> None:
        """Load one symbol from DB, compute signals, optionally save to cache."""
        from bot.backtest.engine import BacktestEngine
        from bot.indicators.volatility import atr as compute_atr
        from bot.indicators.trend import adx as compute_adx

        date_range = self._candle_store.get_date_range(sym)
        if date_range is None:
            return
        min_ts, max_ts = date_range
        if not hasattr(min_ts, 'tzinfo') or min_ts.tzinfo is None:
            min_ts = min_ts.replace(tzinfo=timezone.utc)
            max_ts = max_ts.replace(tzinfo=timezone.utc)

        df_1m = self._candle_store.get_candles(sym, min_ts, max_ts, resample="1m")
        if df_1m.empty:
            return
        for col in ["open", "high", "low", "close", "volume"]:
            df_1m[col] = df_1m[col].astype(np.float32)
        self._preloaded_1m[sym] = df_1m

        df_5m = df_1m.resample("5min").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()
        self._preloaded_5m[sym] = df_5m

        try:
            df_1h = BacktestEngine._resample_1h(df_5m)
            df_4h = BacktestEngine._resample_4h(df_5m)
            df_1d = BacktestEngine._resample_1d(df_5m)

            signals_5m = BacktestEngine._precompute_signals(df_5m, sym)
            signals_1h = BacktestEngine._precompute_signals(df_1h, sym)
            signals_4h = BacktestEngine._precompute_signals(df_4h, sym)
            signals_1d = BacktestEngine._precompute_signals(df_1d, sym)

            regime_adx_4h   = compute_adx(df_4h["high"], df_4h["low"], df_4h["close"]).values
            regime_atr_4h   = compute_atr(df_4h["high"], df_4h["low"], df_4h["close"]).values
            regime_close_4h = df_4h["close"].values
            regime_atr_1h   = compute_atr(df_1h["high"], df_1h["low"], df_1h["close"]).values
            regime_close_1h = df_1h["close"].values
            atr_1h_series   = compute_atr(df_1h["high"], df_1h["low"], df_1h["close"]).values

            self._preloaded_signals[sym] = {
                "df_1h": df_1h, "df_4h": df_4h, "df_1d": df_1d,
                "signals_5m": signals_5m, "signals_1h": signals_1h,
                "signals_4h": signals_4h, "signals_1d": signals_1d,
                "regime_adx_4h": regime_adx_4h, "regime_atr_4h": regime_atr_4h,
                "regime_close_4h": regime_close_4h,
                "regime_atr_1h": regime_atr_1h, "regime_close_1h": regime_close_1h,
                "atr_1h_series": atr_1h_series,
                "5m_timestamps": df_5m.index, "1h_timestamps": df_1h.index,
                "4h_timestamps": df_4h.index, "1d_timestamps": df_1d.index,
            }

            # Save to disk cache for other workers / future runs
            if cache_dir:
                self._save_to_cache(sym, cache_dir, df_1m, df_5m,
                                    df_1h, df_4h, df_1d,
                                    signals_5m, signals_1h, signals_4h, signals_1d,
                                    regime_adx_4h, regime_atr_4h, regime_close_4h,
                                    regime_atr_1h, regime_close_1h, atr_1h_series)

            logger.debug("Pre-computed signals for %s (%d 5m bars)", sym, len(df_5m))
        except Exception as exc:
            logger.warning("Failed to pre-compute signals for %s: %s", sym, exc)
            self._preloaded_signals.pop(sym, None)

    @staticmethod
    def _save_to_cache(sym, cache_dir, df_1m, df_5m,
                       df_1h, df_4h, df_1d,
                       sig_5m, sig_1h, sig_4h, sig_1d,
                       regime_adx_4h, regime_atr_4h, regime_close_4h,
                       regime_atr_1h, regime_close_1h, atr_1h_series) -> None:
        """Save precomputed data to .npz for fast worker loading."""
        cache_file = os.path.join(cache_dir, f"{sym.replace('-', '_')}.npz")

        def _ohlcv(df):
            return np.column_stack([
                df["open"].values, df["high"].values,
                df["low"].values, df["close"].values,
                df["volume"].values,
            ]).astype(np.float32)

        def _to_epoch_us(idx):
            """Convert DatetimeIndex to epoch microseconds (resolution-agnostic)."""
            return idx.values.astype("datetime64[us]").astype(np.int64)

        arrays = {
            "ts_1m": _to_epoch_us(df_1m.index),
            "ohlcv_1m": _ohlcv(df_1m),
            "ts_5m": _to_epoch_us(df_5m.index),
            "ohlcv_5m": _ohlcv(df_5m),
            "ts_1h": _to_epoch_us(df_1h.index),
            "ohlcv_1h": _ohlcv(df_1h),
            "ts_4h": _to_epoch_us(df_4h.index),
            "ohlcv_4h": _ohlcv(df_4h),
            "ts_1d": _to_epoch_us(df_1d.index),
            "ohlcv_1d": _ohlcv(df_1d),
            "regime_adx_4h": regime_adx_4h.astype(np.float32),
            "regime_atr_4h": regime_atr_4h.astype(np.float32),
            "regime_close_4h": regime_close_4h.astype(np.float32),
            "regime_atr_1h": regime_atr_1h.astype(np.float32),
            "regime_close_1h": regime_close_1h.astype(np.float32),
            "atr_1h_series": atr_1h_series.astype(np.float32),
        }
        # Flatten signal dicts into sig_{tf}_{key} arrays
        for tf, sigs in [("5m", sig_5m), ("1h", sig_1h), ("4h", sig_4h), ("1d", sig_1d)]:
            for key, val in sigs.items():
                arr = np.asarray(val, dtype=np.float32)
                arrays[f"sig_{tf}_{key}"] = arr

        np.savez(cache_file, **arrays)
        logger.debug("Saved signal cache for %s (%.1f MB)", sym,
                     os.path.getsize(cache_file) / 1024 / 1024)

    @classmethod
    def warm_signal_cache(cls, candle_store, symbols: List[str],
                          cache_dir: str) -> None:
        """Pre-populate the disk cache in the parent process.

        Called once by RLTrainer before spawning SubprocVecEnv workers.
        Workers then load from cache (~2s) instead of DB + compute (~30 min).
        """
        os.makedirs(cache_dir, exist_ok=True)
        cls._signal_cache_dir = cache_dir

        # Build a temporary env just to compute windows and figure out symbols
        already_cached = set()
        for f in os.listdir(cache_dir):
            if f.endswith(".npz"):
                already_cached.add(f.replace("_", "-").replace(".npz", ""))

        # Determine which symbols need caching
        unique_symbols = set(symbols)
        if "BTC-EUR" not in unique_symbols:
            unique_symbols.add("BTC-EUR")

        to_compute = sorted(unique_symbols - already_cached)
        if not to_compute:
            logger.info("Signal cache: all %d symbols already cached in %s",
                        len(unique_symbols), cache_dir)
            return

        logger.info("Signal cache: computing %d/%d symbols (cached: %d) → %s",
                     len(to_compute), len(unique_symbols),
                     len(already_cached), cache_dir)

        from bot.backtest.engine import BacktestEngine
        from bot.indicators.volatility import atr as compute_atr
        from bot.indicators.trend import adx as compute_adx

        for i, sym in enumerate(to_compute):
            date_range = candle_store.get_date_range(sym)
            if date_range is None:
                continue
            min_ts, max_ts = date_range
            if not hasattr(min_ts, 'tzinfo') or min_ts.tzinfo is None:
                min_ts = min_ts.replace(tzinfo=timezone.utc)
                max_ts = max_ts.replace(tzinfo=timezone.utc)

            df_1m = candle_store.get_candles(sym, min_ts, max_ts, resample="1m")
            if df_1m.empty:
                continue
            for col in ["open", "high", "low", "close", "volume"]:
                df_1m[col] = df_1m[col].astype(np.float32)

            df_5m = df_1m.resample("5min").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum",
            }).dropna()

            try:
                df_1h = BacktestEngine._resample_1h(df_5m)
                df_4h = BacktestEngine._resample_4h(df_5m)
                df_1d = BacktestEngine._resample_1d(df_5m)

                sig_5m = BacktestEngine._precompute_signals(df_5m, sym)
                sig_1h = BacktestEngine._precompute_signals(df_1h, sym)
                sig_4h = BacktestEngine._precompute_signals(df_4h, sym)
                sig_1d = BacktestEngine._precompute_signals(df_1d, sym)

                regime_adx_4h   = compute_adx(df_4h["high"], df_4h["low"], df_4h["close"]).values
                regime_atr_4h   = compute_atr(df_4h["high"], df_4h["low"], df_4h["close"]).values
                regime_close_4h = df_4h["close"].values
                regime_atr_1h   = compute_atr(df_1h["high"], df_1h["low"], df_1h["close"]).values
                regime_close_1h = df_1h["close"].values
                atr_1h_series   = compute_atr(df_1h["high"], df_1h["low"], df_1h["close"]).values

                cls._save_to_cache(sym, cache_dir, df_1m, df_5m,
                                   df_1h, df_4h, df_1d,
                                   sig_5m, sig_1h, sig_4h, sig_1d,
                                   regime_adx_4h, regime_atr_4h, regime_close_4h,
                                   regime_atr_1h, regime_close_1h, atr_1h_series)

                logger.info("Signal cache: %d/%d %s done (%d 5m bars)",
                            i + 1, len(to_compute), sym, len(df_5m))
            except Exception as exc:
                logger.warning("Signal cache: failed %s: %s", sym, exc)

        logger.info("Signal cache: complete — %d symbols cached", len(to_compute))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs_dim = 32 if self._n_segments > 1 else 27
        if not self._windows:
            self._current_obs = np.zeros(obs_dim, dtype=np.float32)
            return self._current_obs, {}

        # Pick random window
        idx = self.np_random.integers(0, len(self._windows))
        sym, start, end = self._windows[idx]
        self._current_symbol = sym

        # Build segment boundaries for multi-step
        self._segment_idx = 0
        self._cumulative_pnl = 0.0
        self._cumulative_trades = 0
        self._cumulative_signals = 0
        self._max_drawdown = 0.0
        if self._n_segments > 1:
            total_secs = (end - start).total_seconds()
            seg_secs = total_secs / self._n_segments
            self._segment_boundaries = []
            for s in range(self._n_segments):
                seg_start = start + timedelta(seconds=seg_secs * s)
                seg_end = start + timedelta(seconds=seg_secs * (s + 1))
                self._segment_boundaries.append((seg_start, seg_end))
            seg_start, seg_end = self._segment_boundaries[0]
        else:
            self._segment_boundaries = [(start, end)]
            seg_start, seg_end = start, end

        # Slice from preloaded RAM data (zero DB I/O)
        candles_1m = self._preloaded_1m.get(sym, pd.DataFrame())
        if not candles_1m.empty:
            candles_1m = candles_1m.loc[seg_start:seg_end]

        # BTC candles for correlation features
        btc_1m = None
        if sym != "BTC-EUR":
            btc_df = self._preloaded_1m.get("BTC-EUR")
            if btc_df is not None:
                btc_1m = btc_df.loc[seg_start:seg_end]

        # Compute market features
        market_features = extract_features(candles_1m, btc_1m)

        # Build full observation (market features + performance context for multi-step)
        if self._n_segments > 1:
            perf_features = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            self._current_obs = np.concatenate([market_features, perf_features])
        else:
            self._current_obs = market_features

        # Slice 5m from preloaded data
        candles_5m = self._preloaded_5m.get(sym, pd.DataFrame())
        self._current_candles_5m = candles_5m.loc[seg_start:seg_end] if not candles_5m.empty else None

        return self._current_obs, {}

    def step(self, action):
        params = action_to_params(action, self._strategy)

        # Run backtest on current segment
        try:
            if self._current_candles_5m is None or len(self._current_candles_5m) < 100:
                result = BacktestResult()
            else:
                sym = self._current_symbol or "TEST-EUR"
                precomputed = self._preloaded_signals.get(sym)
                engine = BacktestEngine(
                    df=self._current_candles_5m,
                    symbol=sym,
                    strategy_params=params,
                    target_strategy=self._strategy,
                )
                result = engine.run(precomputed=precomputed)
        except Exception as e:
            logger.warning("Backtest failed in env: %s", e)
            result = BacktestResult()

        reward = self._compute_reward(result)

        # Update cumulative stats
        self._cumulative_pnl += result.total_pnl
        self._cumulative_trades += result.total_trades
        self._cumulative_signals += result.signals_generated
        self._max_drawdown = max(self._max_drawdown, result.max_drawdown_pct)

        # Advance to next segment (multi-step) or terminate (single-step)
        self._segment_idx += 1
        terminated = self._segment_idx >= self._n_segments

        if not terminated:
            # Prepare observation for next segment
            seg_start, seg_end = self._segment_boundaries[self._segment_idx]
            sym = self._current_symbol or "TEST-EUR"

            candles_1m = self._preloaded_1m.get(sym, pd.DataFrame())
            if not candles_1m.empty:
                candles_1m = candles_1m.loc[seg_start:seg_end]

            btc_1m = None
            if sym != "BTC-EUR":
                btc_df = self._preloaded_1m.get("BTC-EUR")
                if btc_df is not None:
                    btc_1m = btc_df.loc[seg_start:seg_end]

            market_features = extract_features(candles_1m, btc_1m)

            # Performance context: normalized cumulative stats
            perf_features = np.array([
                max(min(self._cumulative_pnl / 1000.0, 1.0), -1.0),  # norm PnL
                min(self._cumulative_trades / 50.0, 1.0),             # norm trades
                min(self._cumulative_signals / 100.0, 1.0),           # norm signals
                min(self._max_drawdown / 30.0, 1.0),                  # norm drawdown
                self._segment_idx / self._n_segments,                 # progress
            ], dtype=np.float32)
            self._current_obs = np.concatenate([market_features, perf_features])

            # Slice 5m candles for next segment
            candles_5m = self._preloaded_5m.get(sym, pd.DataFrame())
            self._current_candles_5m = candles_5m.loc[seg_start:seg_end] if not candles_5m.empty else None

        return self._current_obs, reward, terminated, False, {"result": result}

    def _compute_reward(self, result: BacktestResult) -> float:
        # P&L is the primary objective (60% weight)
        pnl_norm = max(min(result.total_pnl / 500.0, 5.0), -5.0)
        sharpe_clipped = max(min(result.sharpe_ratio, 3.0), -3.0)
        pf = result.profit_factor
        pf_clipped = min(pf, 5.0) if not (math.isinf(pf) or math.isnan(pf)) else 0.0

        base = (pnl_norm * 0.60
                + sharpe_clipped * 0.25
                + pf_clipped * 0.15)

        penalty = 0.0
        if result.max_drawdown_pct > 20.0:
            penalty -= 3.0

        # Gradient penalty for 0-trade episodes using signals_generated
        # This gives PPO a learning signal even when no trades occur:
        #   0 signals = -1.5 (params too restrictive, nothing even triggers)
        #   Some signals but 0 trades = -0.5 (signals exist but filtered/unfilled)
        if result.total_trades == 0:
            if result.signals_generated == 0:
                penalty -= 1.5
            else:
                # Scale: more signals = closer to trading = less penalty
                sig_ratio = min(result.signals_generated / 20.0, 1.0)
                penalty -= 1.0 - sig_ratio * 0.5  # -1.0 (1 signal) to -0.5 (20+ signals)

        return base + penalty
