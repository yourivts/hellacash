"""Gymnasium environment for PPO-based trading parameter optimization."""
from __future__ import annotations

import logging
import math
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces

from bot.backtest.engine import BacktestEngine, BacktestResult
from bot.learning.feature_extractor import extract_features

logger = logging.getLogger(__name__)

# Parameter mapping table: (name, min, max, is_int)
PARAM_TABLE: List[Tuple[str, float, float, bool]] = [
    ("atr_multiplier",          1.0,  15.0, False),
    ("rr_ratio",                1.0,   8.0, False),
    ("base_risk_pct",           1.0,   8.0, False),
    ("min_profit_multiple",     0.5,   8.0, False),
    ("max_hold_hours",          6.0, 720.0, True),
    ("quiet_atr_threshold",     0.1,   5.0, False),
    ("regime_adx_threshold",    5.0,  50.0, False),
    ("ranging_adx_threshold",   5.0,  50.0, False),
    ("signal_strength_min",     0.1,   1.0, False),
    ("tf_weight_1h",            0.0,   1.0, False),
    ("tf_weight_4h",            0.0,   1.0, False),
    ("tf_weight_1d",            0.0,   1.0, False),
    ("max_concurrent_positions", 1.0, 10.0, True),
    ("confidence_size_scaling",  0.0,  2.0, False),
    ("ema200_filter_pct",       0.0,  10.0, False),
    ("volatile_atr_threshold",  1.0,  10.0, False),
    ("consecutive_confirms",    1.0,   5.0, True),
    ("confluence_boost",        1.0,   2.0, False),
    ("drawdown_scale_pct",      1.0,  15.0, False),
    ("max_position_pct",        0.1,   0.5, False),
    ("trail_activation_mult",   0.5,   3.0, False),
]

RANGE_EXTRA = ("range_max_hold_hours", 6.0, 168.0, True)


def action_to_params(action: np.ndarray, strategy: str) -> Dict[str, Any]:
    """Map [-1, 1] action array to parameter dict."""
    table = list(PARAM_TABLE)
    if strategy == "range":
        table.append(RANGE_EXTRA)

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

    def __init__(self, candle_store, symbols: List[str], strategy: str,
                 window_days: int = 90, validation_mode: bool = False):
        super().__init__()
        self._candle_store = candle_store
        self._symbols = symbols
        self._strategy = strategy
        self._window_days = window_days
        self._validation_mode = validation_mode

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(27,), dtype=np.float32,
        )
        n_actions = 22 if strategy == "range" else 21
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_actions,), dtype=np.float32,
        )

        # Pre-compute available windows per symbol
        self._windows: List[Tuple[str, datetime, datetime]] = []
        self._build_window_list()

        self._current_obs: Optional[np.ndarray] = None
        self._current_candles_5m = None
        self._current_symbol: Optional[str] = None

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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if not self._windows:
            self._current_obs = np.zeros(27, dtype=np.float32)
            return self._current_obs, {}

        # Pick random window
        idx = self.np_random.integers(0, len(self._windows))
        sym, start, end = self._windows[idx]
        self._current_symbol = sym

        # Get 1m candles for features
        candles_1m = self._candle_store.get_candles(sym, start, end, resample="1m")

        # Get BTC candles for correlation features
        btc_1m = None
        if sym != "BTC-EUR":
            btc_1m = self._candle_store.get_candles("BTC-EUR", start, end, resample="1m")

        # Compute features
        self._current_obs = extract_features(candles_1m, btc_1m)

        # Get 5m candles for backtesting
        self._current_candles_5m = self._candle_store.get_candles(sym, start, end, resample="5m")

        return self._current_obs, {}

    def step(self, action):
        params = action_to_params(action, self._strategy)

        # Run backtest
        try:
            if self._current_candles_5m is None or len(self._current_candles_5m) < 100:
                result = BacktestResult()
            else:
                candle_dicts = []
                for ts, row in self._current_candles_5m.iterrows():
                    candle_dicts.append({
                        "timestamp": str(ts),
                        "open": row["open"], "high": row["high"],
                        "low": row["low"], "close": row["close"],
                        "volume": row["volume"],
                        "symbol": self._current_symbol or "TEST-EUR",
                    })
                engine = BacktestEngine(
                    candle_dicts,
                    strategy_params=params,
                    target_strategy=self._strategy,
                )
                result = engine.run()
        except Exception as e:
            logger.warning("Backtest failed in env: %s", e)
            result = BacktestResult()

        reward = self._compute_reward(result)
        return self._current_obs, reward, True, False, {"result": result}

    def _compute_reward(self, result: BacktestResult) -> float:
        pnl_norm = max(min(result.total_pnl / 1000.0, 3.0), -3.0)
        sharpe_clipped = max(min(result.sharpe_ratio, 3.0), -3.0)
        pf = result.profit_factor
        pf_clipped = min(pf, 5.0) if not (math.isinf(pf) or math.isnan(pf)) else 5.0
        ppf_clipped = max(min(result.profit_per_fee, 5.0), -5.0)

        base = (sharpe_clipped * 0.30
                + pnl_norm * 0.30
                + pf_clipped * 0.20
                + ppf_clipped * 0.20)

        penalty = 0.0
        if result.max_drawdown_pct > 20.0:
            penalty -= 5.0
        if result.total_trades < 3:
            penalty -= 3.0
        if result.total_trades > 0 and result.win_rate < 20.0:
            penalty -= 2.0

        return base + penalty
