"""GPU-accelerated vectorized environment for PPO training with stable-baselines3.

Implements the ``VecEnv`` interface with batched CUDA backtests: all N envs
run their backtest in a single GPU kernel launch, eliminating the sequential
overhead of ``DummyVecEnv``.

Key optimisations
-----------------
* **Pre-computed feature bank** -- all 27 market features for every valid
  (symbol, segment) window are computed once during ``__init__``.  Zero runtime
  feature extraction cost.
* **Batched GPU launch** -- one ``launch_backtest`` call per ``step()``,
  regardless of ``n_envs``.
* **Vectorised reward computation** -- pure numpy, no Python loops.
* **Minimal CPU-GPU transfer** -- only the small params / seg_info arrays
  go to GPU; only 8 scalar results per env come back.
* **Auto-reset with numpy fancy indexing** -- no Python loop over terminated
  envs.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from bot.learning.feature_extractor import N_FEATURES, extract_features
from bot.learning.gpu_backtest_kernel import (
    STRATEGY_NAME_TO_ID,
    build_batch_inputs,
    compile_kernel,
    launch_backtest,
    prepare_gpu_data,
)
from bot.learning.rl_environment import action_to_params, get_param_table

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_OBS_DIM = 32           # 27 market features + 5 performance context
_PERF_DIM = 5           # cumulative performance features
_BARS_PER_DAY = 288     # 24h * 60min / 5min
_MIN_SEGMENT_BARS = 100 # minimum usable segment length


# ---------------------------------------------------------------------------
# Vectorised reward (numpy, no loops)
# ---------------------------------------------------------------------------

def _batch_compute_reward(
    pnl: np.ndarray,
    sharpe: np.ndarray,
    profit_factor: np.ndarray,
    max_dd: np.ndarray,
    total_trades: np.ndarray,
    signals_generated: np.ndarray,
) -> np.ndarray:
    """Compute reward for a whole batch at once.  Mirrors
    ``TradingParamEnv._compute_reward`` but operates on arrays.
    """
    # Normalise / clip
    pnl_norm = np.clip(pnl / 500.0, -5.0, 5.0)
    sharpe_clipped = np.clip(sharpe, -3.0, 3.0)

    # profit_factor: clamp bad values to 0, good ones to [0, 5]
    pf_safe = np.where(
        np.isfinite(profit_factor),
        np.clip(profit_factor, 0.0, 5.0),
        0.0,
    )

    base = pnl_norm * 0.60 + sharpe_clipped * 0.25 + pf_safe * 0.15

    # Penalties
    penalty = np.zeros_like(pnl, dtype=np.float64)

    # Max drawdown penalty
    penalty = np.where(max_dd > 20.0, penalty - 3.0, penalty)

    # Zero-trade penalties
    zero_trades = total_trades == 0
    zero_signals = signals_generated == 0

    # Case 1: zero trades AND zero signals -> -1.5
    penalty = np.where(zero_trades & zero_signals, penalty - 1.5, penalty)

    # Case 2: zero trades but some signals -> gradient penalty
    has_signals = zero_trades & ~zero_signals
    sig_ratio = np.clip(signals_generated / 20.0, 0.0, 1.0)
    gradient_penalty = -(1.0 - sig_ratio * 0.5)
    penalty = np.where(has_signals, penalty + gradient_penalty, penalty)

    return base + penalty


# ---------------------------------------------------------------------------
# GpuTradingVecEnv
# ---------------------------------------------------------------------------

class GpuTradingVecEnv(VecEnv):
    """GPU-accelerated vectorised RL environment for trading param tuning.

    Parameters
    ----------
    candle_store
        Database-backed candle store (needs ``get_date_range``, ``get_candles``).
    symbols : list[str]
        Tradeable symbols (e.g. ``["BTC-EUR", "ETH-EUR"]``).
    strategy : str
        One of ``"orderflow"``, ``"range"``, ``"squeeze"``,
        ``"funding_contrarian"``.
    n_envs : int
        Number of parallel environments (GPU threads per step).
    n_segments : int
        Number of sub-segments per episode (multi-step).
    window_days : int
        Total episode window in calendar days.
    validation_mode : bool
        If True use the held-out 10 % validation windows.
    gpu_data_days : int
        Days of history to load onto the GPU via ``prepare_gpu_data``.
    """

    metadata: dict = {}  # SB3 compat

    def __init__(
        self,
        candle_store,
        symbols: List[str],
        strategy: str,
        n_envs: int = 512,
        n_segments: int = 3,
        window_days: int = 90,
        validation_mode: bool = False,
        gpu_data_days: int = 0,
        _gpu_data: Optional[Dict] = None,
        _kernel: Optional[Any] = None,
    ) -> None:
        self._candle_store = candle_store
        self._symbols = list(symbols)
        self._strategy = strategy
        self._n_segments = n_segments
        self._window_days = window_days
        self._validation_mode = validation_mode
        self._strategy_id = STRATEGY_NAME_TO_ID[strategy]

        param_table = get_param_table(strategy)
        self._param_table = param_table
        n_actions = len(param_table)

        obs_space = spaces.Box(
            low=-1.0, high=1.0, shape=(_OBS_DIM,), dtype=np.float32,
        )
        act_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_actions,), dtype=np.float32,
        )

        super().__init__(n_envs, obs_space, act_space)

        # -- 1. Build window list (bar-level) --------------------------
        # Each window = (symbol_idx, segment_bars_list, feature_indices_list)
        # We store flat tuples for speed.
        self._train_windows: List[Tuple] = []
        self._val_windows: List[Tuple] = []

        # -- 2. Prepare GPU data (reuse if provided) -------------------
        if _gpu_data is not None and _kernel is not None:
            self._gpu_data = _gpu_data
            self._kernel = _kernel
            self._gpu_data_days = 0
            logger.info("GpuVecEnv: reusing pre-built GPU data and kernel")
        else:
            # Determine total days of data to load onto GPU.
            if gpu_data_days <= 0:
                max_days = 0
                for sym in self._symbols:
                    dr = candle_store.get_date_range(sym)
                    if dr is None:
                        continue
                    min_ts, max_ts = dr
                    span = (max_ts - min_ts).days
                    if span > max_days:
                        max_days = span
                gpu_data_days = max(max_days + 1, window_days + 30)
            self._gpu_data_days = gpu_data_days

            logger.info("GpuVecEnv: preparing GPU data (%d days) ...", gpu_data_days)
            self._gpu_data = prepare_gpu_data(
                candle_store, self._symbols, days=gpu_data_days,
            )
            self._kernel = compile_kernel()

        # Symbol meta shortcut
        self._sym_meta = self._gpu_data["symbol_meta"]
        # Map symbol name -> index in sym_meta
        self._sym_name_to_idx: Dict[str, int] = {
            m["symbol"]: i for i, m in enumerate(self._sym_meta)
        }

        # -- 3. Pre-compute feature bank ------------------------------
        self._features_bank: np.ndarray = np.empty(0, dtype=np.float32)
        # Window descriptors: (sym_meta_idx, seg0_start_bar, seg0_end_bar,
        #                      seg1_start_bar, seg1_end_bar, ...,
        #                      feat_idx_0, feat_idx_1, ...)
        # We flatten this for memory efficiency.
        # Each window entry: n_segments * 2 bar pairs + n_segments feature indices
        self._window_sym_idx: np.ndarray = np.empty(0, dtype=np.int32)
        self._window_seg_bars: np.ndarray = np.empty(0, dtype=np.int32)
        self._window_feat_idx: np.ndarray = np.empty(0, dtype=np.int32)

        self._build_windows_and_features()

        if self._validation_mode:
            self._active_windows = self._val_windows
        else:
            self._active_windows = self._train_windows

        n_win = len(self._active_windows)
        logger.info(
            "GpuVecEnv: %d active windows, %d features in bank, "
            "%d envs, strategy=%s",
            n_win, len(self._features_bank), n_envs, strategy,
        )

        if n_win == 0:
            raise ValueError(
                "No valid windows found for GPU vec env. "
                "Check that candle data spans >= window_days."
            )

        # -- 4. Per-env state ------------------------------------------
        self._segment_idx = np.zeros(n_envs, dtype=np.int32)
        self._cum_pnl = np.zeros(n_envs, dtype=np.float64)
        self._cum_trades = np.zeros(n_envs, dtype=np.int32)
        self._cum_signals = np.zeros(n_envs, dtype=np.int32)
        self._max_dd = np.zeros(n_envs, dtype=np.float64)
        self._current_obs = np.zeros((n_envs, _OBS_DIM), dtype=np.float32)
        self._window_assignments = np.zeros(n_envs, dtype=np.int32)

        # step_async / step_wait staging
        self._pending_actions: Optional[np.ndarray] = None

        # RNG
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Window + feature bank construction
    # ------------------------------------------------------------------

    def _build_windows_and_features(self) -> None:
        """Enumerate all valid (symbol, 90-day window) combos, split into
        ``n_segments`` sub-windows, pre-compute features for each segment,
        and store them in ``_features_bank``.
        """
        step_days = 14
        segment_days = self._window_days // self._n_segments

        all_features: List[np.ndarray] = []
        train_windows: List[Tuple] = []
        val_windows: List[Tuple] = []

        feat_idx_counter = 0

        for sm_idx, sm in enumerate(self._sym_meta):
            symbol = sm["symbol"]
            n_5m = sm["n_5m"]

            # Minimum bars for one full window
            window_bars = self._window_days * _BARS_PER_DAY
            if n_5m < window_bars:
                continue

            # Train / val split (90 % / 10 % by bar count)
            split_bar = int(n_5m * 0.9)

            # Iterate over sliding windows
            start_bar = 0
            step_bars = step_days * _BARS_PER_DAY

            while start_bar + window_bars <= n_5m:
                end_bar = start_bar + window_bars

                # Determine if this window falls in train or val
                # Window midpoint determines assignment
                mid_bar = (start_bar + end_bar) // 2
                is_val = mid_bar >= split_bar

                # Build per-segment boundaries and features
                seg_bar_pairs = []
                feat_indices = []
                valid = True

                for seg in range(self._n_segments):
                    seg_start = start_bar + seg * (segment_days * _BARS_PER_DAY)
                    seg_end = start_bar + (seg + 1) * (segment_days * _BARS_PER_DAY)
                    seg_end = min(seg_end, end_bar)

                    if (seg_end - seg_start) < _MIN_SEGMENT_BARS:
                        valid = False
                        break

                    seg_bar_pairs.append((seg_start, seg_end))

                    # Extract features for this segment using 1m candle data
                    feats = self._extract_segment_features(
                        symbol, sm, seg_start, seg_end,
                    )
                    all_features.append(feats)
                    feat_indices.append(feat_idx_counter)
                    feat_idx_counter += 1

                if not valid:
                    start_bar += step_bars
                    continue

                window_entry = (sm_idx, seg_bar_pairs, feat_indices)
                if is_val:
                    val_windows.append(window_entry)
                else:
                    train_windows.append(window_entry)

                start_bar += step_bars

        # Stack features into a contiguous array
        if all_features:
            self._features_bank = np.stack(all_features, axis=0)
        else:
            self._features_bank = np.zeros((0, N_FEATURES), dtype=np.float32)

        self._train_windows = train_windows
        self._val_windows = val_windows

        logger.info(
            "GpuVecEnv: built %d train windows, %d val windows, "
            "%d features in bank",
            len(train_windows), len(val_windows), len(self._features_bank),
        )

    def _extract_segment_features(
        self,
        symbol: str,
        sm: dict,
        seg_start_bar: int,
        seg_end_bar: int,
    ) -> np.ndarray:
        """Extract 27 market features for a segment using the candle store.

        Uses 1m candles (resampled internally by ``extract_features``).
        Falls back to zeros if data is insufficient.
        """
        # Convert bar indices to approximate timestamps.
        # We need 1m candles for feature extraction, so query the candle store.
        dr = self._candle_store.get_date_range(symbol)
        if dr is None:
            return np.zeros(N_FEATURES, dtype=np.float32)

        min_ts, max_ts = dr
        if not hasattr(min_ts, "tzinfo") or min_ts.tzinfo is None:
            min_ts = min_ts.replace(tzinfo=timezone.utc)
            max_ts = max_ts.replace(tzinfo=timezone.utc)

        # Convert 5m bar offset to timestamp
        seg_start_dt = min_ts + timedelta(minutes=seg_start_bar * 5)
        seg_end_dt = min_ts + timedelta(minutes=seg_end_bar * 5)

        # Clamp to data range
        seg_start_dt = max(seg_start_dt, min_ts)
        seg_end_dt = min(seg_end_dt, max_ts)

        try:
            candles_1m = self._candle_store.get_candles(
                symbol, seg_start_dt, seg_end_dt, resample="1m",
            )
        except Exception:
            return np.zeros(N_FEATURES, dtype=np.float32)

        if candles_1m is None or candles_1m.empty:
            return np.zeros(N_FEATURES, dtype=np.float32)

        # BTC candles for correlation features
        btc_1m = None
        if symbol != "BTC-EUR":
            try:
                btc_1m = self._candle_store.get_candles(
                    "BTC-EUR", seg_start_dt, seg_end_dt, resample="1m",
                )
            except Exception:
                pass

        return extract_features(candles_1m, btc_1m)

    # ------------------------------------------------------------------
    # VecEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset ALL environments and return initial observations."""
        n = self.num_envs
        self._segment_idx[:] = 0
        self._cum_pnl[:] = 0.0
        self._cum_trades[:] = 0
        self._cum_signals[:] = 0
        self._max_dd[:] = 0.0

        # Assign random windows to each env
        self._window_assignments = self._rng.integers(
            0, len(self._active_windows), size=n,
        )

        # Build initial observations from feature bank
        for i in range(n):
            self._current_obs[i] = self._build_obs(i, segment=0)

        return self._current_obs.copy()

    def step_async(self, actions: np.ndarray) -> None:
        """Store actions for processing in ``step_wait``."""
        self._pending_actions = np.asarray(actions, dtype=np.float32)

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """Execute GPU backtest for all envs and return results."""
        actions = self._pending_actions
        assert actions is not None, "step_async must be called before step_wait"
        self._pending_actions = None

        n = self.num_envs

        # -- 1. Map actions to param dicts ----------------------------
        param_dicts: List[Dict[str, Any]] = []
        strategy_ids: List[int] = []
        symbol_indices: List[int] = []
        windows: List[Tuple[int, int]] = []

        for i in range(n):
            pdict = action_to_params(actions[i], self._strategy)

            # Convert hold hours to bars if present
            if "max_hold_hours" in pdict:
                pdict["max_hold_bars"] = pdict["max_hold_hours"] * 12.0
            if "range_max_hold_hours" in pdict:
                pdict["range_max_hold_bars"] = pdict["range_max_hold_hours"] * 12.0

            param_dicts.append(pdict)
            strategy_ids.append(self._strategy_id)

            win_idx = self._window_assignments[i]
            win = self._active_windows[win_idx]
            sm_idx = win[0]
            seg_idx = self._segment_idx[i]
            seg_bars = win[1][seg_idx]  # (start_bar, end_bar)

            symbol_indices.append(sm_idx)
            windows.append(seg_bars)

        # -- 2. Build batch inputs and launch GPU kernel --------------
        params_gpu, seg_info_gpu, batch_size = build_batch_inputs(
            self._gpu_data,
            param_dicts,
            strategy_ids,
            symbol_indices=symbol_indices,
            windows=windows,
        )

        results = launch_backtest(
            self._kernel,
            self._gpu_data,
            params_gpu,
            seg_info_gpu,
            batch_size,
        )

        # -- 3. Extract result arrays ---------------------------------
        pnl = results["pnl"]
        trades = results["trades"].astype(np.int32)
        wins = results["wins"].astype(np.int32)
        signals = results["signals"].astype(np.int32)
        max_dd = results["max_dd"]
        sharpe = results["sharpe"]
        sum_win_pnl = results["sum_win_pnl"]
        sum_loss_pnl = results["sum_loss_pnl"]

        # Profit factor (suppress divide-by-zero for zero-loss cases)
        with np.errstate(divide="ignore", invalid="ignore"):
            profit_factor = np.where(
                np.abs(sum_loss_pnl) > 1e-9,
                sum_win_pnl / np.abs(sum_loss_pnl),
                np.where(sum_win_pnl > 0, 5.0, 0.0),
            )

        # -- 4. Compute vectorised rewards ----------------------------
        rewards = _batch_compute_reward(
            pnl, sharpe, profit_factor, max_dd,
            trades.astype(np.float64), signals.astype(np.float64),
        ).astype(np.float32)

        # -- 5. Update cumulative state -------------------------------
        self._cum_pnl += pnl
        self._cum_trades += trades
        self._cum_signals += signals
        self._max_dd = np.maximum(self._max_dd, max_dd)

        # Advance segment counters
        self._segment_idx += 1

        # -- 6. Determine termination ---------------------------------
        dones = self._segment_idx >= self._n_segments

        # -- 7. Build infos -------------------------------------------
        infos: List[dict] = []
        for i in range(n):
            info: Dict[str, Any] = {
                "segment_pnl": float(pnl[i]),
                "segment_trades": int(trades[i]),
                "segment_signals": int(signals[i]),
                "segment_sharpe": float(sharpe[i]),
                "segment_max_dd": float(max_dd[i]),
                "segment_profit_factor": float(profit_factor[i]),
                "segment_idx": int(self._segment_idx[i]) - 1,
            }
            if dones[i]:
                # Terminal info: episode summary
                info["terminal_observation"] = self._current_obs[i].copy()
                info["episode"] = {
                    "r": float(rewards[i]),
                    "l": int(self._segment_idx[i]),
                    "cum_pnl": float(self._cum_pnl[i]),
                    "cum_trades": int(self._cum_trades[i]),
                    "cum_signals": int(self._cum_signals[i]),
                    "max_dd": float(self._max_dd[i]),
                }
            infos.append(info)

        # -- 8. Build next observations -------------------------------
        # For non-terminated envs: advance to next segment observation
        continuing = ~dones
        cont_indices = np.where(continuing)[0]
        for i in cont_indices:
            self._current_obs[i] = self._build_obs(
                i, segment=self._segment_idx[i],
            )

        # -- 9. Auto-reset terminated envs ----------------------------
        done_indices = np.where(dones)[0]
        if len(done_indices) > 0:
            self._auto_reset(done_indices)

        return (
            self._current_obs.copy(),
            rewards,
            dones.astype(np.bool_),
            infos,
        )

    def close(self) -> None:
        """Release GPU resources."""
        self._gpu_data = None
        self._kernel = None
        self._features_bank = None

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Seed the RNG."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            return [seed]
        s = int(self._rng.integers(0, 2**31))
        self._rng = np.random.default_rng(s)
        return [s]

    def env_is_wrapped(self, wrapper_class, indices=None):
        """Check if envs are wrapped -- they are not."""
        if indices is None:
            return [False] * self.num_envs
        return [False] * len(indices)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call a method on sub-environments (not applicable for GPU vec env)."""
        raise NotImplementedError(
            "GpuTradingVecEnv does not support env_method"
        )

    def get_attr(self, attr_name, indices=None):
        """Get attribute from sub-environments."""
        if indices is None:
            indices = range(self.num_envs)
        return [getattr(self, attr_name, None) for _ in indices]

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute on sub-environments (no-op for GPU vec env)."""
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_obs(self, env_idx: int, segment: int) -> np.ndarray:
        """Build the 32-dim observation for *env_idx* at the given segment.

        First 27 dims: market features from the pre-computed bank.
        Last 5 dims: normalised cumulative performance context.
        """
        obs = np.zeros(_OBS_DIM, dtype=np.float32)

        win_idx = self._window_assignments[env_idx]
        win = self._active_windows[win_idx]
        feat_indices = win[2]  # list of feature bank indices

        if segment < len(feat_indices):
            fidx = feat_indices[segment]
            if fidx < len(self._features_bank):
                obs[:N_FEATURES] = self._features_bank[fidx]

        # Performance context (zeros on first segment)
        if segment > 0:
            obs[N_FEATURES] = float(np.clip(
                self._cum_pnl[env_idx] / 1000.0, -1.0, 1.0,
            ))
            obs[N_FEATURES + 1] = float(np.clip(
                self._cum_trades[env_idx] / 50.0, 0.0, 1.0,
            ))
            obs[N_FEATURES + 2] = float(np.clip(
                self._cum_signals[env_idx] / 100.0, 0.0, 1.0,
            ))
            obs[N_FEATURES + 3] = float(np.clip(
                self._max_dd[env_idx] / 30.0, 0.0, 1.0,
            ))
            obs[N_FEATURES + 4] = float(segment / self._n_segments)

        return obs

    def _auto_reset(self, indices: np.ndarray) -> None:
        """Auto-reset terminated envs to new random windows.

        Uses numpy indexing to reset all terminated envs at once.
        """
        n_reset = len(indices)

        # Reset cumulative state
        self._segment_idx[indices] = 0
        self._cum_pnl[indices] = 0.0
        self._cum_trades[indices] = 0
        self._cum_signals[indices] = 0
        self._max_dd[indices] = 0.0

        # Assign new random windows
        new_windows = self._rng.integers(
            0, len(self._active_windows), size=n_reset,
        )
        self._window_assignments[indices] = new_windows

        # Build initial observations for reset envs
        for i in indices:
            self._current_obs[i] = self._build_obs(i, segment=0)

    # ------------------------------------------------------------------
    # SB3 compatibility extras
    # ------------------------------------------------------------------

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        """Not supported -- no rendering."""
        raise NotImplementedError

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Not supported."""
        return None

    @staticmethod
    def _compute_reward_single(
        pnl: float,
        sharpe: float,
        profit_factor: float,
        max_dd: float,
        total_trades: int,
        signals_generated: int,
    ) -> float:
        """Scalar version of the reward function (for debugging / tests)."""
        pnl_norm = max(min(pnl / 500.0, 5.0), -5.0)
        sharpe_clipped = max(min(sharpe, 3.0), -3.0)
        if math.isinf(profit_factor) or math.isnan(profit_factor):
            pf_clipped = 0.0
        else:
            pf_clipped = min(profit_factor, 5.0)

        base = pnl_norm * 0.60 + sharpe_clipped * 0.25 + pf_clipped * 0.15

        penalty = 0.0
        if max_dd > 20.0:
            penalty -= 3.0
        if total_trades == 0:
            if signals_generated == 0:
                penalty -= 1.5
            else:
                sig_ratio = min(signals_generated / 20.0, 1.0)
                penalty -= 1.0 - sig_ratio * 0.5

        return base + penalty
