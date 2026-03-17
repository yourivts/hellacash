"""GPU-accelerated backtest kernel -- CUDA C via CuPy RawKernel.

Runs N_BATCH independent backtests simultaneously on GPU, one per CUDA
thread.  Each thread iterates sequentially over 5-minute bars, managing
up to MAX_POSITIONS open positions with full exit/entry logic that
mirrors the CPU Numba kernel in ``bot.backtest.numba_loop``.

Usage::

    from bot.learning.gpu_backtest_kernel import compile_kernel, prepare_gpu_data

    kernel = compile_kernel()
    gpu = prepare_gpu_data(candle_store, symbols=["BTC-EUR"])
    # ... launch kernel with params / seg_info ...
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Signal array column indices  (must match packed layout in CUDA kernel)
# ---------------------------------------------------------------------------

# 1h signals: 13 columns per bar (row-major: bar * 13 + col)
SIG_RSI = 0
SIG_MACD_HIST = 1
SIG_BB_LOWER = 2
SIG_BB_UPPER = 3
SIG_BB_MID = 4
SIG_BB_BW = 5
SIG_VSR = 6
SIG_CMF = 7
SIG_OBV = 8
SIG_OPEN = 9
SIG_HIGH = 10
SIG_LOW = 11
SIG_CLOSE = 12
N_SIG_1H = 13

# 4h signals: 3 columns per bar
SIG4_ADX = 0
SIG4_EMA50 = 1
SIG4_RSI = 2
N_SIG_4H = 3

# 1d signals: 2 columns per bar
SIG1D_EMA200 = 0
SIG1D_EMA50 = 1
N_SIG_1D = 2

# Regime 4h: 3 columns per bar
REG_ADX = 0
REG_ATR = 1
REG_CLOSE = 2
N_REG_4H = 3

# Regime 1h: 3 columns per bar
REG1H_ATR = 0
REG1H_CLOSE = 1
REG1H_ATR_SERIES = 2
N_REG_1H = 3

# Parameter indices (into params[tid * 22 + idx])
P_ATR_MULT = 0
P_RR_RATIO = 1
P_BASE_RISK_PCT = 2
P_SIGNAL_MIN = 3
P_TF_W_1H = 4
P_TF_W_4H = 5
P_TF_W_1D = 6
P_CONF_SCALE = 7
P_EMA200_FILTER = 8
P_VOL_ATR_THRESH = 9
P_CONSEC_CONFIRMS = 10
P_CONFLUENCE = 11
P_DD_SCALE = 12
P_MAX_POS_PCT = 13
P_TRAIL_ACT = 14
P_MAX_HOLD_BARS = 15
P_RANGE_HOLD_BARS = 16
P_MIN_ATR_PCT = 17
P_QUIET_ATR = 18
P_REGIME_ADX = 19
P_RANGING_ADX = 20
P_MIN_PROFIT_MULT = 21
N_PARAMS = 22

# Strategy IDs
STRAT_ORDERFLOW = 0
STRAT_RANGE = 1
STRAT_SQUEEZE = 2
STRAT_FUNDING = 3
STRAT_BREAKOUT = 4
STRAT_TREND = 5
STRAT_MOMENTUM = 6
STRAT_MEANREV = 7

STRATEGY_NAME_TO_ID = {
    "orderflow": STRAT_ORDERFLOW,
    "range": STRAT_RANGE,
    "squeeze": STRAT_SQUEEZE,
    "funding_contrarian": STRAT_FUNDING,
    "breakout": STRAT_BREAKOUT,
    "trend_following": STRAT_TREND,
    "momentum": STRAT_MOMENTUM,
    "mean_reversion": STRAT_MEANREV,
}

# ---------------------------------------------------------------------------
# CUDA C kernel source
# ---------------------------------------------------------------------------

BACKTEST_KERNEL_SRC = r"""
extern "C" __global__
void backtest_kernel(
    /* -- Candle data (concatenated across symbols) ---------------- */
    const float*  __restrict__ closes,       // [total_5m_bars]
    const float*  __restrict__ highs,        // [total_5m_bars]
    const float*  __restrict__ lows,         // [total_5m_bars]
    const int     total_5m_bars,

    /* -- Timeframe maps (concatenated) --------------------------- */
    const int*    __restrict__ h1_map,       // [total_5m_bars]
    const int*    __restrict__ h4_map,       // [total_5m_bars]
    const int*    __restrict__ h1d_map,      // [total_5m_bars]

    /* -- Signal arrays (packed row-major) ------------------------ */
    const float*  __restrict__ sig_1h,       // [total_1h * 13]
    const int     total_1h,
    const float*  __restrict__ sig_4h,       // [total_4h * 3]
    const int     total_4h,
    const float*  __restrict__ sig_1d,       // [total_1d * 2]
    const int     total_1d,
    const float*  __restrict__ sig_5m,       // [total_5m_bars]  (RSI only)

    /* -- Regime arrays (packed row-major) ------------------------ */
    const float*  __restrict__ regime_data,  // [total_4h * 3]  (adx, atr, close)
    const float*  __restrict__ regime_1h,    // [total_1h * 3]  (atr, close, atr_series)

    /* -- Per-batch inputs ---------------------------------------- */
    const double* __restrict__ params,       // [N_BATCH * 22]
    const int*    __restrict__ seg_info,     // [N_BATCH * 5]
    const int     batch_size,

    /* -- Per-batch outputs --------------------------------------- */
    double* __restrict__ out_pnl,            // [N_BATCH]
    int*    __restrict__ out_trades,         // [N_BATCH]
    int*    __restrict__ out_wins,           // [N_BATCH]
    int*    __restrict__ out_signals,        // [N_BATCH]
    double* __restrict__ out_max_dd,         // [N_BATCH]
    double* __restrict__ out_sharpe,         // [N_BATCH]
    double* __restrict__ out_sum_win_pnl,    // [N_BATCH]
    double* __restrict__ out_sum_loss_pnl    // [N_BATCH]
)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= batch_size) return;

    /* ================================================================
     * Constants
     * ================================================================ */
    #define MAKER_FEE_PCT  0.0015
    #define TAKER_FEE_PCT  0.0025
    #define MAX_POSITIONS  10
    #define DIR_LONG       1
    #define DIR_SHORT     -1
    #define DIR_NEUTRAL    0
    #define STRAT_ORDERFLOW 0
    #define STRAT_RANGE     1
    #define STRAT_SQUEEZE   2
    #define STRAT_FUNDING   3
    #define WARMUP_BARS    60
    #define SIGNAL_EVERY   12
    #define INITIAL_CAPITAL 10000.0
    #define SLIPPAGE_PCT   0.0005

    /* Parameter indices */
    #define P_ATR_MULT         0
    #define P_RR_RATIO         1
    #define P_BASE_RISK_PCT    2
    #define P_SIGNAL_MIN       3
    #define P_TF_W_1H          4
    #define P_TF_W_4H          5
    #define P_TF_W_1D          6
    #define P_CONF_SCALE       7
    #define P_EMA200_FILTER    8
    #define P_VOL_ATR_THRESH   9
    #define P_CONSEC_CONFIRMS 10
    #define P_CONFLUENCE      11
    #define P_DD_SCALE        12
    #define P_MAX_POS_PCT     13
    #define P_TRAIL_ACT       14
    #define P_MAX_HOLD_BARS   15
    #define P_RANGE_HOLD_BARS 16
    #define P_MIN_ATR_PCT     17
    #define P_QUIET_ATR       18
    #define P_REGIME_ADX      19
    #define P_RANGING_ADX     20
    #define P_MIN_PROFIT_MULT 21

    /* Signal column indices */
    #define SIG_RSI       0
    #define SIG_MACD_HIST 1
    #define SIG_BB_LOWER  2
    #define SIG_BB_UPPER  3
    #define SIG_BB_MID    4
    #define SIG_BB_BW     5
    #define SIG_VSR       6
    #define SIG_CMF       7
    #define SIG_OBV       8
    #define SIG_OPEN      9
    #define SIG_HIGH     10
    #define SIG_LOW      11
    #define SIG_CLOSE    12
    #define N_SIG_1H     13

    #define SIG4_ADX      0
    #define SIG4_EMA50    1
    #define SIG4_RSI      2
    #define N_SIG_4H      3

    #define SIG1D_EMA200  0
    #define SIG1D_EMA50   1

    #define REG_ADX       0
    #define REG_ATR       1
    #define REG_CLOSE     2

    #define REG1H_ATR        0
    #define REG1H_CLOSE      1
    #define REG1H_ATR_SERIES 2

    /* Regime encoding */
    #define REGIME_QUIET     0
    #define REGIME_VOLATILE  1
    #define REGIME_TRENDING  2
    #define REGIME_RANGING   3
    #define REGIME_NEUTRAL   4

    /* ================================================================
     * Load per-thread segment info and parameters
     * ================================================================ */
    int si_base   = tid * 5;
    int start_5m  = __ldg(&seg_info[si_base + 0]);
    int end_5m    = __ldg(&seg_info[si_base + 1]);
    int offset_5m = __ldg(&seg_info[si_base + 2]);
    int strategy_id = __ldg(&seg_info[si_base + 3]);
    int n_bars    = __ldg(&seg_info[si_base + 4]);

    /* Bounds check */
    if (start_5m < 0 || end_5m > total_5m_bars || start_5m >= end_5m) {
        out_pnl[tid] = 0.0;
        out_trades[tid] = 0;
        out_wins[tid] = 0;
        out_signals[tid] = 0;
        out_max_dd[tid] = 0.0;
        out_sharpe[tid] = 0.0;
        out_sum_win_pnl[tid] = 0.0;
        out_sum_loss_pnl[tid] = 0.0;
        return;
    }

    int p_base = tid * 22;
    double atr_multiplier       = params[p_base + P_ATR_MULT];
    double rr_ratio             = params[p_base + P_RR_RATIO];
    double base_risk_pct        = params[p_base + P_BASE_RISK_PCT];
    double signal_strength_min  = params[p_base + P_SIGNAL_MIN];
    double tf_weight_1h         = params[p_base + P_TF_W_1H];
    double tf_weight_4h         = params[p_base + P_TF_W_4H];
    double tf_weight_1d         = params[p_base + P_TF_W_1D];
    double confidence_size_scaling = params[p_base + P_CONF_SCALE];
    double ema200_filter_pct    = params[p_base + P_EMA200_FILTER];
    double volatile_atr_thresh  = params[p_base + P_VOL_ATR_THRESH];
    int    consecutive_confirms = (int)params[p_base + P_CONSEC_CONFIRMS];
    double confluence_boost     = params[p_base + P_CONFLUENCE];
    double drawdown_scale_pct   = params[p_base + P_DD_SCALE];
    double max_position_pct     = params[p_base + P_MAX_POS_PCT];
    double trail_activation_mult = params[p_base + P_TRAIL_ACT];
    int    max_hold_bars        = (int)params[p_base + P_MAX_HOLD_BARS];
    int    range_max_hold_bars  = (int)params[p_base + P_RANGE_HOLD_BARS];
    double min_atr_pct          = params[p_base + P_MIN_ATR_PCT];
    double quiet_atr_threshold  = params[p_base + P_QUIET_ATR];
    double regime_adx_threshold = params[p_base + P_REGIME_ADX];
    double ranging_adx_threshold = params[p_base + P_RANGING_ADX];
    double min_profit_multiple  = params[p_base + P_MIN_PROFIT_MULT];

    /* ================================================================
     * Position state (thread-local arrays)
     * ================================================================ */
    bool   pos_active[MAX_POSITIONS];
    int    pos_dir[MAX_POSITIONS];        /* DIR_LONG or DIR_SHORT */
    double pos_entry[MAX_POSITIONS];
    double pos_size[MAX_POSITIONS];
    double pos_sl[MAX_POSITIONS];
    double pos_tp[MAX_POSITIONS];
    double pos_peak[MAX_POSITIONS];       /* highest (long) / lowest (short) */
    int    pos_entry_bar[MAX_POSITIONS];
    double pos_entry_fee[MAX_POSITIONS];
    int    pos_strategy[MAX_POSITIONS];
    bool   pos_tp_shifted[MAX_POSITIONS];
    double pos_range_mid[MAX_POSITIONS];
    double pos_range_upper[MAX_POSITIONS];
    double pos_range_lower[MAX_POSITIONS];

    for (int j = 0; j < MAX_POSITIONS; j++) {
        pos_active[j] = false;
        pos_dir[j] = 0;
        pos_entry[j] = 0.0;
        pos_size[j] = 0.0;
        pos_sl[j] = 0.0;
        pos_tp[j] = 0.0;
        pos_peak[j] = 0.0;
        pos_entry_bar[j] = 0;
        pos_entry_fee[j] = 0.0;
        pos_strategy[j] = 0;
        pos_tp_shifted[j] = false;
        pos_range_mid[j] = 0.0;
        pos_range_upper[j] = 0.0;
        pos_range_lower[j] = 0.0;
    }

    /* ================================================================
     * Running state
     * ================================================================ */
    double balance       = INITIAL_CAPITAL;
    double peak_balance  = INITIAL_CAPITAL;
    double total_fees    = 0.0;
    double total_pnl_out = 0.0;
    double sum_win_pnl   = 0.0;
    double sum_loss_pnl  = 0.0;
    int    total_trades  = 0;
    int    winning_trades = 0;
    int    signals_gen   = 0;
    double max_drawdown  = 0.0;

    /* Consecutive confirmation state */
    int prev_signal_dir = DIR_NEUTRAL;
    int signal_streak   = 0;
    int last_trade_close_bar = -999;

    /* ATR% history for position sizing (circular buffer) */
    double atr_history[200];
    int    atr_hist_len = 0;
    int    atr_hist_idx = 0;
    for (int j = 0; j < 200; j++) atr_history[j] = 0.0;

    /* Welford's online algorithm for Sharpe ratio */
    double welf_n    = 0.0;
    double welf_mean = 0.0;
    double welf_m2   = 0.0;
    double prev_equity = INITIAL_CAPITAL;

    /* ================================================================
     * Helper: inline close_position
     * ================================================================ */
    #define CLOSE_POSITION(j_slot, exit_px, reason_is_tp) do { \
        int _d = pos_dir[j_slot]; \
        double _exit_price = (exit_px); \
        double _slipped, _pchange; \
        if (_d == DIR_SHORT) { \
            _slipped = _exit_price * (1.0 + SLIPPAGE_PCT); \
            _pchange = (pos_entry[j_slot] - _slipped) / pos_entry[j_slot]; \
        } else { \
            _slipped = _exit_price * (1.0 - SLIPPAGE_PCT); \
            _pchange = (_slipped - pos_entry[j_slot]) / pos_entry[j_slot]; \
        } \
        double _gross = pos_size[j_slot] * _pchange; \
        double _qty = pos_size[j_slot] / pos_entry[j_slot]; \
        double _fee_rate = (reason_is_tp) ? MAKER_FEE_PCT : TAKER_FEE_PCT; \
        double _exit_fee = _slipped * _qty * _fee_rate; \
        total_fees += _exit_fee; \
        double _pnl = _gross - _exit_fee - pos_entry_fee[j_slot]; \
        balance += pos_size[j_slot] + _pnl; \
        total_pnl_out += _pnl; \
        total_trades++; \
        if (_pnl > 0.0) { \
            winning_trades++; \
            sum_win_pnl += _pnl; \
        } else { \
            sum_loss_pnl += fabs(_pnl); \
        } \
        pos_active[j_slot] = false; \
        last_trade_close_bar = _local_i; \
    } while(0)

    /* ================================================================
     * MAIN BAR LOOP
     * ================================================================ */
    int local_n = end_5m - start_5m;  /* number of 5m bars in this segment */

    for (int _local_i = WARMUP_BARS; _local_i < local_n; _local_i++) {
        int abs_i = start_5m + _local_i;  /* absolute index into candle arrays */

        /* Bounds check on candle arrays */
        if (abs_i < 0 || abs_i >= total_5m_bars) continue;

        double price = (double)__ldg(&closes[abs_i]);
        double hi    = (double)__ldg(&highs[abs_i]);
        double lo    = (double)__ldg(&lows[abs_i]);

        /* Map to timeframe indices */
        int h1_idx  = __ldg(&h1_map[abs_i]);
        int h4_idx  = __ldg(&h4_map[abs_i]);
        int h1d_idx = __ldg(&h1d_map[abs_i]);

        /* Count active positions */
        int num_active = 0;
        for (int j = 0; j < MAX_POSITIONS; j++) {
            if (pos_active[j]) num_active++;
        }

        /* ==============================================================
         * EXIT CHECKING (every bar)
         * ============================================================== */
        if (num_active > 0) {
            double atr_val = 0.0;
            if (h1_idx >= 0 && h1_idx < total_1h) {
                atr_val = (double)__ldg(&regime_1h[h1_idx * 3 + REG1H_ATR_SERIES]);
            }

            for (int j = 0; j < MAX_POSITIONS; j++) {
                if (!pos_active[j]) continue;
                int d = pos_dir[j];

                /* -- Time-based exit ------------------------------- */
                int hold_limit = (pos_strategy[j] == STRAT_RANGE)
                                 ? range_max_hold_bars : max_hold_bars;
                if (_local_i - pos_entry_bar[j] >= hold_limit) {
                    CLOSE_POSITION(j, price, false);
                    continue;
                }

                /* -- Range dynamic TP ------------------------------ */
                if (pos_strategy[j] == STRAT_RANGE && !pos_tp_shifted[j]) {
                    bool crossed = false;
                    if (d == DIR_LONG  && price >= pos_range_mid[j]) crossed = true;
                    if (d == DIR_SHORT && price <= pos_range_mid[j]) crossed = true;
                    int sig_abs_i = abs_i;  /* absolute index for 5m RSI */
                    if (crossed && sig_abs_i >= 3) {
                        double cur_rsi = (sig_abs_i < total_5m_bars)
                            ? (double)__ldg(&sig_5m[sig_abs_i]) : 50.0;
                        double prev_rsi = ((sig_abs_i - 3) >= 0 && (sig_abs_i - 3) < total_5m_bars)
                            ? (double)__ldg(&sig_5m[sig_abs_i - 3]) : cur_rsi;

                        bool should_shift = false;
                        if (d == DIR_LONG  && cur_rsi > prev_rsi) {
                            should_shift = true;
                            pos_tp[j] = pos_range_upper[j];
                        } else if (d == DIR_SHORT && cur_rsi < prev_rsi) {
                            should_shift = true;
                            pos_tp[j] = pos_range_lower[j];
                        }
                        if (should_shift) {
                            pos_tp_shifted[j] = true;
                        } else {
                            /* Close at mid-band (maker fee) */
                            CLOSE_POSITION(j, pos_range_mid[j], true);
                            continue;
                        }
                    }
                }

                /* -- Update trailing price ------------------------- */
                if (d == DIR_SHORT) {
                    if (price < pos_peak[j]) pos_peak[j] = price;
                } else {
                    if (price > pos_peak[j]) pos_peak[j] = price;
                }

                /* -- Trailing stop logic --------------------------- */
                if (pos_strategy[j] == STRAT_RANGE && pos_tp_shifted[j]) {
                    /* Tight 1% trail for range after TP shift */
                    if (d == DIR_LONG) {
                        double trail_level = pos_peak[j] * 0.99;
                        if (trail_level > pos_sl[j]) pos_sl[j] = trail_level;
                    } else {
                        double trail_level = pos_peak[j] * 1.01;
                        if (trail_level < pos_sl[j]) pos_sl[j] = trail_level;
                    }
                } else if (atr_val > 0.0) {
                    /* ATR trailing stop with activation threshold */
                    double trail_dist = atr_val * atr_multiplier;
                    double entry_p = pos_entry[j];
                    double cur_stop = pos_sl[j];
                    if (entry_p > 0.0 && trail_activation_mult > 0.0) {
                        double stop_dist = fabs(entry_p - cur_stop);
                        double profit;
                        if (d == DIR_LONG) {
                            profit = pos_peak[j] - entry_p;
                        } else {
                            profit = entry_p - pos_peak[j];
                        }
                        if (stop_dist > 0.0 && profit < stop_dist * trail_activation_mult) {
                            /* Don't trail yet -- profit hasn't reached activation */
                        } else {
                            if (d == DIR_LONG) {
                                double new_stop = pos_peak[j] - trail_dist;
                                if (new_stop > cur_stop) pos_sl[j] = new_stop;
                            } else {
                                double new_stop = pos_peak[j] + trail_dist;
                                if (cur_stop > 0.0 && new_stop < cur_stop)
                                    pos_sl[j] = new_stop;
                                else if (cur_stop <= 0.0)
                                    pos_sl[j] = new_stop;
                            }
                        }
                    } else {
                        if (d == DIR_LONG) {
                            double new_stop = pos_peak[j] - trail_dist;
                            if (new_stop > cur_stop) pos_sl[j] = new_stop;
                        } else {
                            double new_stop = pos_peak[j] + trail_dist;
                            if (cur_stop > 0.0 && new_stop < cur_stop)
                                pos_sl[j] = new_stop;
                            else if (cur_stop <= 0.0)
                                pos_sl[j] = new_stop;
                        }
                    }
                }

                /* -- SL / TP trigger ------------------------------- */
                int triggered = -1;
                double trigger_price = price;
                if (d == DIR_SHORT) {
                    if (hi >= pos_sl[j]) {
                        triggered = 0;  /* stop loss */
                        trigger_price = pos_sl[j];
                    } else if (lo <= pos_tp[j]) {
                        triggered = 1;  /* take profit */
                        trigger_price = pos_tp[j];
                    }
                } else {
                    if (lo <= pos_sl[j]) {
                        triggered = 0;
                        trigger_price = pos_sl[j];
                    } else if (hi >= pos_tp[j]) {
                        triggered = 1;
                        trigger_price = pos_tp[j];
                    }
                }

                if (triggered >= 0) {
                    bool is_tp = (triggered == 1);
                    CLOSE_POSITION(j, trigger_price, is_tp);
                }
            } /* end for j (exit checks) */
        } /* end if num_active > 0 */

        /* -- Update peak equity ---------------------------------- */
        double current_equity = balance;
        for (int j = 0; j < MAX_POSITIONS; j++) {
            if (pos_active[j]) {
                if (pos_dir[j] == DIR_SHORT) {
                    double change = (pos_entry[j] - price) / pos_entry[j];
                    current_equity += pos_size[j] * (1.0 + change);
                } else {
                    current_equity += pos_size[j] * (price / pos_entry[j]);
                }
            }
        }
        if (current_equity > peak_balance)
            peak_balance = current_equity;

        /* -- Max drawdown tracking ------------------------------- */
        if (peak_balance > 0.0) {
            double dd_pct = (peak_balance - current_equity) / peak_balance * 100.0;
            if (dd_pct > max_drawdown) max_drawdown = dd_pct;
        }

        /* -- Welford's online Sharpe ----------------------------- */
        if (prev_equity > 0.0) {
            double ret = (current_equity / prev_equity) - 1.0;
            welf_n += 1.0;
            double delta = ret - welf_mean;
            welf_mean += delta / welf_n;
            double delta2 = ret - welf_mean;
            welf_m2 += delta * delta2;
        }
        prev_equity = current_equity;

        /* ==============================================================
         * SIGNAL GENERATION (every SIGNAL_EVERY bars after warmup)
         * ============================================================== */
        if ((_local_i - WARMUP_BARS) % SIGNAL_EVERY != 0) continue;

        /* h1_idx warmup check */
        if (h1_idx < 30) continue;

        /* -- Regime detection ------------------------------------ */
        double atr_val_4h = 0.0, price_4h = price, adx_4h = 20.0;
        if (h4_idx >= 0 && h4_idx < total_4h) {
            adx_4h   = (double)__ldg(&regime_data[h4_idx * 3 + REG_ADX]);
            atr_val_4h = (double)__ldg(&regime_data[h4_idx * 3 + REG_ATR]);
            price_4h = (double)__ldg(&regime_data[h4_idx * 3 + REG_CLOSE]);
        }
        double atr_pct_4h = (price_4h > 0.0) ? (atr_val_4h / price_4h * 100.0) : 0.0;

        int regime = REGIME_NEUTRAL;
        if (atr_pct_4h < quiet_atr_threshold)
            regime = REGIME_QUIET;
        else if (atr_pct_4h > volatile_atr_thresh)
            regime = REGIME_VOLATILE;
        else if (adx_4h > regime_adx_threshold)
            regime = REGIME_TRENDING;
        else if (adx_4h < ranging_adx_threshold)
            regime = REGIME_RANGING;

        if (regime == REGIME_QUIET) continue;

        /* 1h ATR% for position sizing */
        double atr_val_1h = 0.0, price_1h = price;
        if (h1_idx >= 0 && h1_idx < total_1h) {
            atr_val_1h = (double)__ldg(&regime_1h[h1_idx * 3 + REG1H_ATR]);
            price_1h   = (double)__ldg(&regime_1h[h1_idx * 3 + REG1H_CLOSE]);
        }
        double atr_pct = (price_1h > 0.0) ? (atr_val_1h / price_1h * 100.0) : 0.0;

        if (atr_pct < min_atr_pct) continue;

        /* ==============================================================
         * STRATEGY EVALUATION
         * ============================================================== */
        int    sig_dir = DIR_NEUTRAL;
        double sig_strength = 0.0;

        if (strategy_id == STRAT_ORDERFLOW) {
            if (h1_idx >= 2 && h1_idx < total_1h) {
                double o = (double)__ldg(&sig_1h[h1_idx * N_SIG_1H + SIG_OPEN]);
                double h = (double)__ldg(&sig_1h[h1_idx * N_SIG_1H + SIG_HIGH]);
                double l = (double)__ldg(&sig_1h[h1_idx * N_SIG_1H + SIG_LOW]);
                double c = (double)__ldg(&sig_1h[h1_idx * N_SIG_1H + SIG_CLOSE]);
                double candle_range = h - l;
                if (candle_range > 0.0) {
                    double body_ratio = fabs(c - o) / candle_range;
                    double wick_lower = (fmin(o, c) - l) / candle_range;
                    double wick_upper = (h - fmax(o, c)) / candle_range;
                    double vol_surge = (h1_idx < total_1h)
                        ? (double)__ldg(&sig_1h[h1_idx * N_SIG_1H + SIG_VSR]) : 1.0;
                    double rsi_1h = (h1_idx < total_1h)
                        ? (double)__ldg(&sig_1h[h1_idx * N_SIG_1H + SIG_RSI]) : 50.0;
                    double cmf_val = (h1_idx < total_1h)
                        ? (double)__ldg(&sig_1h[h1_idx * N_SIG_1H + SIG_CMF]) : 0.0;
                    double obv_now = (h1_idx < total_1h)
                        ? (double)__ldg(&sig_1h[h1_idx * N_SIG_1H + SIG_OBV]) : 0.0;
                    double obv_prev = (h1_idx >= 12 && (h1_idx - 12) < total_1h)
                        ? (double)__ldg(&sig_1h[(h1_idx - 12) * N_SIG_1H + SIG_OBV]) : obv_now;
                    double price_prev_of = (h1_idx >= 12 && (h1_idx - 12) < total_1h)
                        ? (double)__ldg(&sig_1h[(h1_idx - 12) * N_SIG_1H + SIG_CLOSE]) : price;

                    double obv_div = 0.0;
                    if (price < price_prev_of && obv_now > obv_prev) obv_div = 1.0;
                    else if (price > price_prev_of && obv_now < obv_prev) obv_div = -1.0;

                    if (body_ratio <= 0.4 && vol_surge >= 1.5 &&
                        rsi_1h >= 30.0 && rsi_1h <= 70.0) {
                        bool buying  = (wick_lower >= 0.3);
                        bool selling = (wick_upper >= 0.3);
                        if (buying || selling) {
                            int confirming = 0;
                            if (buying) {
                                sig_dir = DIR_LONG;
                                if (cmf_val > 0.0) confirming++;
                                if (obv_div > 0.0) confirming++;
                            } else {
                                sig_dir = DIR_SHORT;
                                if (cmf_val < 0.0) confirming++;
                                if (obv_div < 0.0) confirming++;
                            }
                            sig_strength = fmin(0.4 + confirming * 0.3, 1.0);
                        }
                    }
                }
            }
        }
        else if (strategy_id == STRAT_RANGE) {
            if (h1_idx >= 2 && h1_idx < total_1h) {
                double bb_lower = (double)__ldg(&sig_1h[h1_idx * N_SIG_1H + SIG_BB_LOWER]);
                double bb_upper = (double)__ldg(&sig_1h[h1_idx * N_SIG_1H + SIG_BB_UPPER]);
                double bb_mid   = (double)__ldg(&sig_1h[h1_idx * N_SIG_1H + SIG_BB_MID]);
                double bb_bw    = (double)__ldg(&sig_1h[h1_idx * N_SIG_1H + SIG_BB_BW]);
                double adx_4h_val = (h4_idx >= 0 && h4_idx < total_4h)
                    ? (double)__ldg(&sig_4h[h4_idx * N_SIG_4H + SIG4_ADX]) : 25.0;
                double rsi_1h = (double)__ldg(&sig_1h[h1_idx * N_SIG_1H + SIG_RSI]);

                double MIN_BW = 0.02, MAX_BW = 0.15, MAX_ADX = 25.0;
                double BAND_PROX = 1.5, RSI_OS = 35.0, RSI_OB = 65.0;

                if (bb_bw >= MIN_BW && bb_bw <= MAX_BW && adx_4h_val <= MAX_ADX) {
                    double band_range = bb_upper - bb_lower;
                    if (band_range > 0.0) {
                        double prox_thresh = price * (BAND_PROX / 100.0);
                        bool near_lower = ((price - bb_lower) <= prox_thresh);
                        bool near_upper = ((bb_upper - price) <= prox_thresh);
                        if (near_lower && rsi_1h < RSI_OS) {
                            sig_dir = DIR_LONG;
                            sig_strength = fmin(0.5 + (RSI_OS - rsi_1h) / 30.0, 1.0);
                        } else if (near_upper && rsi_1h > RSI_OB) {
                            sig_dir = DIR_SHORT;
                            sig_strength = fmin(0.5 + (rsi_1h - RSI_OB) / 30.0, 1.0);
                        }
                    }
                }
            }
        }
        else if (strategy_id == STRAT_SQUEEZE) {
            if (h1_idx >= 2 && h1_idx < total_1h) {
                double bb_bw = (double)__ldg(&sig_1h[h1_idx * N_SIG_1H + SIG_BB_BW]);
                double bb_bw_prev = (h1_idx > 0 && (h1_idx - 1) < total_1h)
                    ? (double)__ldg(&sig_1h[(h1_idx - 1) * N_SIG_1H + SIG_BB_BW]) : 0.0;
                double bb_upper = (double)__ldg(&sig_1h[h1_idx * N_SIG_1H + SIG_BB_UPPER]);
                double bb_lower = (double)__ldg(&sig_1h[h1_idx * N_SIG_1H + SIG_BB_LOWER]);
                double vol_surge = (double)__ldg(&sig_1h[h1_idx * N_SIG_1H + SIG_VSR]);

                /* EMA50 slope from 4h */
                double ema50_slope = 0.0;
                if (h4_idx >= 0 && h4_idx < total_4h) {
                    int lookback = (h4_idx < 10) ? h4_idx : 10;
                    if (lookback > 0) {
                        double ema_now  = (double)__ldg(&sig_4h[h4_idx * N_SIG_4H + SIG4_EMA50]);
                        double ema_back = (double)__ldg(&sig_4h[(h4_idx - lookback) * N_SIG_4H + SIG4_EMA50]);
                        ema50_slope = ema_now - ema_back;
                    }
                }

                double COMP_THRESH = 0.04, EXP_THRESH = 0.04, MIN_VS = 1.3;
                if (bb_bw_prev < COMP_THRESH && bb_bw >= EXP_THRESH && vol_surge >= MIN_VS) {
                    double sq_width = bb_upper - bb_lower;
                    if (sq_width > 0.0) {
                        if (price > bb_upper && ema50_slope > 0.0) {
                            sig_dir = DIR_LONG;
                            sig_strength = fmin(0.6 + vol_surge * 0.1, 1.0);
                        } else if (price < bb_lower && ema50_slope < 0.0) {
                            sig_dir = DIR_SHORT;
                            sig_strength = fmin(0.6 + vol_surge * 0.1, 1.0);
                        }
                    }
                }
            }
        }
        else if (strategy_id == STRAT_FUNDING) {
            if (h1_idx >= 2 && h1_idx < total_1h) {
                double rsi_1h = (double)__ldg(&sig_1h[h1_idx * N_SIG_1H + SIG_RSI]);
                double rsi_4h = (h4_idx >= 0 && h4_idx < total_4h)
                    ? (double)__ldg(&sig_4h[h4_idx * N_SIG_4H + SIG4_RSI]) : 50.0;
                double macd_h = (double)__ldg(&sig_1h[h1_idx * N_SIG_1H + SIG_MACD_HIST]);
                double macd_h_prev = (h1_idx > 0 && (h1_idx - 1) < total_1h)
                    ? (double)__ldg(&sig_1h[(h1_idx - 1) * N_SIG_1H + SIG_MACD_HIST]) : 0.0;

                double RSI_OB = 72.0, RSI_OS = 28.0;
                double RSI_4H_OB = 60.0, RSI_4H_OS = 40.0;

                if (rsi_1h > RSI_OB) {
                    if (rsi_4h > RSI_4H_OB && macd_h < macd_h_prev) {
                        sig_dir = DIR_SHORT;
                        sig_strength = fmin((rsi_1h - RSI_OB) / 20.0 + 0.5, 1.0);
                    }
                } else if (rsi_1h < RSI_OS) {
                    if (rsi_4h < RSI_4H_OS && macd_h > macd_h_prev) {
                        sig_dir = DIR_LONG;
                        sig_strength = fmin((RSI_OS - rsi_1h) / 20.0 + 0.5, 1.0);
                    }
                }
            }
        }

        /* -- Consecutive confirmation ---------------------------- */
        if (sig_dir != DIR_NEUTRAL) {
            if (sig_dir == prev_signal_dir) {
                signal_streak++;
            } else {
                prev_signal_dir = sig_dir;
                signal_streak = 1;
            }
        } else {
            prev_signal_dir = DIR_NEUTRAL;
            signal_streak = 0;
        }

        /* -- EMA200 trend filter --------------------------------- */
        if (sig_dir != DIR_NEUTRAL) {
            double ema200_1d = price;
            if (h1d_idx >= 0 && h1d_idx < total_1d) {
                ema200_1d = (double)__ldg(&sig_1d[h1d_idx * 2 + SIG1D_EMA200]);
            }
            if (ema200_1d > 0.0 && ema200_filter_pct > 0.0) {
                double ema200_dist = (price - ema200_1d) / ema200_1d * 100.0;
                if (ema200_dist < -ema200_filter_pct && sig_dir == DIR_LONG)
                    sig_dir = DIR_NEUTRAL;
                else if (ema200_dist > ema200_filter_pct && sig_dir == DIR_SHORT)
                    sig_dir = DIR_NEUTRAL;
            }
        }

        /* -- TF weight adjustment -------------------------------- */
        if (sig_dir != DIR_NEUTRAL) {
            double tf_score = tf_weight_1h;
            if (h4_idx >= 0 && h4_idx < total_4h) {
                int lb = (h4_idx < 3) ? h4_idx : 3;
                if (lb > 0) {
                    double slope = (double)__ldg(&sig_4h[h4_idx * N_SIG_4H + SIG4_EMA50])
                                 - (double)__ldg(&sig_4h[(h4_idx - lb) * N_SIG_4H + SIG4_EMA50]);
                    if ((sig_dir == DIR_LONG && slope > 0.0) ||
                        (sig_dir == DIR_SHORT && slope < 0.0))
                        tf_score += tf_weight_4h;
                }
            }
            if (h1d_idx >= 0 && h1d_idx < total_1d) {
                int lb = (h1d_idx < 3) ? h1d_idx : 3;
                if (lb > 0) {
                    double slope = (double)__ldg(&sig_1d[h1d_idx * 2 + SIG1D_EMA50])
                                 - (double)__ldg(&sig_1d[(h1d_idx - lb) * 2 + SIG1D_EMA50]);
                    if ((sig_dir == DIR_LONG && slope > 0.0) ||
                        (sig_dir == DIR_SHORT && slope < 0.0))
                        tf_score += tf_weight_1d;
                }
            }
            double total_tf = tf_weight_1h + tf_weight_4h + tf_weight_1d;
            if (total_tf > 0.0)
                sig_strength *= tf_score / total_tf;
        }

        /* ==============================================================
         * ENTRY LOGIC
         * ============================================================== */
        /* Re-count active positions (some may have been closed above) */
        num_active = 0;
        for (int j = 0; j < MAX_POSITIONS; j++) {
            if (pos_active[j]) num_active++;
        }

        int max_open = MAX_POSITIONS;
        /* Use max_open from implicit constraint (max_position_pct implies cap) */

        if (sig_dir != DIR_NEUTRAL &&
            sig_strength >= signal_strength_min &&
            signal_streak >= consecutive_confirms &&
            num_active < max_open)
        {
            signals_gen++;

            /* Fill rate model: check if limit order would fill on next bar */
            int next_abs = abs_i + 1;
            if (next_abs < end_5m && next_abs < total_5m_bars) {
                double next_lo = (double)__ldg(&lows[next_abs]);
                double next_hi = (double)__ldg(&highs[next_abs]);
                if (sig_dir == DIR_LONG && next_lo > price) continue;
                if (sig_dir == DIR_SHORT && next_hi < price) continue;
            }

            /* Compute ATR-based stops */
            double atr_for_stops = 0.0;
            if (h1_idx >= 0 && h1_idx < total_1h) {
                atr_for_stops = (double)__ldg(&regime_1h[h1_idx * 3 + REG1H_ATR_SERIES]);
            }

            double sl, tp;
            if (atr_for_stops <= 0.0) {
                /* Fallback stops */
                if (sig_dir == DIR_SHORT) {
                    sl = price * 1.02;
                    tp = price * 0.97;
                } else {
                    sl = price * 0.98;
                    tp = price * 1.03;
                }
            } else {
                double total_fee_pct_val = (MAKER_FEE_PCT + TAKER_FEE_PCT) * 100.0;
                if (sig_dir == DIR_LONG) {
                    sl = price - atr_multiplier * atr_for_stops;
                    double risk = price - sl;
                    double fee_comp = price * (total_fee_pct_val / 100.0);
                    tp = price + rr_ratio * risk + fee_comp;
                } else {
                    sl = price + atr_multiplier * atr_for_stops;
                    double risk = sl - price;
                    double fee_comp = price * (total_fee_pct_val / 100.0);
                    tp = price - rr_ratio * risk - fee_comp;
                }
            }

            /* Position sizing (inline fixed_fractional_size) */
            double equity = current_equity;
            double stop_dist_pct = atr_multiplier * atr_pct;
            if (stop_dist_pct < 0.1) stop_dist_pct = 0.1;

            /* Compute median ATR from history */
            double median_atr;
            if (atr_hist_len >= 5) {
                /* Approximate median: mean of history (cheaper than sort on GPU) */
                double sum_atr = 0.0;
                int actual_len = (atr_hist_len < 200) ? atr_hist_len : 200;
                for (int k = 0; k < actual_len; k++)
                    sum_atr += atr_history[k];
                median_atr = sum_atr / (double)actual_len;
                if (median_atr <= 0.0) median_atr = 0.1;
            } else {
                median_atr = (atr_pct > 0.1) ? atr_pct : 0.1;
            }

            double size_eur;
            if (equity > 0.0 && stop_dist_pct > 0.0 && median_atr > 0.0) {
                double vol_scale = atr_pct / median_atr;
                if (vol_scale < 0.5) vol_scale = 0.5;
                if (vol_scale > 2.0) vol_scale = 2.0;
                double risk_eur = equity * (base_risk_pct / 100.0) * vol_scale;
                size_eur = risk_eur / (stop_dist_pct / 100.0);
                if (size_eur > equity * 0.30) size_eur = equity * 0.30;
                if (size_eur < 10.0) size_eur = 10.0;
            } else {
                size_eur = 10.0;
            }

            /* Fee gate check */
            double tp_dist_pct = (price > 0.0) ? fabs(tp - price) / price * 100.0 : 0.0;
            double entry_fee_est = size_eur * MAKER_FEE_PCT;
            double exit_fee_est  = size_eur * TAKER_FEE_PCT;
            double total_fees_est = entry_fee_est + exit_fee_est;
            double expected_profit = size_eur * (tp_dist_pct / 100.0);
            double fee_ratio = (total_fees_est > 0.0) ? expected_profit / total_fees_est : 999.0;
            if (fee_ratio < min_profit_multiple) continue;

            /* Store ATR in history */
            if (atr_pct > 0.0) {
                if (atr_hist_len < 200) {
                    atr_history[atr_hist_len] = atr_pct;
                    atr_hist_len++;
                } else {
                    atr_history[atr_hist_idx % 200] = atr_pct;
                }
                atr_hist_idx++;
            }

            /* Confidence scaling */
            if (confidence_size_scaling > 0.0) {
                double factor = 1.0 + (sig_strength - 0.5) * confidence_size_scaling;
                if (factor < 0.2) factor = 0.2;
                size_eur *= factor;
            }

            /* Cap position */
            double max_pos = INITIAL_CAPITAL * max_position_pct;
            if (size_eur > max_pos) size_eur = max_pos;

            if (size_eur < 10.0 || size_eur > balance) continue;

            /* Entry fee and cost */
            double entry_fee = size_eur * MAKER_FEE_PCT;
            double cost = size_eur + entry_fee;
            if (cost > balance) continue;

            balance -= cost;
            total_fees += entry_fee;

            /* Drawdown scaling */
            double drawdown_pct_val = (peak_balance > 0.0)
                ? ((peak_balance - current_equity) / peak_balance * 100.0) : 0.0;
            if (drawdown_pct_val >= drawdown_scale_pct)
                size_eur *= 0.5;

            /* Apply slippage to entry */
            double slipped_entry;
            if (sig_dir == DIR_SHORT)
                slipped_entry = price * (1.0 - SLIPPAGE_PCT);
            else
                slipped_entry = price * (1.0 + SLIPPAGE_PCT);

            /* Find free position slot */
            int slot = -1;
            for (int j = 0; j < MAX_POSITIONS; j++) {
                if (!pos_active[j]) { slot = j; break; }
            }
            if (slot < 0) continue;

            pos_active[slot]     = true;
            pos_dir[slot]        = sig_dir;
            pos_entry[slot]      = slipped_entry;
            pos_size[slot]       = size_eur;
            pos_sl[slot]         = sl;
            pos_tp[slot]         = tp;
            pos_peak[slot]       = slipped_entry;
            pos_entry_bar[slot]  = _local_i;
            pos_entry_fee[slot]  = entry_fee;
            pos_strategy[slot]   = strategy_id;
            pos_tp_shifted[slot] = false;
            pos_range_mid[slot]  = 0.0;
            pos_range_upper[slot] = 0.0;
            pos_range_lower[slot] = 0.0;

            /* Range strategy: override stops with band-based stops */
            if (strategy_id == STRAT_RANGE && h1_idx >= 0 && h1_idx < total_1h) {
                double bb_lower = (double)__ldg(&sig_1h[h1_idx * N_SIG_1H + SIG_BB_LOWER]);
                double bb_upper = (double)__ldg(&sig_1h[h1_idx * N_SIG_1H + SIG_BB_UPPER]);
                double bb_mid   = (double)__ldg(&sig_1h[h1_idx * N_SIG_1H + SIG_BB_MID]);
                double bandwidth = bb_upper - bb_lower;
                pos_range_mid[slot]   = bb_mid;
                pos_range_upper[slot] = bb_upper;
                pos_range_lower[slot] = bb_lower;
                if (sig_dir == DIR_LONG) {
                    pos_sl[slot] = bb_lower - 0.5 * bandwidth;
                    pos_tp[slot] = bb_mid;
                } else {
                    pos_sl[slot] = bb_upper + 0.5 * bandwidth;
                    pos_tp[slot] = bb_mid;
                }
            }
        } /* end entry logic */
    } /* end main bar loop */

    /* ================================================================
     * CLOSE REMAINING POSITIONS
     * ================================================================ */
    if (local_n > 0) {
        int last_abs = start_5m + local_n - 1;
        if (last_abs >= 0 && last_abs < total_5m_bars) {
            double last_price = (double)__ldg(&closes[last_abs]);
            int _local_i = local_n - 1;  /* needed for CLOSE_POSITION macro */
            for (int j = 0; j < MAX_POSITIONS; j++) {
                if (pos_active[j]) {
                    CLOSE_POSITION(j, last_price, false);
                }
            }
        }
    }

    /* ================================================================
     * WRITE OUTPUTS
     * ================================================================ */
    out_pnl[tid]          = total_pnl_out;
    out_trades[tid]       = total_trades;
    out_wins[tid]         = winning_trades;
    out_signals[tid]      = signals_gen;
    out_max_dd[tid]       = max_drawdown;
    out_sum_win_pnl[tid]  = sum_win_pnl;
    out_sum_loss_pnl[tid] = sum_loss_pnl;

    /* Sharpe ratio: annualised from 5-minute bar returns */
    double sharpe = 0.0;
    if (welf_n > 1.0 && welf_m2 > 0.0) {
        double variance = welf_m2 / (welf_n - 1.0);
        double std_dev  = sqrt(variance);
        if (std_dev > 0.0) {
            /* 105,120 = 365.25 * 24 * 12 (5-min bars per year) */
            sharpe = (welf_mean / std_dev) * sqrt(105120.0);
        }
    }
    out_sharpe[tid] = sharpe;

    #undef CLOSE_POSITION
    #undef MAKER_FEE_PCT
    #undef TAKER_FEE_PCT
    #undef MAX_POSITIONS
    #undef DIR_LONG
    #undef DIR_SHORT
    #undef DIR_NEUTRAL
    #undef STRAT_ORDERFLOW
    #undef STRAT_RANGE
    #undef STRAT_SQUEEZE
    #undef STRAT_FUNDING
    #undef WARMUP_BARS
    #undef SIGNAL_EVERY
    #undef INITIAL_CAPITAL
    #undef SLIPPAGE_PCT
}
"""

# ---------------------------------------------------------------------------
# Compilation helper
# ---------------------------------------------------------------------------


def compile_kernel():
    """Compile and return a CuPy RawKernel for the backtest.

    Returns
    -------
    cupy.RawKernel
        Ready-to-launch GPU kernel.
    """
    import cupy as cp

    kernel = cp.RawKernel(
        BACKTEST_KERNEL_SRC,
        "backtest_kernel",
        options=(
            "-std=c++14",
            "--use_fast_math",
        ),
    )
    return kernel


# ---------------------------------------------------------------------------
# GPU data preparation
# ---------------------------------------------------------------------------


def prepare_gpu_data(
    candle_store,
    symbols: List[str],
    days: int = 120,
) -> Dict[str, Any]:
    """Load candle + indicator data for all symbols, pack into GPU arrays.

    Parameters
    ----------
    candle_store : CandleStore
        Database-backed store for historical candles.
    symbols : list[str]
        Symbols to load (e.g. ["BTC-EUR", "ETH-EUR"]).
    days : int
        Number of days of history to load.

    Returns
    -------
    dict
        Keys: ``closes``, ``highs``, ``lows``, ``h1_map``, ``h4_map``,
        ``h1d_map``, ``sig_1h``, ``sig_4h``, ``sig_1d``, ``sig_5m``,
        ``regime_data``, ``regime_1h``, plus metadata needed for
        ``seg_info`` construction.
    """
    import cupy as cp
    import pandas as pd
    from datetime import datetime, timedelta, timezone

    from bot.backtest.engine import BacktestEngine

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)

    # Accumulators for concatenated arrays
    all_closes = []
    all_highs = []
    all_lows = []
    all_h1_map = []
    all_h4_map = []
    all_h1d_map = []
    all_sig_1h = []
    all_sig_4h = []
    all_sig_1d = []
    all_sig_5m = []
    all_regime_data = []
    all_regime_1h = []

    # Per-symbol metadata for seg_info construction
    symbol_meta = []
    offset_5m = 0
    offset_1h = 0
    offset_4h = 0
    offset_1d = 0

    for symbol in symbols:
        logger.info("GPU data prep: loading %s ...", symbol)

        # Load 5m candles
        df_5m = candle_store.get_candles(symbol, start_dt, end_dt, resample="5m")
        if df_5m.empty or len(df_5m) < 100:
            logger.warning("GPU data prep: skipping %s (only %d bars)",
                           symbol, len(df_5m) if not df_5m.empty else 0)
            continue

        n_5m = len(df_5m)

        # Resample to higher TFs
        df_1h = BacktestEngine._resample_1h(df_5m)
        df_4h = BacktestEngine._resample_4h(df_5m)
        df_1d = BacktestEngine._resample_1d(df_5m)

        n_1h = len(df_1h)
        n_4h = len(df_4h)
        n_1d = len(df_1d)

        if n_1h < 30 or n_4h < 2 or n_1d < 1:
            logger.warning("GPU data prep: skipping %s (insufficient TF bars)", symbol)
            continue

        # Compute indicators
        precomp_1h = BacktestEngine._precompute_signals(df_1h, symbol)
        precomp_4h = BacktestEngine._precompute_signals(df_4h, symbol)
        precomp_1d = BacktestEngine._precompute_signals(df_1d, symbol)
        precomp_5m = BacktestEngine._precompute_signals(df_5m, symbol)

        # Build timeframe maps (absolute indices offset by current accumulation)
        from bot.backtest.numba_loop import build_timeframe_maps
        h1_map_local, h4_map_local, h1d_map_local = build_timeframe_maps(
            df_5m.index, df_1h.index, df_4h.index, df_1d.index,
        )
        # Shift maps to absolute indices
        h1_map_abs = h1_map_local + offset_1h
        h4_map_abs = h4_map_local + offset_4h
        h1d_map_abs = h1d_map_local + offset_1d

        # Pack 1h signals: 13 columns per bar, row-major
        sig_1h_packed = np.zeros((n_1h, N_SIG_1H), dtype=np.float32)
        _safe_copy = lambda arr, n: arr[:n] if len(arr) >= n else np.pad(arr, (0, n - len(arr)))
        sig_1h_packed[:, SIG_RSI]       = _safe_copy(precomp_1h["rsi"], n_1h)
        sig_1h_packed[:, SIG_MACD_HIST] = _safe_copy(precomp_1h["macd_hist"], n_1h)
        sig_1h_packed[:, SIG_BB_LOWER]  = _safe_copy(precomp_1h["bb_lower"], n_1h)
        sig_1h_packed[:, SIG_BB_UPPER]  = _safe_copy(precomp_1h["bb_upper"], n_1h)
        sig_1h_packed[:, SIG_BB_MID]    = _safe_copy(precomp_1h["bb_mid"], n_1h)
        sig_1h_packed[:, SIG_BB_BW]     = _safe_copy(precomp_1h["bb_bandwidth"], n_1h)
        sig_1h_packed[:, SIG_VSR]       = _safe_copy(precomp_1h["vsr"], n_1h)
        sig_1h_packed[:, SIG_CMF]       = _safe_copy(precomp_1h["cmf"], n_1h)
        sig_1h_packed[:, SIG_OBV]       = _safe_copy(precomp_1h["obv"], n_1h)
        sig_1h_packed[:, SIG_OPEN]      = _safe_copy(precomp_1h["open"], n_1h)
        sig_1h_packed[:, SIG_HIGH]      = _safe_copy(precomp_1h["high"], n_1h)
        sig_1h_packed[:, SIG_LOW]       = _safe_copy(precomp_1h["low"], n_1h)
        sig_1h_packed[:, SIG_CLOSE]     = _safe_copy(precomp_1h["close"], n_1h)

        # Pack 4h signals: 3 columns per bar
        sig_4h_packed = np.zeros((n_4h, N_SIG_4H), dtype=np.float32)
        sig_4h_packed[:, SIG4_ADX]   = _safe_copy(precomp_4h["adx"], n_4h)
        sig_4h_packed[:, SIG4_EMA50] = _safe_copy(precomp_4h["ema50"], n_4h)
        sig_4h_packed[:, SIG4_RSI]   = _safe_copy(precomp_4h["rsi"], n_4h)

        # Pack 1d signals: 2 columns per bar
        sig_1d_packed = np.zeros((n_1d, N_SIG_1D), dtype=np.float32)
        sig_1d_packed[:, SIG1D_EMA200] = _safe_copy(precomp_1d["ema200"], n_1d)
        sig_1d_packed[:, SIG1D_EMA50]  = _safe_copy(precomp_1d["ema50"], n_1d)

        # 5m RSI (used for range dynamic TP)
        sig_5m_rsi = _safe_copy(precomp_5m["rsi"], n_5m).astype(np.float32)

        # Regime arrays -- 4h
        from bot.indicators.volatility import atr as compute_atr
        from bot.indicators.trend import adx as compute_adx
        regime_adx_4h = compute_adx(df_4h["high"], df_4h["low"], df_4h["close"]).values
        regime_atr_4h = compute_atr(df_4h["high"], df_4h["low"], df_4h["close"]).values
        regime_close_4h = df_4h["close"].values

        regime_data_packed = np.zeros((n_4h, N_REG_4H), dtype=np.float32)
        regime_data_packed[:, REG_ADX]   = _safe_copy(regime_adx_4h, n_4h)
        regime_data_packed[:, REG_ATR]   = _safe_copy(regime_atr_4h, n_4h)
        regime_data_packed[:, REG_CLOSE] = _safe_copy(regime_close_4h, n_4h)

        # Regime arrays -- 1h
        regime_atr_1h = compute_atr(df_1h["high"], df_1h["low"], df_1h["close"]).values
        regime_close_1h = df_1h["close"].values
        atr_1h_series = compute_atr(df_1h["high"], df_1h["low"], df_1h["close"]).values

        regime_1h_packed = np.zeros((n_1h, N_REG_1H), dtype=np.float32)
        regime_1h_packed[:, REG1H_ATR]        = _safe_copy(regime_atr_1h, n_1h)
        regime_1h_packed[:, REG1H_CLOSE]      = _safe_copy(regime_close_1h, n_1h)
        regime_1h_packed[:, REG1H_ATR_SERIES]  = _safe_copy(atr_1h_series, n_1h)

        # Accumulate
        all_closes.append(df_5m["close"].values.astype(np.float32))
        all_highs.append(df_5m["high"].values.astype(np.float32))
        all_lows.append(df_5m["low"].values.astype(np.float32))
        all_h1_map.append(h1_map_abs)
        all_h4_map.append(h4_map_abs)
        all_h1d_map.append(h1d_map_abs)
        all_sig_1h.append(sig_1h_packed.ravel())
        all_sig_4h.append(sig_4h_packed.ravel())
        all_sig_1d.append(sig_1d_packed.ravel())
        all_sig_5m.append(sig_5m_rsi)
        all_regime_data.append(regime_data_packed.ravel())
        all_regime_1h.append(regime_1h_packed.ravel())

        symbol_meta.append({
            "symbol": symbol,
            "start_5m": offset_5m,
            "end_5m": offset_5m + n_5m,
            "offset_5m": offset_5m,
            "n_5m": n_5m,
            "n_1h": n_1h,
            "n_4h": n_4h,
            "n_1d": n_1d,
        })

        offset_5m += n_5m
        offset_1h += n_1h
        offset_4h += n_4h
        offset_1d += n_1d

    if not all_closes:
        raise ValueError("No valid symbol data loaded for GPU backtest")

    # Concatenate all arrays
    closes_np = np.concatenate(all_closes)
    highs_np = np.concatenate(all_highs)
    lows_np = np.concatenate(all_lows)
    h1_map_np = np.concatenate(all_h1_map).astype(np.int32)
    h4_map_np = np.concatenate(all_h4_map).astype(np.int32)
    h1d_map_np = np.concatenate(all_h1d_map).astype(np.int32)
    sig_1h_np = np.concatenate(all_sig_1h)
    sig_4h_np = np.concatenate(all_sig_4h)
    sig_1d_np = np.concatenate(all_sig_1d)
    sig_5m_np = np.concatenate(all_sig_5m)
    regime_data_np = np.concatenate(all_regime_data)
    regime_1h_np = np.concatenate(all_regime_1h)

    total_5m = len(closes_np)
    total_1h = offset_1h
    total_4h = offset_4h
    total_1d = offset_1d

    logger.info(
        "GPU data prep: %d symbols, %d 5m bars, %d 1h, %d 4h, %d 1d",
        len(symbol_meta), total_5m, total_1h, total_4h, total_1d,
    )

    # Transfer to GPU
    result = {
        "closes": cp.asarray(closes_np),
        "highs": cp.asarray(highs_np),
        "lows": cp.asarray(lows_np),
        "total_5m_bars": total_5m,
        "h1_map": cp.asarray(h1_map_np),
        "h4_map": cp.asarray(h4_map_np),
        "h1d_map": cp.asarray(h1d_map_np),
        "sig_1h": cp.asarray(sig_1h_np),
        "total_1h": total_1h,
        "sig_4h": cp.asarray(sig_4h_np),
        "total_4h": total_4h,
        "sig_1d": cp.asarray(sig_1d_np),
        "total_1d": total_1d,
        "sig_5m": cp.asarray(sig_5m_np),
        "regime_data": cp.asarray(regime_data_np),
        "regime_1h": cp.asarray(regime_1h_np),
        "symbol_meta": symbol_meta,
    }

    return result


# ---------------------------------------------------------------------------
# Convenience: build seg_info + params arrays for a batch of runs
# ---------------------------------------------------------------------------


def build_batch_inputs(
    gpu_data: Dict[str, Any],
    param_dicts: List[Dict[str, float]],
    strategy_ids: List[int],
    symbol_indices: Optional[List[int]] = None,
    windows: Optional[List[tuple]] = None,
) -> tuple:
    """Build ``params`` and ``seg_info`` GPU arrays for a batch launch.

    Parameters
    ----------
    gpu_data : dict
        Output of :func:`prepare_gpu_data`.
    param_dicts : list[dict]
        One parameter dict per environment in the batch.
    strategy_ids : list[int]
        Strategy ID per environment.
    symbol_indices : list[int] or None
        Index into ``gpu_data["symbol_meta"]`` per environment.
        Defaults to 0 (first symbol) for all.
    windows : list[tuple] or None
        Optional (start_bar, end_bar) window within each symbol's data.
        Uses full range if None.

    Returns
    -------
    (params_gpu, seg_info_gpu, batch_size)
        CuPy arrays ready for kernel launch.
    """
    import cupy as cp

    batch_size = len(param_dicts)
    meta = gpu_data["symbol_meta"]

    if symbol_indices is None:
        symbol_indices = [0] * batch_size

    # Parameter name -> index mapping
    _PARAM_MAP = {
        "atr_multiplier": P_ATR_MULT,
        "rr_ratio": P_RR_RATIO,
        "base_risk_pct": P_BASE_RISK_PCT,
        "signal_strength_min": P_SIGNAL_MIN,
        "tf_weight_1h": P_TF_W_1H,
        "tf_weight_4h": P_TF_W_4H,
        "tf_weight_1d": P_TF_W_1D,
        "confidence_size_scaling": P_CONF_SCALE,
        "ema200_filter_pct": P_EMA200_FILTER,
        "volatile_atr_threshold": P_VOL_ATR_THRESH,
        "consecutive_confirms": P_CONSEC_CONFIRMS,
        "confluence_boost": P_CONFLUENCE,
        "drawdown_scale_pct": P_DD_SCALE,
        "max_position_pct": P_MAX_POS_PCT,
        "trail_activation_mult": P_TRAIL_ACT,
        "max_hold_bars": P_MAX_HOLD_BARS,
        "range_max_hold_bars": P_RANGE_HOLD_BARS,
        "min_atr_pct": P_MIN_ATR_PCT,
        "quiet_atr_threshold": P_QUIET_ATR,
        "regime_adx_threshold": P_REGIME_ADX,
        "ranging_adx_threshold": P_RANGING_ADX,
        "min_profit_multiple": P_MIN_PROFIT_MULT,
    }

    # Default values
    _DEFAULTS = {
        "atr_multiplier": 2.0,
        "rr_ratio": 2.0,
        "base_risk_pct": 3.0,
        "signal_strength_min": 0.0,
        "tf_weight_1h": 1.0,
        "tf_weight_4h": 1.0,
        "tf_weight_1d": 1.0,
        "confidence_size_scaling": 0.0,
        "ema200_filter_pct": 2.0,
        "volatile_atr_threshold": 4.0,
        "consecutive_confirms": 1.0,
        "confluence_boost": 1.2,
        "drawdown_scale_pct": 3.0,
        "max_position_pct": 0.30,
        "trail_activation_mult": 1.5,
        "max_hold_bars": 576.0,       # 48h * 12
        "range_max_hold_bars": 864.0, # 72h * 12
        "min_atr_pct": 0.0,
        "quiet_atr_threshold": 1.0,
        "regime_adx_threshold": 24.0,
        "ranging_adx_threshold": 20.0,
        "min_profit_multiple": 3.0,
    }

    params_np = np.zeros((batch_size, N_PARAMS), dtype=np.float64)
    seg_info_np = np.zeros((batch_size, 5), dtype=np.int32)

    for i in range(batch_size):
        pdict = param_dicts[i]
        sid = symbol_indices[i]
        sm = meta[sid]

        # Fill params with defaults then override
        for name, idx in _PARAM_MAP.items():
            val = pdict.get(name, _DEFAULTS.get(name, 0.0))
            # Convert hours to bars for hold time parameters
            if name == "max_hold_bars" and "max_hold_hours" in pdict:
                val = pdict["max_hold_hours"] * 12.0
            elif name == "range_max_hold_bars" and "range_max_hold_hours" in pdict:
                val = pdict["range_max_hold_hours"] * 12.0
            params_np[i, idx] = float(val)

        # Segment info
        if windows is not None and i < len(windows):
            w_start, w_end = windows[i]
            seg_info_np[i, 0] = sm["start_5m"] + w_start  # absolute start
            seg_info_np[i, 1] = sm["start_5m"] + w_end    # absolute end
        else:
            seg_info_np[i, 0] = sm["start_5m"]
            seg_info_np[i, 1] = sm["end_5m"]

        seg_info_np[i, 2] = sm["start_5m"]   # offset_5m
        seg_info_np[i, 3] = strategy_ids[i]
        seg_info_np[i, 4] = seg_info_np[i, 1] - seg_info_np[i, 0]  # n_bars

    return cp.asarray(params_np.ravel()), cp.asarray(seg_info_np.ravel()), batch_size


def launch_backtest(
    kernel,
    gpu_data: Dict[str, Any],
    params_gpu,
    seg_info_gpu,
    batch_size: int,
    threads_per_block: int = 128,
) -> Dict[str, np.ndarray]:
    """Launch the GPU backtest kernel and collect results.

    Parameters
    ----------
    kernel : cupy.RawKernel
        Compiled kernel from :func:`compile_kernel`.
    gpu_data : dict
        Output of :func:`prepare_gpu_data`.
    params_gpu : cupy.ndarray
        Packed parameter array (float64).
    seg_info_gpu : cupy.ndarray
        Packed segment info array (int32).
    batch_size : int
        Number of environments.
    threads_per_block : int
        CUDA threads per block (default 128 for RTX 2060 SUPER).

    Returns
    -------
    dict
        Keys: ``pnl``, ``trades``, ``wins``, ``signals``, ``max_dd``,
        ``sharpe``, ``sum_win_pnl``, ``sum_loss_pnl`` -- all numpy arrays
        of length ``batch_size``.
    """
    import cupy as cp

    blocks = (batch_size + threads_per_block - 1) // threads_per_block

    # Allocate output arrays
    out_pnl = cp.zeros(batch_size, dtype=cp.float64)
    out_trades = cp.zeros(batch_size, dtype=cp.int32)
    out_wins = cp.zeros(batch_size, dtype=cp.int32)
    out_signals = cp.zeros(batch_size, dtype=cp.int32)
    out_max_dd = cp.zeros(batch_size, dtype=cp.float64)
    out_sharpe = cp.zeros(batch_size, dtype=cp.float64)
    out_sum_win = cp.zeros(batch_size, dtype=cp.float64)
    out_sum_loss = cp.zeros(batch_size, dtype=cp.float64)

    kernel(
        (blocks,), (threads_per_block,),
        (
            gpu_data["closes"],
            gpu_data["highs"],
            gpu_data["lows"],
            np.int32(gpu_data["total_5m_bars"]),
            gpu_data["h1_map"],
            gpu_data["h4_map"],
            gpu_data["h1d_map"],
            gpu_data["sig_1h"],
            np.int32(gpu_data["total_1h"]),
            gpu_data["sig_4h"],
            np.int32(gpu_data["total_4h"]),
            gpu_data["sig_1d"],
            np.int32(gpu_data["total_1d"]),
            gpu_data["sig_5m"],
            gpu_data["regime_data"],
            gpu_data["regime_1h"],
            params_gpu,
            seg_info_gpu,
            np.int32(batch_size),
            out_pnl,
            out_trades,
            out_wins,
            out_signals,
            out_max_dd,
            out_sharpe,
            out_sum_win,
            out_sum_loss,
        ),
    )

    # Synchronize and copy back to CPU
    cp.cuda.Stream.null.synchronize()

    return {
        "pnl": out_pnl.get(),
        "trades": out_trades.get(),
        "wins": out_wins.get(),
        "signals": out_signals.get(),
        "max_dd": out_max_dd.get(),
        "sharpe": out_sharpe.get(),
        "sum_win_pnl": out_sum_win.get(),
        "sum_loss_pnl": out_sum_loss.get(),
    }
