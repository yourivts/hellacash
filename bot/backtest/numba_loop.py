"""Numba-JIT compiled backtest bar loop.

Replaces the pure-Python ``for i in range(WARMUP, n)`` loop in
``BacktestEngine.run()`` with a compiled kernel that operates entirely
on numpy arrays.  All position and trade state is kept in fixed-size
parallel arrays — no Python objects inside the hot path.

The kernel returns flat numpy arrays which the caller converts back
into ``TradeRecord`` / ``BacktestResult`` objects.
"""
from __future__ import annotations

import numpy as np
import numba

# ---------------------------------------------------------------------------
# Constants (duplicated here so Numba can see them as compile-time literals)
# ---------------------------------------------------------------------------
MAKER_FEE_PCT = 0.0015
TAKER_FEE_PCT = 0.0025
MAX_POSITIONS = 10
MAX_TRADES = 500

# Strategy IDs
STRAT_ORDERFLOW = 0
STRAT_RANGE = 1
STRAT_SQUEEZE = 2
STRAT_FUNDING = 3

# Exit reason IDs
EXIT_STOP_LOSS = 0
EXIT_TAKE_PROFIT = 1
EXIT_TIME = 2
EXIT_END_OF_DATA = 3
EXIT_RANGE_MID = 4

# Direction encoding
DIR_LONG = 1
DIR_SHORT = -1
DIR_NEUTRAL = 0

# Regime encoding
REGIME_QUIET = 0
REGIME_VOLATILE = 1
REGIME_TRENDING = 2
REGIME_RANGING = 3
REGIME_NEUTRAL = 4


# ---------------------------------------------------------------------------
# Helper: pre-compute 5m→1h/4h/1d bar index mappings
# ---------------------------------------------------------------------------

def build_timeframe_maps(
    timestamps_5m: np.ndarray,
    h1_timestamps: np.ndarray,
    h4_timestamps: np.ndarray,
    h1d_timestamps: np.ndarray,
) -> tuple:
    """Pre-compute per-5m-bar index into 1h/4h/1d arrays.

    Returns (h1_map, h4_map, h1d_map) — int32 arrays of length len(timestamps_5m).
    """
    n = len(timestamps_5m)
    h1_map = np.searchsorted(h1_timestamps, timestamps_5m, side="right").astype(np.int32) - 1
    h4_map = np.searchsorted(h4_timestamps, timestamps_5m, side="right").astype(np.int32) - 1
    h1d_map = np.searchsorted(h1d_timestamps, timestamps_5m, side="right").astype(np.int32) - 1
    return h1_map, h4_map, h1d_map


# ---------------------------------------------------------------------------
# The main Numba kernel
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def _backtest_kernel(
    # ── Candle data (window slice) ────────────────────────────────
    closes: np.ndarray,       # float64[n]
    highs: np.ndarray,        # float64[n]
    lows: np.ndarray,         # float64[n]
    # ── Time-frame index maps (window-relative) ──────────────────
    h1_map: np.ndarray,       # int32[n]  — 5m bar → 1h bar index
    h4_map: np.ndarray,       # int32[n]  — 5m bar → 4h bar index
    h1d_map: np.ndarray,      # int32[n]  — 5m bar → 1d bar index
    # ── Precomputed indicator arrays (full-history) ──────────────
    # Offsets to shift window-local index into full-history arrays
    offset_5m: int,
    # 1h signals
    sig_1h_rsi: np.ndarray,
    sig_1h_macd_hist: np.ndarray,
    sig_1h_bb_lower: np.ndarray,
    sig_1h_bb_upper: np.ndarray,
    sig_1h_bb_mid: np.ndarray,
    sig_1h_bb_bandwidth: np.ndarray,
    sig_1h_vsr: np.ndarray,
    sig_1h_cmf: np.ndarray,
    sig_1h_obv: np.ndarray,
    sig_1h_open: np.ndarray,
    sig_1h_high: np.ndarray,
    sig_1h_low: np.ndarray,
    sig_1h_close: np.ndarray,
    # 4h signals
    sig_4h_adx: np.ndarray,
    sig_4h_ema50: np.ndarray,
    sig_4h_rsi: np.ndarray,
    # 1d signals
    sig_1d_ema200: np.ndarray,
    sig_1d_ema50: np.ndarray,
    # 5m signals (for range dynamic TP)
    sig_5m_rsi: np.ndarray,
    # Regime / ATR arrays (full-history)
    regime_adx_4h: np.ndarray,
    regime_atr_4h: np.ndarray,
    regime_close_4h: np.ndarray,
    regime_atr_1h: np.ndarray,
    regime_close_1h: np.ndarray,
    atr_1h_series: np.ndarray,
    # ── Strategy parameters (flat) ───────────────────────────────
    strategy_id: int,
    atr_multiplier: float,
    rr_ratio: float,
    base_risk_pct: float,
    signal_strength_min: float,
    tf_weight_1h: float,
    tf_weight_4h: float,
    tf_weight_1d: float,
    confidence_size_scaling: float,
    ema200_filter_pct: float,
    volatile_atr_threshold: float,
    consecutive_confirms: int,
    confluence_boost: float,
    drawdown_scale_pct: float,
    max_position_pct: float,
    trail_activation_mult: float,
    max_hold_bars: int,
    range_max_hold_bars: int,
    min_atr_pct: float,
    quiet_atr_threshold: float,
    regime_adx_threshold: float,
    ranging_adx_threshold: float,
    min_profit_multiple: float,
    # ── Engine constants ─────────────────────────────────────────
    warmup: int,
    signal_every: int,
    max_open: int,
    initial_capital: float,
    slippage_pct: float,
):
    """Run the full backtest bar loop.  Returns a structured result tuple."""
    n = len(closes)

    # ── Position state (parallel arrays, fixed max size) ──────────
    pos_active = np.zeros(MAX_POSITIONS, dtype=numba.boolean)
    pos_dir = np.zeros(MAX_POSITIONS, dtype=numba.int8)      # 1=LONG, -1=SHORT
    pos_entry = np.zeros(MAX_POSITIONS, dtype=numba.float64)
    pos_size = np.zeros(MAX_POSITIONS, dtype=numba.float64)
    pos_sl = np.zeros(MAX_POSITIONS, dtype=numba.float64)
    pos_tp = np.zeros(MAX_POSITIONS, dtype=numba.float64)
    pos_peak = np.zeros(MAX_POSITIONS, dtype=numba.float64)   # highest/lowest
    pos_entry_bar = np.zeros(MAX_POSITIONS, dtype=numba.int32)
    pos_entry_fee = np.zeros(MAX_POSITIONS, dtype=numba.float64)
    pos_strategy = np.zeros(MAX_POSITIONS, dtype=numba.int8)
    pos_tp_shifted = np.zeros(MAX_POSITIONS, dtype=numba.boolean)
    pos_range_mid = np.zeros(MAX_POSITIONS, dtype=numba.float64)
    pos_range_upper = np.zeros(MAX_POSITIONS, dtype=numba.float64)
    pos_range_lower = np.zeros(MAX_POSITIONS, dtype=numba.float64)

    # ── Trade output arrays ───────────────────────────────────────
    trade_dir = np.zeros(MAX_TRADES, dtype=numba.int8)
    trade_entry_price = np.zeros(MAX_TRADES, dtype=numba.float64)
    trade_exit_price = np.zeros(MAX_TRADES, dtype=numba.float64)
    trade_size = np.zeros(MAX_TRADES, dtype=numba.float64)
    trade_pnl = np.zeros(MAX_TRADES, dtype=numba.float64)
    trade_pnl_pct = np.zeros(MAX_TRADES, dtype=numba.float64)
    trade_reason = np.zeros(MAX_TRADES, dtype=numba.int8)
    trade_strategy = np.zeros(MAX_TRADES, dtype=numba.int8)
    num_trades = 0

    # ── Equity tracking ──────────────────────────────────────────
    equity_curve = np.empty(n - warmup, dtype=numba.float64)
    balance = initial_capital
    peak_balance = initial_capital
    total_fees_paid = 0.0
    signals_generated = 0
    signals_filled = 0
    quiet_hours_skipped = 0

    # ── Consecutive confirmation state ────────────────────────────
    prev_signal_dir = 0  # DIR_NEUTRAL
    signal_streak = 0
    last_trade_close_bar = -999

    # ── ATR% history for position sizing ──────────────────────────
    atr_history = np.zeros(200, dtype=numba.float64)
    atr_history_len = 0
    atr_history_idx = 0  # circular buffer index

    # ==================================================================
    # MAIN BAR LOOP
    # ==================================================================
    for i in range(warmup, n):
        price = closes[i]
        hi = highs[i]
        lo = lows[i]
        abs_i = i + offset_5m
        eq_idx = i - warmup

        # Map to timeframe indices
        h1_idx = h1_map[i]
        h4_idx = h4_map[i]
        h1d_idx = h1d_map[i]

        # ── Count active positions ────────────────────────────────
        num_active = 0
        for j in range(MAX_POSITIONS):
            if pos_active[j]:
                num_active += 1

        # ==============================================================
        # EXIT CHECKING (every bar)
        # ==============================================================
        if num_active > 0:
            atr_val = 0.0
            if 0 <= h1_idx < len(atr_1h_series):
                atr_val = atr_1h_series[h1_idx]

            for j in range(MAX_POSITIONS):
                if not pos_active[j]:
                    continue
                d = pos_dir[j]

                # Time-based exit
                hold_limit = range_max_hold_bars if pos_strategy[j] == STRAT_RANGE else max_hold_bars
                if i - pos_entry_bar[j] >= hold_limit:
                    # Close position (time exit)
                    exit_price = price
                    if d == DIR_SHORT:
                        slipped = exit_price * (1.0 + slippage_pct)
                        pchange = (pos_entry[j] - slipped) / pos_entry[j]
                    else:
                        slipped = exit_price * (1.0 - slippage_pct)
                        pchange = (slipped - pos_entry[j]) / pos_entry[j]
                    gross = pos_size[j] * pchange
                    qty = pos_size[j] / pos_entry[j]
                    exit_fee = slipped * qty * TAKER_FEE_PCT
                    total_fees_paid += exit_fee
                    pnl = gross - exit_fee - pos_entry_fee[j]
                    balance += pos_size[j] + pnl
                    pos_active[j] = False
                    if num_trades < MAX_TRADES:
                        trade_dir[num_trades] = d
                        trade_entry_price[num_trades] = pos_entry[j]
                        trade_exit_price[num_trades] = exit_price
                        trade_size[num_trades] = pos_size[j]
                        trade_pnl[num_trades] = pnl
                        trade_pnl_pct[num_trades] = (pnl / pos_size[j]) * 100.0 if pos_size[j] > 0 else 0.0
                        trade_reason[num_trades] = EXIT_TIME
                        trade_strategy[num_trades] = pos_strategy[j]
                        num_trades += 1
                    last_trade_close_bar = i
                    continue

                # Range dynamic TP
                if pos_strategy[j] == STRAT_RANGE and not pos_tp_shifted[j]:
                    crossed = False
                    if d == DIR_LONG and price >= pos_range_mid[j]:
                        crossed = True
                    elif d == DIR_SHORT and price <= pos_range_mid[j]:
                        crossed = True
                    if crossed and abs_i >= 3:
                        cur_rsi = sig_5m_rsi[abs_i] if abs_i < len(sig_5m_rsi) else 50.0
                        prev_rsi = sig_5m_rsi[abs_i - 3] if (abs_i - 3) >= 0 and (abs_i - 3) < len(sig_5m_rsi) else cur_rsi
                        should_shift = False
                        if d == DIR_LONG and cur_rsi > prev_rsi:
                            should_shift = True
                            pos_tp[j] = pos_range_upper[j]
                        elif d == DIR_SHORT and cur_rsi < prev_rsi:
                            should_shift = True
                            pos_tp[j] = pos_range_lower[j]
                        if should_shift:
                            pos_tp_shifted[j] = True
                        else:
                            # Close at mid
                            exit_price = pos_range_mid[j]
                            if d == DIR_SHORT:
                                slipped = exit_price * (1.0 + slippage_pct)
                                pchange = (pos_entry[j] - slipped) / pos_entry[j]
                            else:
                                slipped = exit_price * (1.0 - slippage_pct)
                                pchange = (slipped - pos_entry[j]) / pos_entry[j]
                            gross = pos_size[j] * pchange
                            qty = pos_size[j] / pos_entry[j]
                            exit_fee = slipped * qty * MAKER_FEE_PCT
                            total_fees_paid += exit_fee
                            pnl = gross - exit_fee - pos_entry_fee[j]
                            balance += pos_size[j] + pnl
                            pos_active[j] = False
                            if num_trades < MAX_TRADES:
                                trade_dir[num_trades] = d
                                trade_entry_price[num_trades] = pos_entry[j]
                                trade_exit_price[num_trades] = exit_price
                                trade_size[num_trades] = pos_size[j]
                                trade_pnl[num_trades] = pnl
                                trade_pnl_pct[num_trades] = (pnl / pos_size[j]) * 100.0 if pos_size[j] > 0 else 0.0
                                trade_reason[num_trades] = EXIT_RANGE_MID
                                trade_strategy[num_trades] = pos_strategy[j]
                                num_trades += 1
                            last_trade_close_bar = i
                            continue

                # Update trailing price
                if d == DIR_SHORT:
                    if price < pos_peak[j]:
                        pos_peak[j] = price
                else:
                    if price > pos_peak[j]:
                        pos_peak[j] = price

                # Trailing stop logic
                if pos_strategy[j] == STRAT_RANGE and pos_tp_shifted[j]:
                    # Tight 1% trail for range after TP shift
                    if d == DIR_LONG:
                        trail_level = pos_peak[j] * 0.99
                        if trail_level > pos_sl[j]:
                            pos_sl[j] = trail_level
                    else:
                        trail_level = pos_peak[j] * 1.01
                        if trail_level < pos_sl[j]:
                            pos_sl[j] = trail_level
                elif atr_val > 0:
                    # ATR trailing stop with activation threshold
                    trail_dist = atr_val * atr_multiplier
                    entry_p = pos_entry[j]
                    cur_stop = pos_sl[j]
                    if entry_p > 0 and trail_activation_mult > 0:
                        stop_dist = abs(entry_p - cur_stop)
                        if d == DIR_LONG:
                            profit = pos_peak[j] - entry_p
                        else:
                            profit = entry_p - pos_peak[j]
                        if stop_dist > 0 and profit < stop_dist * trail_activation_mult:
                            pass  # don't trail yet
                        else:
                            if d == DIR_LONG:
                                new_stop = pos_peak[j] - trail_dist
                                if new_stop > cur_stop:
                                    pos_sl[j] = new_stop
                            else:
                                new_stop = pos_peak[j] + trail_dist
                                if cur_stop > 0 and new_stop < cur_stop:
                                    pos_sl[j] = new_stop
                                elif cur_stop <= 0:
                                    pos_sl[j] = new_stop
                    else:
                        if d == DIR_LONG:
                            new_stop = pos_peak[j] - trail_dist
                            if new_stop > cur_stop:
                                pos_sl[j] = new_stop
                        else:
                            new_stop = pos_peak[j] + trail_dist
                            if cur_stop > 0 and new_stop < cur_stop:
                                pos_sl[j] = new_stop
                            elif cur_stop <= 0:
                                pos_sl[j] = new_stop

                # SL/TP trigger
                triggered = -1  # no trigger
                trigger_price = price
                if d == DIR_SHORT:
                    if hi >= pos_sl[j]:
                        triggered = EXIT_STOP_LOSS
                        trigger_price = pos_sl[j]
                    elif lo <= pos_tp[j]:
                        triggered = EXIT_TAKE_PROFIT
                        trigger_price = pos_tp[j]
                else:
                    if lo <= pos_sl[j]:
                        triggered = EXIT_STOP_LOSS
                        trigger_price = pos_sl[j]
                    elif hi >= pos_tp[j]:
                        triggered = EXIT_TAKE_PROFIT
                        trigger_price = pos_tp[j]

                if triggered >= 0:
                    exit_price = trigger_price
                    if d == DIR_SHORT:
                        slipped = exit_price * (1.0 + slippage_pct)
                        pchange = (pos_entry[j] - slipped) / pos_entry[j]
                    else:
                        slipped = exit_price * (1.0 - slippage_pct)
                        pchange = (slipped - pos_entry[j]) / pos_entry[j]
                    gross = pos_size[j] * pchange
                    qty = pos_size[j] / pos_entry[j]
                    fee_rate = MAKER_FEE_PCT if triggered == EXIT_TAKE_PROFIT else TAKER_FEE_PCT
                    exit_fee = slipped * qty * fee_rate
                    total_fees_paid += exit_fee
                    pnl = gross - exit_fee - pos_entry_fee[j]
                    balance += pos_size[j] + pnl
                    pos_active[j] = False
                    if num_trades < MAX_TRADES:
                        trade_dir[num_trades] = d
                        trade_entry_price[num_trades] = pos_entry[j]
                        trade_exit_price[num_trades] = exit_price
                        trade_size[num_trades] = pos_size[j]
                        trade_pnl[num_trades] = pnl
                        trade_pnl_pct[num_trades] = (pnl / pos_size[j]) * 100.0 if pos_size[j] > 0 else 0.0
                        trade_reason[num_trades] = triggered
                        trade_strategy[num_trades] = pos_strategy[j]
                        num_trades += 1
                    last_trade_close_bar = i

        # ── Update peak equity ────────────────────────────────────
        current_equity = balance
        for j in range(MAX_POSITIONS):
            if pos_active[j]:
                if pos_dir[j] == DIR_SHORT:
                    change = (pos_entry[j] - price) / pos_entry[j]
                    current_equity += pos_size[j] * (1.0 + change)
                else:
                    current_equity += pos_size[j] * (price / pos_entry[j])
        if current_equity > peak_balance:
            peak_balance = current_equity

        # ==============================================================
        # SIGNAL GENERATION (every SIGNAL_EVERY bars)
        # ==============================================================
        if (i - warmup) % signal_every != 0:
            equity_curve[eq_idx] = current_equity
            continue

        # Cooldown
        if i - last_trade_close_bar < 0:  # cooldown_bars = 0
            equity_curve[eq_idx] = current_equity
            continue

        if h1_idx < 30:
            equity_curve[eq_idx] = current_equity
            continue

        # ── Regime detection ──────────────────────────────────────
        atr_val_4h = regime_atr_4h[h4_idx] if 0 <= h4_idx < len(regime_atr_4h) else 0.0
        price_4h = regime_close_4h[h4_idx] if 0 <= h4_idx < len(regime_close_4h) else price
        adx_4h = regime_adx_4h[h4_idx] if 0 <= h4_idx < len(regime_adx_4h) else 20.0
        atr_pct_4h = (atr_val_4h / price_4h * 100.0) if price_4h > 0 else 0.0

        regime = REGIME_NEUTRAL
        if atr_pct_4h < quiet_atr_threshold:
            regime = REGIME_QUIET
        elif atr_pct_4h > volatile_atr_threshold:
            regime = REGIME_VOLATILE
        elif adx_4h > regime_adx_threshold:
            regime = REGIME_TRENDING
        elif adx_4h < ranging_adx_threshold:
            regime = REGIME_RANGING

        if regime == REGIME_QUIET:
            quiet_hours_skipped += 1
            equity_curve[eq_idx] = current_equity
            continue

        # 1h ATR% for position sizing
        atr_val_1h = regime_atr_1h[h1_idx] if 0 <= h1_idx < len(regime_atr_1h) else 0.0
        price_1h = regime_close_1h[h1_idx] if 0 <= h1_idx < len(regime_close_1h) else price
        atr_pct = (atr_val_1h / price_1h) * 100.0 if price_1h > 0 else 0.0

        if atr_pct < min_atr_pct:
            equity_curve[eq_idx] = current_equity
            continue

        # ── Strategy evaluation ───────────────────────────────────
        sig_dir = DIR_NEUTRAL
        sig_strength = 0.0

        if strategy_id == STRAT_ORDERFLOW:
            if h1_idx >= 2:
                o = sig_1h_open[h1_idx]
                h = sig_1h_high[h1_idx]
                l = sig_1h_low[h1_idx]
                c = sig_1h_close[h1_idx]
                candle_range = h - l
                if candle_range > 0:
                    body_ratio = abs(c - o) / candle_range
                    wick_lower = (min(o, c) - l) / candle_range
                    wick_upper = (h - max(o, c)) / candle_range
                    vol_surge = sig_1h_vsr[h1_idx] if h1_idx < len(sig_1h_vsr) else 1.0
                    rsi_1h = sig_1h_rsi[h1_idx] if h1_idx < len(sig_1h_rsi) else 50.0
                    cmf_val = sig_1h_cmf[h1_idx] if h1_idx < len(sig_1h_cmf) else 0.0
                    obv_now = sig_1h_obv[h1_idx] if h1_idx < len(sig_1h_obv) else 0.0
                    obv_prev = sig_1h_obv[h1_idx - 12] if h1_idx >= 12 and h1_idx < len(sig_1h_obv) else obv_now
                    price_prev = sig_1h_close[h1_idx - 12] if h1_idx >= 12 else price
                    obv_div = 0.0
                    if price < price_prev and obv_now > obv_prev:
                        obv_div = 1.0
                    elif price > price_prev and obv_now < obv_prev:
                        obv_div = -1.0

                    # Inline orderflow evaluate_1h
                    if body_ratio <= 0.4 and vol_surge >= 1.5 and 30.0 <= rsi_1h <= 70.0:
                        buying = wick_lower >= 0.3
                        selling = wick_upper >= 0.3
                        if buying or selling:
                            confirming = 0
                            if buying:
                                sig_dir = DIR_LONG
                                if cmf_val > 0:
                                    confirming += 1
                                if obv_div > 0:
                                    confirming += 1
                            else:
                                sig_dir = DIR_SHORT
                                if cmf_val < 0:
                                    confirming += 1
                                if obv_div < 0:
                                    confirming += 1
                            sig_strength = min(0.4 + confirming * 0.3, 1.0)

        elif strategy_id == STRAT_RANGE:
            if h1_idx >= 2:
                bb_lower = sig_1h_bb_lower[h1_idx] if h1_idx < len(sig_1h_bb_lower) else price
                bb_upper = sig_1h_bb_upper[h1_idx] if h1_idx < len(sig_1h_bb_upper) else price
                bb_mid = sig_1h_bb_mid[h1_idx] if h1_idx < len(sig_1h_bb_mid) else price
                bb_bw = sig_1h_bb_bandwidth[h1_idx] if h1_idx < len(sig_1h_bb_bandwidth) else 0.0
                adx_4h_val = sig_4h_adx[h4_idx] if 0 <= h4_idx < len(sig_4h_adx) else 25.0
                rsi_1h = sig_1h_rsi[h1_idx] if h1_idx < len(sig_1h_rsi) else 50.0

                # Inline range evaluate_1h
                # Constants from RangeTradingStrategy
                MIN_BW = 0.02
                MAX_BW = 0.15
                MAX_ADX = 25.0
                BAND_PROX = 1.5
                RSI_OS = 35.0
                RSI_OB = 65.0
                if MIN_BW <= bb_bw <= MAX_BW and adx_4h_val <= MAX_ADX:
                    band_range = bb_upper - bb_lower
                    if band_range > 0:
                        prox_thresh = price * (BAND_PROX / 100.0)
                        near_lower = (price - bb_lower) <= prox_thresh
                        near_upper = (bb_upper - price) <= prox_thresh
                        if near_lower and rsi_1h < RSI_OS:
                            sig_dir = DIR_LONG
                            sig_strength = min(0.5 + (RSI_OS - rsi_1h) / 30.0, 1.0)
                        elif near_upper and rsi_1h > RSI_OB:
                            sig_dir = DIR_SHORT
                            sig_strength = min(0.5 + (rsi_1h - RSI_OB) / 30.0, 1.0)

        elif strategy_id == STRAT_SQUEEZE:
            if h1_idx >= 2:
                bb_bw = sig_1h_bb_bandwidth[h1_idx] if h1_idx < len(sig_1h_bb_bandwidth) else 0.0
                bb_bw_prev = sig_1h_bb_bandwidth[h1_idx - 1] if h1_idx > 0 and (h1_idx - 1) < len(sig_1h_bb_bandwidth) else 0.0
                bb_upper = sig_1h_bb_upper[h1_idx] if h1_idx < len(sig_1h_bb_upper) else price
                bb_lower = sig_1h_bb_lower[h1_idx] if h1_idx < len(sig_1h_bb_lower) else price
                vol_surge = sig_1h_vsr[h1_idx] if h1_idx < len(sig_1h_vsr) else 1.0

                # EMA50 slope from 4h
                ema50_slope = 0.0
                if 0 <= h4_idx < len(sig_4h_ema50):
                    lookback = min(10, h4_idx)
                    if lookback > 0:
                        ema50_slope = sig_4h_ema50[h4_idx] - sig_4h_ema50[h4_idx - lookback]

                # Inline squeeze evaluate_1h
                COMP_THRESH = 0.04
                EXP_THRESH = 0.04
                MIN_VS = 1.3
                if bb_bw_prev < COMP_THRESH and bb_bw >= EXP_THRESH and vol_surge >= MIN_VS:
                    sq_width = bb_upper - bb_lower
                    if sq_width > 0:
                        if price > bb_upper:
                            if ema50_slope > 0:
                                sig_dir = DIR_LONG
                                sig_strength = min(0.6 + vol_surge * 0.1, 1.0)
                        elif price < bb_lower:
                            if ema50_slope < 0:
                                sig_dir = DIR_SHORT
                                sig_strength = min(0.6 + vol_surge * 0.1, 1.0)

        elif strategy_id == STRAT_FUNDING:
            if h1_idx >= 2:
                rsi_1h = sig_1h_rsi[h1_idx] if h1_idx < len(sig_1h_rsi) else 50.0
                rsi_4h = sig_4h_rsi[h4_idx] if 0 <= h4_idx < len(sig_4h_rsi) else 50.0
                macd_h = sig_1h_macd_hist[h1_idx] if h1_idx < len(sig_1h_macd_hist) else 0.0
                macd_h_prev = sig_1h_macd_hist[h1_idx - 1] if h1_idx > 0 and (h1_idx - 1) < len(sig_1h_macd_hist) else 0.0

                # Inline funding_contrarian evaluate_1h
                RSI_OB = 72.0
                RSI_OS = 28.0
                RSI_4H_OB = 60.0
                RSI_4H_OS = 40.0
                if rsi_1h > RSI_OB:
                    if rsi_4h > RSI_4H_OB and macd_h < macd_h_prev:
                        sig_dir = DIR_SHORT
                        sig_strength = min((rsi_1h - RSI_OB) / 20.0 + 0.5, 1.0)
                elif rsi_1h < RSI_OS:
                    if rsi_4h < RSI_4H_OS and macd_h > macd_h_prev:
                        sig_dir = DIR_LONG
                        sig_strength = min((RSI_OS - rsi_1h) / 20.0 + 0.5, 1.0)

        # ── Consecutive confirmation ──────────────────────────────
        if sig_dir != DIR_NEUTRAL:
            if sig_dir == prev_signal_dir:
                signal_streak += 1
            else:
                prev_signal_dir = sig_dir
                signal_streak = 1
        else:
            prev_signal_dir = DIR_NEUTRAL
            signal_streak = 0

        # ── EMA200 trend filter ───────────────────────────────────
        if sig_dir != DIR_NEUTRAL:
            ema200_1d = sig_1d_ema200[h1d_idx] if 0 <= h1d_idx < len(sig_1d_ema200) else price
            ema200_dist = (price - ema200_1d) / ema200_1d * 100.0 if ema200_1d > 0 else 0.0
            if ema200_filter_pct > 0:
                if ema200_dist < -ema200_filter_pct and sig_dir == DIR_LONG:
                    sig_dir = DIR_NEUTRAL
                elif ema200_dist > ema200_filter_pct and sig_dir == DIR_SHORT:
                    sig_dir = DIR_NEUTRAL

        # ── TF weight adjustment ──────────────────────────────────
        if sig_dir != DIR_NEUTRAL:
            tf_score = tf_weight_1h
            if 0 <= h4_idx < len(sig_4h_ema50):
                lb = min(3, h4_idx)
                if lb > 0:
                    slope = sig_4h_ema50[h4_idx] - sig_4h_ema50[h4_idx - lb]
                    if (sig_dir == DIR_LONG and slope > 0) or (sig_dir == DIR_SHORT and slope < 0):
                        tf_score += tf_weight_4h
            if 0 <= h1d_idx < len(sig_1d_ema50):
                lb = min(3, h1d_idx)
                if lb > 0:
                    slope = sig_1d_ema50[h1d_idx] - sig_1d_ema50[h1d_idx - lb]
                    if (sig_dir == DIR_LONG and slope > 0) or (sig_dir == DIR_SHORT and slope < 0):
                        tf_score += tf_weight_1d
            total_tf = tf_weight_1h + tf_weight_4h + tf_weight_1d
            if total_tf > 0:
                sig_strength *= tf_score / total_tf

        # ── Entry logic ───────────────────────────────────────────
        # Re-count active positions (some may have been closed above)
        num_active = 0
        for j in range(MAX_POSITIONS):
            if pos_active[j]:
                num_active += 1

        if (
            sig_dir != DIR_NEUTRAL
            and sig_strength >= signal_strength_min
            and signal_streak >= consecutive_confirms
            and num_active < max_open
        ):
            signals_generated += 1

            # Fill rate model
            if i + 1 < n:
                if sig_dir == DIR_LONG and lows[i + 1] > price:
                    equity_curve[eq_idx] = current_equity
                    continue
                elif sig_dir == DIR_SHORT and highs[i + 1] < price:
                    equity_curve[eq_idx] = current_equity
                    continue

            signals_filled += 1

            # Compute ATR-based stops
            atr_for_stops = atr_1h_series[h1_idx] if 0 <= h1_idx < len(atr_1h_series) else 0.0
            if atr_for_stops <= 0:
                # Fallback stops
                if sig_dir == DIR_SHORT:
                    sl = price * 1.02
                    tp = price * 0.97
                else:
                    sl = price * 0.98
                    tp = price * 1.03
            else:
                if sig_dir == DIR_LONG:
                    sl = price - atr_multiplier * atr_for_stops
                    risk = price - sl
                    fee_comp = price * ((MAKER_FEE_PCT + TAKER_FEE_PCT) * 100.0 / 100.0)
                    tp = price + rr_ratio * risk + fee_comp
                else:
                    sl = price + atr_multiplier * atr_for_stops
                    risk = sl - price
                    fee_comp = price * ((MAKER_FEE_PCT + TAKER_FEE_PCT) * 100.0 / 100.0)
                    tp = price - rr_ratio * risk - fee_comp

            # Fee gate
            tp_dist_pct = abs(tp - price) / price * 100.0 if price > 0 else 0.0

            # Position sizing (inline fixed_fractional_size)
            equity = current_equity
            stop_dist_pct = max(0.1, atr_multiplier * atr_pct)

            # Median ATR from history
            if atr_history_len >= 5:
                # Simple median via sorting a copy
                temp = np.empty(atr_history_len, dtype=numba.float64)
                actual_len = min(atr_history_len, 200)
                for k in range(actual_len):
                    temp[k] = atr_history[k]
                temp_sorted = np.sort(temp[:actual_len])
                median_atr = temp_sorted[actual_len // 2]
            else:
                median_atr = max(atr_pct, 0.1)

            if equity > 0 and stop_dist_pct > 0 and median_atr > 0:
                vol_scale = max(0.5, min(2.0, atr_pct / median_atr))
                risk_eur = equity * (base_risk_pct / 100.0) * vol_scale
                size_eur = risk_eur / (stop_dist_pct / 100.0)
                size_eur = min(size_eur, equity * 0.30)
                size_eur = max(size_eur, 10.0)
            else:
                size_eur = 10.0

            # Fee gate check
            entry_fee_est = size_eur * MAKER_FEE_PCT
            exit_fee_est = size_eur * TAKER_FEE_PCT
            total_fees_est = entry_fee_est + exit_fee_est
            expected_profit = size_eur * (tp_dist_pct / 100.0)
            fee_ratio = expected_profit / total_fees_est if total_fees_est > 0 else 999.0
            if fee_ratio < min_profit_multiple:
                equity_curve[eq_idx] = current_equity
                continue

            # Store ATR history
            if atr_pct > 0:
                if atr_history_len < 200:
                    atr_history[atr_history_len] = atr_pct
                    atr_history_len += 1
                else:
                    atr_history[atr_history_idx % 200] = atr_pct
                atr_history_idx += 1

            # Confidence scaling
            if confidence_size_scaling > 0:
                size_eur *= max(0.2, 1.0 + (sig_strength - 0.5) * confidence_size_scaling)

            # Cap position
            max_pos = initial_capital * max_position_pct
            size_eur = min(size_eur, max_pos)

            if size_eur < 10.0 or size_eur > balance:
                equity_curve[eq_idx] = current_equity
                continue

            # Entry fee and cost
            entry_fee = size_eur * MAKER_FEE_PCT
            cost = size_eur + entry_fee
            if cost > balance:
                equity_curve[eq_idx] = current_equity
                continue

            balance -= cost
            total_fees_paid += entry_fee

            # Drawdown scaling
            drawdown_pct = ((peak_balance - current_equity) / peak_balance * 100.0
                            if peak_balance > 0 else 0.0)
            if drawdown_pct >= drawdown_scale_pct:
                size_eur *= 0.5

            # Apply slippage to entry
            if sig_dir == DIR_SHORT:
                slipped_entry = price * (1.0 - slippage_pct)
            else:
                slipped_entry = price * (1.0 + slippage_pct)

            # Find free position slot
            slot = -1
            for j in range(MAX_POSITIONS):
                if not pos_active[j]:
                    slot = j
                    break
            if slot < 0:
                # No free slot (shouldn't happen with max_open check)
                equity_curve[eq_idx] = current_equity
                continue

            pos_active[slot] = True
            pos_dir[slot] = sig_dir
            pos_entry[slot] = slipped_entry
            pos_size[slot] = size_eur
            pos_sl[slot] = sl
            pos_tp[slot] = tp
            pos_peak[slot] = slipped_entry
            pos_entry_bar[slot] = i
            pos_entry_fee[slot] = entry_fee
            pos_strategy[slot] = strategy_id
            pos_tp_shifted[slot] = False
            pos_range_mid[slot] = 0.0
            pos_range_upper[slot] = 0.0
            pos_range_lower[slot] = 0.0

            # Range strategy: override stops with band-based stops
            if strategy_id == STRAT_RANGE:
                bb_lower = sig_1h_bb_lower[h1_idx] if h1_idx < len(sig_1h_bb_lower) else price
                bb_upper = sig_1h_bb_upper[h1_idx] if h1_idx < len(sig_1h_bb_upper) else price
                bb_mid = sig_1h_bb_mid[h1_idx] if h1_idx < len(sig_1h_bb_mid) else price
                bandwidth = bb_upper - bb_lower
                pos_range_mid[slot] = bb_mid
                pos_range_upper[slot] = bb_upper
                pos_range_lower[slot] = bb_lower
                if sig_dir == DIR_LONG:
                    pos_sl[slot] = bb_lower - 0.5 * bandwidth
                    pos_tp[slot] = bb_mid
                else:
                    pos_sl[slot] = bb_upper + 0.5 * bandwidth
                    pos_tp[slot] = bb_mid

        equity_curve[eq_idx] = current_equity

    # ==================================================================
    # CLOSE REMAINING POSITIONS
    # ==================================================================
    if n > 0:
        last_price = closes[n - 1]
        for j in range(MAX_POSITIONS):
            if pos_active[j]:
                d = pos_dir[j]
                if d == DIR_SHORT:
                    slipped = last_price * (1.0 + slippage_pct)
                    pchange = (pos_entry[j] - slipped) / pos_entry[j]
                else:
                    slipped = last_price * (1.0 - slippage_pct)
                    pchange = (slipped - pos_entry[j]) / pos_entry[j]
                gross = pos_size[j] * pchange
                qty = pos_size[j] / pos_entry[j]
                exit_fee = slipped * qty * TAKER_FEE_PCT
                total_fees_paid += exit_fee
                pnl = gross - exit_fee - pos_entry_fee[j]
                balance += pos_size[j] + pnl
                pos_active[j] = False
                if num_trades < MAX_TRADES:
                    trade_dir[num_trades] = d
                    trade_entry_price[num_trades] = pos_entry[j]
                    trade_exit_price[num_trades] = last_price
                    trade_size[num_trades] = pos_size[j]
                    trade_pnl[num_trades] = pnl
                    trade_pnl_pct[num_trades] = (pnl / pos_size[j]) * 100.0 if pos_size[j] > 0 else 0.0
                    trade_reason[num_trades] = EXIT_END_OF_DATA
                    trade_strategy[num_trades] = pos_strategy[j]
                    num_trades += 1

    return (
        equity_curve,
        trade_pnl[:num_trades].copy(),
        trade_pnl_pct[:num_trades].copy(),
        trade_size[:num_trades].copy(),
        trade_dir[:num_trades].copy(),
        trade_reason[:num_trades].copy(),
        trade_entry_price[:num_trades].copy(),
        trade_exit_price[:num_trades].copy(),
        trade_strategy[:num_trades].copy(),
        total_fees_paid,
        signals_generated,
        signals_filled,
        quiet_hours_skipped,
    )


# ---------------------------------------------------------------------------
# Strategy name mapping
# ---------------------------------------------------------------------------
STRATEGY_NAME_TO_ID = {
    "orderflow": STRAT_ORDERFLOW,
    "range": STRAT_RANGE,
    "squeeze": STRAT_SQUEEZE,
    "funding_contrarian": STRAT_FUNDING,
}

EXIT_REASON_NAMES = {
    EXIT_STOP_LOSS: "stop_loss",
    EXIT_TAKE_PROFIT: "take_profit",
    EXIT_TIME: "time_exit",
    EXIT_END_OF_DATA: "end_of_data",
    EXIT_RANGE_MID: "range_mid_exit",
}
