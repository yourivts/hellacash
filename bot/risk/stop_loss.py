"""ATR-based stop-loss and trailing stop calculations."""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def initial_stops(
    entry_price: float,
    df=None,
    atr_multiplier: float = 3.0,
    rr_ratio: float = 2.0,
    direction: str = "LONG",
    total_fee_pct: float = 0.0,
    atr_value: float = 0.0,
) -> tuple:
    """Calculate initial stop-loss and take-profit levels.

    Returns (stop_loss, take_profit) tuple.
    If atr_value is provided, uses it directly. Otherwise computes from df.
    total_fee_pct: added to TP distance so R:R is net of fees.
    """
    if atr_value <= 0 and df is not None:
        from bot.indicators.volatility import atr as compute_atr
        if len(df) < 15:
            if direction == "SHORT":
                return entry_price * 1.02, entry_price * 0.97
            else:
                return entry_price * 0.98, entry_price * 1.03
        atr_series = compute_atr(df["high"], df["low"], df["close"])
        atr_value = float(atr_series.iloc[-1])

    if atr_value <= 0:
        # Absolute fallback
        if direction == "SHORT":
            return entry_price * 1.02, entry_price * 0.97
        else:
            return entry_price * 0.98, entry_price * 1.03

    if direction == "LONG":
        stop_loss = entry_price - atr_multiplier * atr_value
        risk = entry_price - stop_loss
        fee_compensation = entry_price * (total_fee_pct / 100.0)
        take_profit = entry_price + rr_ratio * risk + fee_compensation
    else:
        stop_loss = entry_price + atr_multiplier * atr_value
        risk = stop_loss - entry_price
        fee_compensation = entry_price * (total_fee_pct / 100.0)
        take_profit = entry_price - rr_ratio * risk - fee_compensation

    logger.debug("StopLoss: %s entry=%.4f ATR=%.4f stop=%.4f tp=%.4f",
                 direction, entry_price, atr_value, stop_loss, take_profit)
    return round(stop_loss, 8), round(take_profit, 8)


def trail_stop(
    current_price: float,
    highest_price: float,
    current_stop: float,
    atr_value: float,
    direction: str = "LONG",
    activation_multiplier: float = 1.0,
    activation_threshold: float = 1.5,
    entry_price: float = 0.0,
) -> float:
    """Trailing stop with activation threshold.

    Stop only starts trailing after price moves activation_threshold * stop_distance
    in profit direction. Before that, original stop is maintained.
    """
    if entry_price > 0 and activation_threshold > 0:
        stop_distance = abs(entry_price - current_stop)
        if direction == "LONG":
            profit = highest_price - entry_price
        else:
            profit = entry_price - highest_price

        if stop_distance > 0 and profit < stop_distance * activation_threshold:
            return current_stop

    trail_dist = atr_value * activation_multiplier
    if direction == "LONG":
        new_stop = highest_price - trail_dist
        return max(current_stop, new_stop)
    else:
        new_stop = highest_price + trail_dist
        return min(current_stop, new_stop) if current_stop > 0 else new_stop


def check_stop_triggered(
    current_price: float,
    stop_loss: float,
    take_profit: float,
    direction: str = "LONG",
) -> Optional[str]:
    """
    Returns 'stop_loss', 'take_profit', or None.
    """
    if direction == "SHORT":
        if current_price >= stop_loss:
            return "stop_loss"
        if current_price <= take_profit:
            return "take_profit"
    else:
        if current_price <= stop_loss:
            return "stop_loss"
        if current_price >= take_profit:
            return "take_profit"
    return None
