"""ATR-based stop-loss and trailing stop calculations."""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from bot.indicators.volatility import atr as compute_atr

logger = logging.getLogger(__name__)


def initial_stops(
    entry_price: float,
    df: pd.DataFrame,
    atr_multiplier: float = 2.0,
    rr_ratio: float = 3.0,
) -> tuple[float, float]:
    """
    Compute initial stop-loss and take-profit prices.

    Returns:
        (stop_loss_price, take_profit_price)
    """
    if len(df) < 15:
        # Fallback: 2% stop, 3% take-profit
        stop = entry_price * 0.98
        tp = entry_price * 1.03
        return stop, tp

    a = compute_atr(df["high"], df["low"], df["close"]).iloc[-1]
    stop = entry_price - atr_multiplier * a
    risk = entry_price - stop
    tp = entry_price + rr_ratio * risk
    logger.debug("StopLoss: entry=%.4f ATR=%.4f stop=%.4f tp=%.4f", entry_price, a, stop, tp)
    return stop, tp


def trail_stop(
    current_price: float,
    highest_price: float,
    current_stop: float,
    atr_value: float,
    activation_multiplier: float = 1.0,
) -> float:
    """
    Update trailing stop: only moves up, never down.
    Activates when price is > entry + 1×ATR above the stop.

    Returns new stop-loss price (may be same as current_stop).
    """
    new_stop = highest_price - atr_value
    if new_stop > current_stop:
        return new_stop
    return current_stop


def check_stop_triggered(
    current_price: float,
    stop_loss: float,
    take_profit: float,
) -> Optional[str]:
    """
    Returns 'stop_loss', 'take_profit', or None.
    """
    if current_price <= stop_loss:
        return "stop_loss"
    if current_price >= take_profit:
        return "take_profit"
    return None
