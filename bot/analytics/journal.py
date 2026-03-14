"""Trade journaling — auto-generated reasoning from indicator snapshots."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def generate_entry_reasoning(
    symbol: str,
    direction: str,
    strategy: str,
    regime: str,
    final_score: float,
    threshold: float,
    mtf_scores: Dict[str, float],
    mtf_agreement: float,
    technical_score: float,
    sentiment_score: float,
    onchain_score: float = 0.0,
    orderbook_imbalance: float = 0.0,
    indicator_values: Optional[Dict[str, Any]] = None,
) -> str:
    parts = []
    n_agree = sum(1 for s in mtf_scores.values() if s * final_score > 0)
    n_total = len(mtf_scores)

    parts.append(
        f"{direction} {symbol} via {strategy} in {regime.upper()} regime."
    )

    tf_detail = ", ".join(f"{tf}: {s:+.2f}" for tf, s in mtf_scores.items())
    parts.append(
        f"MTF agreement: {n_agree}/{n_total} timeframes "
        f"({'bullish' if final_score > 0 else 'bearish'}) ({tf_detail})."
    )

    parts.append(f"Technical composite: {technical_score:+.2f}")

    if indicator_values:
        indicators = []
        if "rsi" in indicator_values:
            rsi = indicator_values["rsi"]
            label = "oversold" if rsi < 30 else "overbought" if rsi > 70 else ""
            indicators.append(f"RSI {rsi:.0f}{' ' + label if label else ''}")
        if "macd_histogram" in indicator_values:
            h = indicator_values["macd_histogram"]
            indicators.append(f"MACD {'rising' if h > 0 else 'falling'}")
        if indicators:
            parts[-1] += f" ({', '.join(indicators)})"
    parts[-1] += "."

    parts.append(f"Sentiment: {sentiment_score:+.2f}.")

    if onchain_score != 0:
        parts.append(f"On-chain: {onchain_score:+.2f}.")
    if orderbook_imbalance != 0:
        parts.append(f"Order book: {orderbook_imbalance:+.2f}.")

    parts.append(f"Final score: {final_score:+.2f} vs threshold {threshold:.2f}.")

    return " ".join(parts)


def generate_exit_reasoning(
    exit_reason: str,
    hold_seconds: int,
    entry_price: float,
    exit_price: float,
    net_pnl: float,
    roi_pct: float,
    indicator_values: Optional[Dict[str, Any]] = None,
) -> str:
    hours = hold_seconds // 3600
    mins = (hold_seconds % 3600) // 60
    hold_str = f"{hours}h {mins}m" if hours > 0 else f"{mins}m"

    reason_label = exit_reason.replace("_", " ")
    parts = [
        f"Exited via {reason_label} at €{exit_price:,.2f}.",
        f"Held {hold_str}.",
    ]

    if indicator_values and "rsi" in indicator_values:
        parts.append(f"RSI at exit: {indicator_values['rsi']:.0f}.")

    sign = "+" if net_pnl >= 0 else ""
    parts.append(f"Net P&L: {sign}€{net_pnl:.2f} ({sign}{roi_pct:.1f}%).")

    return " ".join(parts)
