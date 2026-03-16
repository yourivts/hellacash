"""Fee-aware entry gate: rejects trades where expected profit < N * total fees."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FeeGateResult:
    approved: bool
    fee_ratio: float
    total_fees: float
    expected_profit: float
    reason: str = ""


def check_fee_gate(
    position_size: float,
    tp_distance_pct: float,
    maker_fee_pct: float = 0.15,
    taker_fee_pct: float = 0.25,
    min_profit_multiple: float = 3.0,
    is_short: bool = False,
    expected_hold_hours: float = 0,
    annual_borrow_rate: float = 0.03,
) -> FeeGateResult:
    if position_size <= 0 or tp_distance_pct <= 0:
        return FeeGateResult(
            approved=False, fee_ratio=0.0, total_fees=0.0,
            expected_profit=0.0, reason="invalid position or TP"
        )

    entry_fee = position_size * (maker_fee_pct / 100.0)
    exit_fee = position_size * (taker_fee_pct / 100.0)
    total_fees = entry_fee + exit_fee

    if is_short and expected_hold_hours > 0:
        hourly_rate = annual_borrow_rate / 8760.0
        borrow_fee = position_size * hourly_rate * expected_hold_hours
        total_fees += borrow_fee

    expected_profit = position_size * (tp_distance_pct / 100.0)
    fee_ratio = expected_profit / total_fees if total_fees > 0 else 999.0

    approved = fee_ratio >= min_profit_multiple
    reason = "" if approved else f"fee_ratio {fee_ratio:.2f} < {min_profit_multiple}"

    return FeeGateResult(
        approved=approved,
        fee_ratio=round(fee_ratio, 2),
        total_fees=round(total_fees, 4),
        expected_profit=round(expected_profit, 4),
        reason=reason,
    )
