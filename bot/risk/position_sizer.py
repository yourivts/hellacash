"""Position sizing: fixed fractional with volatility scaling."""
from __future__ import annotations

MIN_POSITION_EUR = 10.0
MAX_POSITION_PCT = 0.30  # 30% of equity


def fixed_fractional_size(
    equity: float,
    base_risk_pct: float,
    stop_distance_pct: float,
    atr_pct: float,
    median_atr_pct: float,
) -> float:
    """Compute position size using fixed fractional with volatility scaling.

    Args:
        equity: Current portfolio value in EUR.
        base_risk_pct: Percentage of equity to risk per trade (e.g., 3.0).
        stop_distance_pct: Stop distance as % of price (e.g., 2.0 for 2%).
        atr_pct: Current ATR as % of price.
        median_atr_pct: Median ATR% over lookback period.

    Returns:
        Position size in EUR.
    """
    if equity <= 0 or stop_distance_pct <= 0 or median_atr_pct <= 0:
        return MIN_POSITION_EUR

    # Volatility scaling: bigger positions in volatile markets
    vol_scale = max(0.5, min(2.0, atr_pct / median_atr_pct))

    risk_eur = equity * (base_risk_pct / 100.0) * vol_scale
    position_size = risk_eur / (stop_distance_pct / 100.0)

    # Hard caps
    position_size = min(position_size, equity * MAX_POSITION_PCT)
    position_size = max(position_size, MIN_POSITION_EUR)

    return round(position_size, 2)
