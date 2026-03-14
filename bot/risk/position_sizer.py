"""Position sizing using quarter-Kelly criterion."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def kelly_size(
    win_rate: float,
    avg_win_pct: float,
    avg_loss_pct: float,
    portfolio_eur: float,
    current_drawdown_pct: float = 0.0,
    max_position_pct: float = 20.0,
    kelly_fraction: float = 0.25,
) -> float:
    """
    Calculate position size in EUR using quarter-Kelly criterion.

    Args:
        win_rate: Historical win rate (0–1). Uses 0.5 if no history.
        avg_win_pct: Average winning trade return as fraction (e.g. 0.03 = 3%).
        avg_loss_pct: Average losing trade loss as fraction (positive number).
        portfolio_eur: Total portfolio value in EUR.
        current_drawdown_pct: Current portfolio drawdown % (0–100).
        max_position_pct: Hard cap on single position as % of portfolio.
        kelly_fraction: Fraction of full Kelly to use (default 0.25 = quarter-Kelly).

    Returns:
        Position size in EUR.
    """
    if win_rate <= 0 or avg_loss_pct <= 0:
        # No history — use a conservative 1% of portfolio
        base_pct = 1.0
    else:
        loss_rate = 1 - win_rate
        # Kelly formula: f = (W * b - L) / b  where b = avg_win / avg_loss
        b = avg_win_pct / max(avg_loss_pct, 0.001)
        full_kelly = (win_rate * b - loss_rate) / b
        full_kelly = max(0.0, min(full_kelly, 1.0))
        base_pct = full_kelly * kelly_fraction * 100.0

    # Reduce size when drawdown is elevated
    if current_drawdown_pct > 3.0:
        scale = max(0.3, 1.0 - (current_drawdown_pct - 3.0) / 10.0)
        base_pct *= scale

    # Hard cap
    capped_pct = max(0.0, min(base_pct, max_position_pct))
    size_eur = portfolio_eur * (capped_pct / 100.0)

    logger.debug(
        "PositionSizer: win_rate=%.2f avg_win=%.3f avg_loss=%.3f → %.1f%% = €%.2f",
        win_rate, avg_win_pct, avg_loss_pct, capped_pct, size_eur,
    )
    return size_eur
