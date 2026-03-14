"""Bitvavo fee schedule — aligned with live fee tiers (March 2026).

Categories (from Bitvavo API `feeCategory` field):
    A  — Standard EUR markets (421 markets)
    B  — Stablecoin pairs: EURCV, EURC, USDC, EUROP, USDCV (5 markets)
    C  — Major crypto: ADA, SOL, XRP, ETH (4 markets)

Short selling adds:
    - Same maker/taker trading fee on open and close
    - Hourly borrowing fee (annualised rate / 8760), settled on close
    - 2% liquidation fee if auto-liquidated
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Volume-based fee tiers ────────────────────────────────────────────────────
# (max_volume_eur, maker_pct, taker_pct)
# max_volume_eur = None means "and above"

CATEGORY_A_TIERS: list[Tuple[Optional[float], float, float]] = [
    (100_000,       0.15, 0.25),
    (250_000,       0.10, 0.20),
    (500_000,       0.08, 0.16),
    (1_000_000,     0.06, 0.12),
    (2_500_000,     0.05, 0.10),
    (5_000_000,     0.04, 0.08),
    (10_000_000,    0.04, 0.06),
    (25_000_000,    0.00, 0.05),
    (100_000_000,   0.00, 0.02),
    (None,          0.00, 0.01),
]

# Category B — stablecoin pairs (flat reduced fees across all tiers)
CATEGORY_B_TIERS: list[Tuple[Optional[float], float, float]] = [
    (None, 0.05, 0.05),
]

# Category C — major crypto (same tiers as A for now, Bitvavo may adjust)
CATEGORY_C_TIERS = CATEGORY_A_TIERS

FEE_TIERS: Dict[str, list] = {
    "A": CATEGORY_A_TIERS,
    "B": CATEGORY_B_TIERS,
    "C": CATEGORY_C_TIERS,
}

# ── Short selling fees ────────────────────────────────────────────────────────

# Default annual borrowing rate (conservative estimate, actual varies per asset)
DEFAULT_BORROW_RATE_ANNUAL = 0.03  # 3% per year
LIQUIDATION_FEE_PCT = 2.0          # 2% if auto-liquidated

# Per-asset overrides (base symbol → annual rate).
# These are approximate — Bitvavo shows live rates in the app.
BORROW_RATES: Dict[str, float] = {
    "BTC":  0.01,   # ~1% p.a.
    "ETH":  0.015,  # ~1.5% p.a.
    "XRP":  0.03,
    "SOL":  0.03,
    "ADA":  0.03,
    "DOGE": 0.05,
    "DOT":  0.04,
    "LINK": 0.04,
    "AVAX": 0.04,
    "MATIC": 0.04,
}

# ── Market fee category cache (populated at startup) ─────────────────────────

_market_categories: Dict[str, str] = {}


def set_market_categories(categories: Dict[str, str]) -> None:
    """Called at startup with {symbol: feeCategory} from Bitvavo markets API."""
    global _market_categories
    _market_categories = dict(categories)
    logger.info("Loaded fee categories for %d markets", len(_market_categories))


def get_fee_category(symbol: str) -> str:
    return _market_categories.get(symbol, "A")


def get_trading_fees(
    symbol: str,
    volume_30d_eur: float = 0.0,
) -> Tuple[float, float]:
    """Return (maker_pct, taker_pct) for a symbol at a given volume tier.

    Returns percentages as decimals (e.g. 0.25 means 0.25%).
    """
    category = get_fee_category(symbol)
    tiers = FEE_TIERS.get(category, CATEGORY_A_TIERS)

    for max_vol, maker, taker in tiers:
        if max_vol is None or volume_30d_eur < max_vol:
            return maker, taker

    # Fallback to last tier
    return tiers[-1][1], tiers[-1][2]


def get_taker_fee(symbol: str, volume_30d_eur: float = 0.0) -> float:
    """Taker fee as a fraction (e.g. 0.0025 for 0.25%)."""
    _, taker_pct = get_trading_fees(symbol, volume_30d_eur)
    return taker_pct / 100.0


def get_maker_fee(symbol: str, volume_30d_eur: float = 0.0) -> float:
    """Maker fee as a fraction (e.g. 0.0015 for 0.15%)."""
    maker_pct, _ = get_trading_fees(symbol, volume_30d_eur)
    return maker_pct / 100.0


def get_borrow_rate_hourly(symbol: str) -> float:
    """Hourly borrowing cost as a fraction of borrowed value."""
    base = symbol.split("-")[0].upper()
    annual = BORROW_RATES.get(base, DEFAULT_BORROW_RATE_ANNUAL)
    return annual / 8760.0  # hours in a year


def compute_trade_fees(
    symbol: str,
    entry_price: float,
    exit_price: float,
    quantity: float,
    direction: str = "LONG",
    hold_hours: float = 0.0,
    volume_30d_eur: float = 0.0,
) -> float:
    """Compute total fees for a round-trip trade.

    Includes:
        - Taker fee on entry (market orders)
        - Taker fee on exit (market orders)
        - Borrowing fee for SHORT positions (accrued hourly)
    """
    taker_frac = get_taker_fee(symbol, volume_30d_eur)

    entry_fee = entry_price * quantity * taker_frac
    exit_fee = exit_price * quantity * taker_frac
    total = entry_fee + exit_fee

    if direction == "SHORT" and hold_hours > 0:
        hourly_rate = get_borrow_rate_hourly(symbol)
        borrow_fee = entry_price * quantity * hourly_rate * hold_hours
        total += borrow_fee

    return total
