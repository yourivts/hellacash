"""Benchmark comparison — bot performance vs buy-and-hold."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkEntry:
    symbol: str
    start_price: float
    end_price: float
    buy_hold_return_pct: float
    buy_hold_pnl_eur: float


@dataclass
class EquityCurvePoint:
    timestamp: datetime
    bot_equity: float
    btc_equity: float
    eth_equity: float


@dataclass
class BenchmarkReport:
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    initial_capital: float = 0.0
    bot_total_pnl: float = 0.0
    bot_return_pct: float = 0.0
    bot_sharpe: float = 0.0
    bot_max_drawdown_pct: float = 0.0
    btc_benchmark: Optional[BenchmarkEntry] = None
    eth_benchmark: Optional[BenchmarkEntry] = None
    per_asset_benchmarks: List[BenchmarkEntry] = field(default_factory=list)
    alpha_vs_btc: float = 0.0
    alpha_vs_eth: float = 0.0
    equity_curve: List[EquityCurvePoint] = field(default_factory=list)


def compute_buy_hold_return(
    symbol: str, start_price: float, end_price: float, initial_capital: float,
) -> BenchmarkEntry:
    if start_price <= 0:
        return BenchmarkEntry(symbol, start_price, end_price, 0.0, 0.0)
    ret_pct = (end_price - start_price) / start_price * 100
    pnl = initial_capital * ret_pct / 100
    return BenchmarkEntry(symbol, start_price, end_price, round(ret_pct, 4), round(pnl, 4))


class BenchmarkEngine:
    """Computes benchmark comparison reports from portfolio and price data."""

    def compute(
        self,
        bot_pnl: float,
        bot_return_pct: float,
        bot_sharpe: float,
        bot_max_dd: float,
        initial_capital: float,
        btc_start: float,
        btc_end: float,
        eth_start: float,
        eth_end: float,
        per_asset_prices: Optional[Dict[str, tuple]] = None,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> BenchmarkReport:
        btc = compute_buy_hold_return("BTC-EUR", btc_start, btc_end, initial_capital)
        eth = compute_buy_hold_return("ETH-EUR", eth_start, eth_end, initial_capital)

        per_asset = []
        if per_asset_prices:
            for sym, (sp, ep) in per_asset_prices.items():
                per_asset.append(compute_buy_hold_return(sym, sp, ep, initial_capital))

        return BenchmarkReport(
            period_start=period_start,
            period_end=period_end,
            initial_capital=initial_capital,
            bot_total_pnl=bot_pnl,
            bot_return_pct=bot_return_pct,
            bot_sharpe=bot_sharpe,
            bot_max_drawdown_pct=bot_max_dd,
            btc_benchmark=btc,
            eth_benchmark=eth,
            per_asset_benchmarks=per_asset,
            alpha_vs_btc=bot_return_pct - btc.buy_hold_return_pct,
            alpha_vs_eth=bot_return_pct - eth.buy_hold_return_pct,
        )
