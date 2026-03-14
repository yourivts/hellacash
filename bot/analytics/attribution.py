"""Performance attribution — P&L breakdown by strategy, regime, session, direction."""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class StrategyAttribution:
    strategy_name: str
    total_pnl: float = 0.0
    trade_count: int = 0
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    pnl_contribution_pct: float = 0.0


@dataclass
class AttributionReport:
    total_pnl: float = 0.0
    total_trades: int = 0
    overall_win_rate: float = 0.0
    by_strategy: List[StrategyAttribution] = field(default_factory=list)
    by_regime: List[StrategyAttribution] = field(default_factory=list)
    by_session: List[StrategyAttribution] = field(default_factory=list)
    by_direction: List[StrategyAttribution] = field(default_factory=list)
    best_strategy: str = ""
    worst_strategy: str = ""
    best_session: str = ""
    worst_session: str = ""


SESSION_HOURS = {
    "Asia": range(0, 8),
    "Europe": range(8, 14),
    "US": range(14, 21),
    "Overlap/Off-hours": range(21, 24),
}


def _classify_session(dt: datetime) -> str:
    h = dt.hour
    for name, hours in SESSION_HOURS.items():
        if h in hours:
            return name
    return "Overlap/Off-hours"


def _group_attribution(trades: list, key_fn) -> List[StrategyAttribution]:
    groups: Dict[str, list] = defaultdict(list)
    for t in trades:
        groups[key_fn(t)].append(t)

    total_pnl = sum(t.net_pnl for t in trades) or 1.0
    results = []
    for name, group in sorted(groups.items()):
        pnl = sum(t.net_pnl for t in group)
        wins = sum(1 for t in group if t.net_pnl > 0)
        results.append(StrategyAttribution(
            strategy_name=name,
            total_pnl=pnl,
            trade_count=len(group),
            win_rate=wins / len(group) if group else 0,
            avg_pnl=pnl / len(group) if group else 0,
            pnl_contribution_pct=pnl / total_pnl * 100 if total_pnl else 0,
        ))
    return results


class AttributionEngine:
    def compute_from_trades(self, trades: list) -> AttributionReport:
        if not trades:
            return AttributionReport()

        total_pnl = sum(t.net_pnl for t in trades)
        wins = sum(1 for t in trades if t.net_pnl > 0)

        by_strategy = _group_attribution(trades, lambda t: t.strategy_name)
        by_regime = _group_attribution(
            trades, lambda t: getattr(t, "market_regime", None) or "unknown"
        )
        by_session = _group_attribution(trades, lambda t: _classify_session(t.created_at))
        by_direction = _group_attribution(trades, lambda t: t.direction)

        best_s = max(by_strategy, key=lambda s: s.total_pnl) if by_strategy else None
        worst_s = min(by_strategy, key=lambda s: s.total_pnl) if by_strategy else None
        best_sess = max(by_session, key=lambda s: s.total_pnl) if by_session else None
        worst_sess = min(by_session, key=lambda s: s.total_pnl) if by_session else None

        return AttributionReport(
            total_pnl=total_pnl,
            total_trades=len(trades),
            overall_win_rate=wins / len(trades) if trades else 0,
            by_strategy=by_strategy,
            by_regime=by_regime,
            by_session=by_session,
            by_direction=by_direction,
            best_strategy=best_s.strategy_name if best_s else "",
            worst_strategy=worst_s.strategy_name if worst_s else "",
            best_session=best_sess.strategy_name if best_sess else "",
            worst_session=worst_sess.strategy_name if worst_sess else "",
        )
