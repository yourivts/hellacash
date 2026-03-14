"""Prometheus metrics for HellaCash."""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import APIRouter, Response

from bot.events.bus import (
    TOPIC_SIGNAL,
    TOPIC_TRADE_CLOSED,
    get_bus,
)

logger = logging.getLogger(__name__)

# ── Gauges ───────────────────────────────────────────────────────────────────
equity_total = Gauge("hellacash_equity_total", "Total portfolio equity in EUR")
drawdown_pct = Gauge("hellacash_drawdown_pct", "Current portfolio drawdown percentage")
open_positions = Gauge("hellacash_open_positions", "Number of open positions")
win_rate = Gauge("hellacash_win_rate", "Current win rate (0-1)")

# ── Counters ─────────────────────────────────────────────────────────────────
trades_total = Counter("hellacash_trades_total", "Total closed trades", ["direction", "exit_reason"])
signals_generated = Counter("hellacash_signals_generated", "Total signals generated", ["direction"])

router = APIRouter()


@router.get("/metrics", include_in_schema=False)
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


async def start_metrics_relay() -> None:
    """Subscribe to event bus topics and update Prometheus metrics."""
    bus = get_bus()

    trade_q = await bus.subscribe(TOPIC_TRADE_CLOSED)
    signal_q = await bus.subscribe(TOPIC_SIGNAL)

    async def _relay():
        while True:
            try:
                event = trade_q.get_nowait()
                p = event.payload
                trades_total.labels(
                    direction=p.get("direction", "LONG"),
                    exit_reason=p.get("exit_reason", "unknown"),
                ).inc()
            except asyncio.QueueEmpty:
                pass
            try:
                event = signal_q.get_nowait()
                p = event.payload
                signals_generated.labels(direction=p.get("direction", "NEUTRAL")).inc()
            except asyncio.QueueEmpty:
                pass
            await asyncio.sleep(0.1)

    asyncio.create_task(_relay())


def update_portfolio_metrics(
    equity: float,
    dd_pct: float,
    positions: int,
    wr: Optional[float],
) -> None:
    """Called periodically to update gauge values."""
    equity_total.set(equity)
    drawdown_pct.set(dd_pct)
    open_positions.set(positions)
    if wr is not None:
        win_rate.set(wr)
