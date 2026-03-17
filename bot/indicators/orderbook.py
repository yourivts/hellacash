"""Order book analysis — bid/ask imbalance, wall detection, support/resistance."""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PriceLevel:
    price: float
    quantity: float
    eur_value: float = 0.0


@dataclass
class DepthBucket:
    price_low: float
    price_high: float
    bid_volume: float = 0.0
    ask_volume: float = 0.0


@dataclass
class DepthAnalysis:
    imbalance: float = 0.0
    spread_pct: float = 0.0
    bid_walls: List[PriceLevel] = field(default_factory=list)
    ask_walls: List[PriceLevel] = field(default_factory=list)
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)
    depth_buckets: List[DepthBucket] = field(default_factory=list)


class _BookState:
    def __init__(self) -> None:
        self.bids: Dict[float, float] = {}
        self.asks: Dict[float, float] = {}
        self.last_update: float = 0.0

    def apply_delta(self, side: str, price: float, qty: float) -> None:
        book = self.bids if side == "bid" else self.asks
        if qty == 0:
            book.pop(price, None)
        else:
            book[price] = qty
        self.last_update = time.time()


class OrderBookProvider:
    """Order book signal provider with depth analysis."""

    name = "orderbook"

    def __init__(self, depth_levels: int = 25) -> None:
        self._books: Dict[str, _BookState] = {}
        self._depth_levels = depth_levels
        self._last_analysis: Dict[str, float] = {}

    def update_level(self, symbol: str, side: str, price: float, qty: float) -> None:
        if symbol not in self._books:
            self._books[symbol] = _BookState()
        self._books[symbol].apply_delta(side, price, qty)

    def set_snapshot(self, symbol: str, bids: list, asks: list) -> None:
        bs = _BookState()
        for price, qty in bids:
            bs.bids[float(price)] = float(qty)
        for price, qty in asks:
            bs.asks[float(price)] = float(qty)
        bs.last_update = time.time()
        self._books[symbol] = bs

    def score(self, symbol: str, **kwargs) -> float:
        if symbol not in self._books:
            return 0.0
        bs = self._books[symbol]
        if not bs.bids or not bs.asks:
            return 0.0
        imbalance = self._compute_imbalance(bs)
        if abs(imbalance) < 0.3:
            return 0.0
        return max(-1.0, min(1.0, imbalance))

    def is_available(self) -> bool:
        return bool(self._books)

    def analyze(self, symbol: str) -> DepthAnalysis:
        if symbol not in self._books:
            return DepthAnalysis()
        bs = self._books[symbol]
        if not bs.bids or not bs.asks:
            return DepthAnalysis()

        imbalance = self._compute_imbalance(bs)
        best_bid = max(bs.bids.keys())
        best_ask = min(bs.asks.keys())
        mid = (best_bid + best_ask) / 2
        spread_pct = (best_ask - best_bid) / mid * 100 if mid > 0 else 0

        # Wall detection
        n = self._depth_levels
        top_bids = sorted(bs.bids.items(), key=lambda x: -x[0])[:n]
        top_asks = sorted(bs.asks.items(), key=lambda x: x[0])[:n]
        all_qtys = [q for _, q in top_bids + top_asks]
        avg_qty = sum(all_qtys) / len(all_qtys) if all_qtys else 1
        threshold = avg_qty * 3

        bid_walls = [PriceLevel(p, q, p * q) for p, q in top_bids if q > threshold]
        ask_walls = [PriceLevel(p, q, p * q) for p, q in top_asks if q > threshold]

        # Support/resistance via price buckets
        bucket_width = mid * 0.005
        support = self._cluster_levels(top_bids, bucket_width, top_n=3)
        resistance = self._cluster_levels(top_asks, bucket_width, top_n=3)

        # Depth buckets for heatmap
        buckets = self._build_depth_buckets(bs, mid, n_buckets=50)

        return DepthAnalysis(
            imbalance=imbalance, spread_pct=spread_pct,
            bid_walls=bid_walls, ask_walls=ask_walls,
            support_levels=support, resistance_levels=resistance,
            depth_buckets=buckets,
        )

    def _compute_imbalance(self, bs: _BookState) -> float:
        n = self._depth_levels
        top_bids = sorted(bs.bids.items(), key=lambda x: -x[0])[:n]
        top_asks = sorted(bs.asks.items(), key=lambda x: x[0])[:n]
        bid_vol = sum(q for _, q in top_bids)
        ask_vol = sum(q for _, q in top_asks)
        total = bid_vol + ask_vol
        if total <= 0:
            return 0.0
        return (bid_vol - ask_vol) / total

    def _cluster_levels(self, levels: list, bucket_width: float, top_n: int = 3) -> List[float]:
        clusters: Dict[float, float] = defaultdict(float)
        for price, qty in levels:
            bucket = round(price / bucket_width) * bucket_width
            clusters[bucket] += qty
        if not clusters:
            return []
        median_vol = sorted(clusters.values())[len(clusters) // 2]
        significant = [(p, v) for p, v in clusters.items() if v > median_vol * 2]
        significant.sort(key=lambda x: -x[1])
        return [p for p, _ in significant[:top_n]]

    def _build_depth_buckets(self, bs: _BookState, mid: float, n_buckets: int = 50) -> List[DepthBucket]:
        spread = mid * 0.10  # ±5%
        low = mid - spread / 2
        step = spread / n_buckets
        buckets = []
        for i in range(n_buckets):
            bl = low + i * step
            bh = bl + step
            bid_v = sum(q for p, q in bs.bids.items() if bl <= p < bh)
            ask_v = sum(q for p, q in bs.asks.items() if bl <= p < bh)
            buckets.append(DepthBucket(bl, bh, bid_v, ask_v))
        return buckets

    async def save_snapshots(self) -> None:
        """Persist current orderbook imbalance data for future backtesting."""
        if not self._books:
            return
        try:
            from bot.data.database import get_session
            from bot.data.models import OrderbookSnapshot

            now = datetime.now(timezone.utc).replace(tzinfo=None)
            async with get_session() as session:
                for symbol, bs in self._books.items():
                    if not bs.bids or not bs.asks:
                        continue
                    imbalance = self._compute_imbalance(bs)
                    best_bid = max(bs.bids.keys())
                    best_ask = min(bs.asks.keys())
                    mid = (best_bid + best_ask) / 2
                    spread_pct = (best_ask - best_bid) / mid * 100 if mid > 0 else 0

                    n = self._depth_levels
                    bid_depth = sum(p * q for p, q in sorted(bs.bids.items(), key=lambda x: -x[0])[:n])
                    ask_depth = sum(p * q for p, q in sorted(bs.asks.items(), key=lambda x: x[0])[:n])

                    snapshot = OrderbookSnapshot(
                        symbol=symbol,
                        imbalance=imbalance,
                        spread_pct=spread_pct,
                        bid_depth_eur=bid_depth,
                        ask_depth_eur=ask_depth,
                        recorded_at=now,
                    )
                    session.add(snapshot)
                await session.commit()
                logger.debug("Saved orderbook snapshots for %d symbols", len(self._books))
        except Exception as e:
            logger.warning("Failed to save orderbook snapshots: %s", e)
