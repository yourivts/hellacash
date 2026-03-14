"""CandleCache — centralised candle storage and lazy-loading helpers."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

MAX_CANDLES = 2000


class CandleCache:
    """Thread-safe, in-memory cache for OHLCV candle data."""

    def __init__(self) -> None:
        # symbol -> interval -> list of dicts (chronological)
        self._cache: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self._is_ready: bool = False
        self._progress: Dict[str, Any] = {"loaded": 0, "total": 0, "done": False}

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @property
    def progress(self) -> Dict[str, Any]:
        return self._progress

    # ── Public API ────────────────────────────────────────────────────────────

    def cache_candle(self, symbol: str, interval: str, candle: Dict[str, Any]) -> None:
        self._cache.setdefault(symbol, {}).setdefault(interval, [])
        cache = self._cache[symbol][interval]
        # Upsert: update existing candle if same timestamp, else append
        ts = candle.get("timestamp")
        if cache and cache[-1].get("timestamp") == ts:
            cache[-1] = candle
        else:
            cache.append(candle)
            if len(cache) > MAX_CANDLES:
                cache.pop(0)

    def get_df(self, symbol: str, interval: str) -> pd.DataFrame:
        rows = self._cache.get(symbol, {}).get(interval, [])
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        return df

    def has_enough(self, symbol: str, interval: str = "5m", minimum: int = 30) -> bool:
        return len(self._cache.get(symbol, {}).get(interval, [])) >= minimum

    async def load_all(self, symbols: List[str], client: Any, batch_size: int = 5) -> None:
        """Load candles for ALL active symbols in concurrent batches."""
        loop = asyncio.get_running_loop()
        total = len(symbols)
        self._progress["total"] = total
        self._progress["loaded"] = 0
        self._progress["done"] = False

        for batch_start in range(0, total, batch_size):
            batch = symbols[batch_start : batch_start + batch_size]
            tasks = []
            for symbol in batch:
                cached_5m = self._cache.get(symbol, {}).get("5m", [])
                if len(cached_5m) >= 30:
                    continue
                tasks.append(self._load_symbol(client, symbol, loop))
            if tasks:
                await asyncio.gather(*tasks)
            self._progress["loaded"] = min(batch_start + batch_size, total)
            if self._progress["loaded"] % 50 == 0 or self._progress["loaded"] == total:
                logger.info("Candle loading progress: %d/%d", self._progress["loaded"], total)
            await asyncio.sleep(1)  # rate limit: 1s pause between batches

        self._progress["done"] = True
        self._is_ready = True
        logger.info("All candles loaded (%d pairs)", total)

    async def lazy_load(self, symbol: str, client: Any) -> None:
        """Fetch candles from REST API if not yet cached."""
        cached_5m = self._cache.get(symbol, {}).get("5m", [])
        if len(cached_5m) >= 30:
            return  # already have enough
        loop = asyncio.get_running_loop()
        for interval in ["5m", "1h"]:
            try:
                candles = await loop.run_in_executor(
                    None, lambda iv=interval: client.get_candles(symbol, iv, limit=1500)
                )
                for c in candles:
                    self.cache_candle(symbol, interval, {
                        "symbol": c.symbol, "interval": c.interval,
                        "timestamp": c.timestamp, "open": c.open,
                        "high": c.high, "low": c.low,
                        "close": c.close, "volume": c.volume,
                    })
            except Exception as e:
                logger.error("Lazy load candles %s %s: %s", symbol, interval, e)

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _load_symbol(self, client: Any, symbol: str, loop: Any) -> None:
        """Load candles for a single symbol."""
        for interval in ["5m", "1h"]:
            try:
                candles = await loop.run_in_executor(
                    None, lambda s=symbol, iv=interval: client.get_candles(s, iv, limit=1500)
                )
                for c in candles:
                    self.cache_candle(symbol, interval, {
                        "symbol": c.symbol, "interval": c.interval,
                        "timestamp": c.timestamp, "open": c.open,
                        "high": c.high, "low": c.low,
                        "close": c.close, "volume": c.volume,
                    })
            except Exception as e:
                logger.error("Failed to load candles for %s %s: %s", symbol, interval, e)
