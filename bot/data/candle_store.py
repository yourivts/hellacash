"""CandleStore — bulk 1m candle download and storage for RL training."""
from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)

# Max entries in the candle DataFrame cache (LRU eviction)
_CACHE_MAX_ENTRIES = 200


class CandleStore:
    """Two access modes:
    - async methods for I/O-bound download (Bitvavo API)
    - sync methods for CPU-bound training (SubprocVecEnv workers)
    Both use the same PostgreSQL candle_1m table.
    """

    def __init__(self, db_url: str) -> None:
        self._db_url = db_url
        self._sync_url = db_url.replace("+asyncpg", "").replace("postgresql+psycopg2", "postgresql")
        if self._sync_url.startswith("postgresql+"):
            self._sync_url = "postgresql" + self._sync_url[self._sync_url.index("://"):]
        # In-memory caches to avoid repeated DB reads during training
        self._candle_cache: Dict[str, pd.DataFrame] = {}
        self._candle_cache_order: List[str] = []  # LRU order
        self._date_range_cache: Dict[str, Optional[Tuple[datetime, datetime]]] = {}

    async def bulk_download(self, symbols: List[str],
                            min_days: int = 90) -> None:
        """Pull all available 1m candles from Bitvavo for each symbol.
        Resumes from last stored timestamp per symbol.
        Skips symbols with <min_days of history (too new for training).
        """
        import asyncio
        for symbol in symbols:
            try:
                span = self._get_data_span_days(symbol)
                if span is not None and span < min_days:
                    logger.info("CandleStore: skipping %s (only %d days of data)", symbol, span)
                    continue
                await self._download_symbol(symbol)
            except Exception as e:
                logger.error("CandleStore: failed to download %s: %s", symbol, e)
        self.clear_cache()  # invalidate stale cached data after download

    async def incremental_update(self, symbols: List[str]) -> None:
        """Pull new 1m candles since last download."""
        await self.bulk_download(symbols)

    async def _download_symbol(self, symbol: str) -> None:
        """Download 1m candles for one symbol from Bitvavo."""
        import asyncio
        from bot.exchange.bitvavo_client import get_client

        loop = asyncio.get_running_loop()
        client = get_client()

        last_ts = self._get_last_timestamp(symbol)
        start_ms = int(last_ts.timestamp() * 1000) if last_ts else 0

        total_stored = 0
        retries = 0
        max_retries = 3

        while True:
            try:
                candles = await loop.run_in_executor(
                    None,
                    lambda: client.bitvavo.candles(symbol.replace("-", ""), "1m",
                                                    {"start": start_ms, "limit": 1440}),
                )
            except Exception as e:
                retries += 1
                if retries > max_retries:
                    logger.warning("CandleStore: giving up on %s after %d retries: %s",
                                   symbol, max_retries, e)
                    break
                wait = min(2 ** retries, 30)
                logger.info("CandleStore: rate limited on %s, waiting %ds", symbol, wait)
                await asyncio.sleep(wait)
                continue

            retries = 0
            if not candles:
                break

            self._store_candles(symbol, candles)
            total_stored += len(candles)

            last_candle_ts = max(c[0] for c in candles)
            start_ms = last_candle_ts + 60_000

            await asyncio.sleep(0.1)

        if total_stored > 0:
            logger.info("CandleStore: stored %d candles for %s", total_stored, symbol)

    def _get_last_timestamp(self, symbol: str) -> Optional[datetime]:
        """Get last stored timestamp for a symbol (sync)."""
        conn = psycopg2.connect(self._sync_url)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT MAX(timestamp) FROM candle_1m WHERE symbol = %s",
                    (symbol,),
                )
                row = cur.fetchone()
                return row[0] if row and row[0] else None
        finally:
            conn.close()

    def _get_data_span_days(self, symbol: str) -> Optional[int]:
        """Return number of days of stored data for a symbol, or None if no data."""
        date_range = self.get_date_range(symbol)
        if date_range is None:
            return None
        return (date_range[1] - date_range[0]).days

    def _store_candles(self, symbol: str, candles: list) -> None:
        """Bulk insert candles into candle_1m table (sync)."""
        conn = psycopg2.connect(self._sync_url)
        try:
            with conn.cursor() as cur:
                values = []
                for c in candles:
                    ts = datetime.fromtimestamp(c[0] / 1000.0, tz=timezone.utc)
                    values.append((symbol, ts, float(c[1]), float(c[2]),
                                   float(c[3]), float(c[4]), float(c[5])))
                psycopg2.extras.execute_values(
                    cur,
                    """INSERT INTO candle_1m (symbol, timestamp, open, high, low, close, volume)
                       VALUES %s ON CONFLICT (symbol, timestamp) DO NOTHING""",
                    values,
                )
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _normalize_freq(resample: str) -> str:
        """Normalize legacy short-form minute offsets ('5m') to pandas-compatible form ('5min').
        Pandas >= 2.2 deprecated 'T' and >= 3.0 removed bare 'm' for minutes.
        """
        import re
        # Replace trailing 'm' (minute) that is NOT followed by other letters (avoids 'ME', 'MS')
        return re.sub(r"^(\d*)m$", lambda match: (match.group(1) or "") + "min", resample)

    def _cache_key(self, symbol: str, start: datetime, end: datetime) -> str:
        """Build a stable cache key for a (symbol, start, end) query."""
        return f"{symbol}|{start.isoformat()}|{end.isoformat()}"

    def _get_cached_1m(self, symbol: str, start: datetime,
                       end: datetime) -> pd.DataFrame:
        """Return cached 1m DataFrame, reading from DB on first access."""
        key = self._cache_key(symbol, start, end)
        if key in self._candle_cache:
            # Move to end of LRU list
            self._candle_cache_order.remove(key)
            self._candle_cache_order.append(key)
            return self._candle_cache[key]

        df = self._read_candles_sync(symbol, start, end)
        # Evict oldest if cache is full
        while len(self._candle_cache_order) >= _CACHE_MAX_ENTRIES:
            evict = self._candle_cache_order.pop(0)
            self._candle_cache.pop(evict, None)
        self._candle_cache[key] = df
        self._candle_cache_order.append(key)
        return df

    def clear_cache(self) -> None:
        """Clear in-memory caches (call after bulk_download)."""
        self._candle_cache.clear()
        self._candle_cache_order.clear()
        self._date_range_cache.clear()

    def get_candles(self, symbol: str, start: datetime, end: datetime,
                    resample: str = "5m") -> pd.DataFrame:
        """Fetch candles for a date range, optionally resampled.
        SYNC — safe to call from SubprocVecEnv workers.
        Results are cached in memory to avoid repeated DB reads during training.
        """
        df = self._get_cached_1m(symbol, start, end)
        if df.empty:
            return df

        if resample in ("1m", "1min"):
            return df

        freq = self._normalize_freq(resample)
        return df.resample(freq).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

    def _read_candles_sync(self, symbol: str, start: datetime,
                           end: datetime) -> pd.DataFrame:
        """Read 1m candles from DB (sync psycopg2 connection)."""
        conn = psycopg2.connect(self._sync_url)
        try:
            df = pd.read_sql(
                "SELECT timestamp, open, high, low, close, volume "
                "FROM candle_1m WHERE symbol = %s AND timestamp >= %s AND timestamp < %s "
                "ORDER BY timestamp",
                conn,
                params=(symbol, start, end),
            )
            if df.empty:
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")
            return df
        finally:
            conn.close()

    def get_date_range(self, symbol: str) -> Optional[tuple]:
        """Return (min_ts, max_ts) for a symbol, or None if no data. Cached."""
        if symbol in self._date_range_cache:
            return self._date_range_cache[symbol]
        conn = psycopg2.connect(self._sync_url)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT MIN(timestamp), MAX(timestamp) FROM candle_1m WHERE symbol = %s",
                    (symbol,),
                )
                row = cur.fetchone()
                result = (row[0], row[1]) if row and row[0] else None
                self._date_range_cache[symbol] = result
                return result
        finally:
            conn.close()

    def available_symbols(self) -> List[str]:
        """Return list of symbols with stored data (sync)."""
        conn = psycopg2.connect(self._sync_url)
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT symbol FROM candle_1m ORDER BY symbol")
                return [row[0] for row in cur.fetchall()]
        finally:
            conn.close()
