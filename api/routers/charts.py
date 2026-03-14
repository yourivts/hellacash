"""Candle data endpoint for charts."""
from __future__ import annotations

import asyncio
import json
import logging
import urllib.request
from datetime import datetime, timezone

from fastapi import APIRouter, Path, Query
from fastapi.responses import JSONResponse

import bot.main as bot_main
from bot.exchange.bitvavo_client import _api_get, _API_BASE

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/candles", tags=["charts"])


def _format_candles(rows) -> list:
    return [
        {
            "timestamp": r["timestamp"].isoformat() if hasattr(r["timestamp"], "isoformat") else str(r["timestamp"]),
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
            "volume": float(r["volume"]),
        }
        for _, r in rows.iterrows()
    ]


def _fetch_public(symbol: str, interval: str, limit: int) -> list:
    """Fallback: fetch directly from Bitvavo public REST API."""
    url = f"https://api.bitvavo.com/v2/{symbol}/candles?interval={interval}&limit={limit}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "hellacash/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = json.loads(resp.read().decode())
        candles = [
            {
                "timestamp": datetime.fromtimestamp(c[0] / 1000, tz=timezone.utc).isoformat(),
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5]),
            }
            for c in raw
        ]
        # Bitvavo returns newest-first; Lightweight Charts needs ascending order
        candles.sort(key=lambda x: x["timestamp"])
        return candles
    except Exception:
        return []


import time

# Cache markets list for 1 hour
_markets_cache: dict = {"data": [], "ts": 0}
MARKETS_TTL = 3600


def _fetch_markets() -> list:
    """Fetch all EUR trading markets from Bitvavo, sorted by 24h volume."""
    now = time.time()
    if _markets_cache["data"] and (now - _markets_cache["ts"]) < MARKETS_TTL:
        return _markets_cache["data"]

    # Get markets
    markets = _api_get(f"{_API_BASE}/markets")

    # Get 24h tickers for volume sorting
    tickers = _api_get(f"{_API_BASE}/ticker/24h")

    ticker_map = {t["market"]: t for t in tickers}

    result = []
    for m in markets:
        sym = m.get("market", "")
        if not sym.endswith("-EUR") or m.get("status") != "trading":
            continue
        t = ticker_map.get(sym, {})
        result.append({
            "symbol": sym,
            "base": sym.split("-")[0],
            "price": float(t.get("last") or 0),
            "volume_24h": float(t.get("volumeQuote") or 0),
        })

    result.sort(key=lambda x: x["volume_24h"], reverse=True)
    result = result[:100]
    _markets_cache["data"] = result
    _markets_cache["ts"] = now
    return result


@router.get("/markets")
async def get_markets():
    """Return only the bot's tradeable symbols (with ticker data for the dropdown)."""
    try:
        symbols = bot_main.get_tradeable_symbols()
        if not symbols:
            # Bot not started yet — fall back to public API
            result = await asyncio.get_event_loop().run_in_executor(None, _fetch_markets)
            return result
        # Fetch tickers for price/volume display
        tickers = await asyncio.get_event_loop().run_in_executor(
            None, lambda: _api_get(f"{_API_BASE}/ticker/24h")
        )
        ticker_map = {t["market"]: t for t in tickers}
        result = []
        for sym in symbols:
            t = ticker_map.get(sym, {})
            result.append({
                "symbol": sym,
                "base": sym.split("-")[0],
                "price": float(t.get("last") or 0),
                "volume_24h": float(t.get("volumeQuote") or 0),
            })
        result.sort(key=lambda x: x["volume_24h"], reverse=True)
        return result
    except Exception as e:
        logger.error("markets fetch failed: %s", e)
        return []


# Ticker cache: avoid hitting API more than once per second per symbol
_ticker_cache: dict = {}  # symbol → {"data": ..., "ts": float}
_TICKER_TTL = 1.0  # seconds


def _fetch_ticker(symbol: str) -> dict:
    """Sync fetch of price + orderbook with 1s cache to avoid rate limits."""
    now = time.time()
    cached = _ticker_cache.get(symbol)
    if cached and (now - cached["ts"]) < _TICKER_TTL:
        return cached["data"]

    price_data = _api_get(f"{_API_BASE}/ticker/price?market={symbol}")
    book_data = _api_get(f"{_API_BASE}/{symbol}/book?depth=10")
    result = {
        "price": price_data.get("price"),
        "bids": book_data.get("bids", []),
        "asks": book_data.get("asks", []),
    }
    _ticker_cache[symbol] = {"data": result, "ts": now}
    return result


@router.get("/ticker/{symbol}")
async def get_ticker_price(symbol: str = Path(...)):
    """Proxy Bitvavo price + orderbook to avoid browser CORS issues."""
    try:
        result = await asyncio.get_event_loop().run_in_executor(None, _fetch_ticker, symbol)
        return JSONResponse(content=result, headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"})
    except Exception as e:
        logger.error("ticker proxy failed for %s: %s", symbol, e)
        return JSONResponse(content={"price": None, "bids": [], "asks": []}, headers={"Cache-Control": "no-store"})


@router.get("/{symbol}")
async def get_candles(
    symbol: str = Path(...),
    interval: str = Query("5m"),
    limit: int = Query(200, le=500),
):
    df = bot_main.get_candle_cache().get_df(symbol, interval)
    if len(df) >= limit:
        rows = df.tail(limit).reset_index()
        return _format_candles(rows)
    # Cache empty or too few candles — fetch from public API
    return _fetch_public(symbol, interval, limit)
