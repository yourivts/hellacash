"""Bitvavo REST API wrapper with paper-trading support."""
from __future__ import annotations

import json as _json
import logging
import time
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Bitvavo fee: 0.25% taker, 0.15% maker (Starter tier)
TAKER_FEE = 0.0025
_API_BASE = "https://api.bitvavo.com/v2"
_HEADERS = {"User-Agent": "hellacash/1.0"}
_MAX_RETRIES = 3

# ── Bitvavo rate limiter ─────────────────────────────────────────────────────
# 1000 weight points per minute. Each REST call = 1 weight point.
# Unauthenticated 429 = blocked for 15 MINUTES by IP.
# Track remaining points from response headers to stay under the limit.
import threading

_rate_lock = threading.Lock()
_rate_remaining: int = 1000       # points left in current window
_rate_reset_at: float = 0.0       # epoch when counter resets
_rate_limit: int = 1000           # total points per window
_rate_blocked_until: float = 0.0  # if 429'd, don't call until this time
_RATE_SAFETY_MARGIN = 50          # stop making calls when this few points remain
_RATE_COOLDOWN = 0.15             # min seconds between calls (~6.6/sec)
_last_api_call: float = 0


def _update_rate_limits(resp) -> None:
    """Read Bitvavo rate limit headers from response."""
    global _rate_remaining, _rate_reset_at, _rate_limit
    try:
        remaining = resp.headers.get("bitvavo-ratelimit-remaining")
        reset_at = resp.headers.get("bitvavo-ratelimit-resetat")
        limit = resp.headers.get("bitvavo-ratelimit-limit")
        if remaining is not None:
            _rate_remaining = int(remaining)
        if reset_at is not None:
            _rate_reset_at = int(reset_at) / 1000.0  # ms → seconds
        if limit is not None:
            _rate_limit = int(limit)
    except (ValueError, TypeError):
        pass


def _wait_for_rate_limit() -> None:
    """Block until we're allowed to make another API call."""
    global _last_api_call
    now = time.time()

    # If we got 429'd, wait until block expires
    if now < _rate_blocked_until:
        wait = _rate_blocked_until - now
        logger.warning("Rate limited — waiting %.0fs until block expires", wait)
        time.sleep(wait)

    # If we're low on points, wait until reset
    if _rate_remaining <= _RATE_SAFETY_MARGIN and _rate_reset_at > now:
        wait = _rate_reset_at - now + 1  # +1s buffer
        logger.info("Rate limit low (%d/%d remaining) — waiting %.0fs for reset",
                     _rate_remaining, _rate_limit, wait)
        time.sleep(wait)

    # Minimum gap between calls
    elapsed = time.time() - _last_api_call
    if elapsed < _RATE_COOLDOWN:
        time.sleep(_RATE_COOLDOWN - elapsed)

    _last_api_call = time.time()


def get_rate_status() -> Dict[str, Any]:
    """Return current rate limit status for dashboard/debugging."""
    return {
        "remaining": _rate_remaining,
        "limit": _rate_limit,
        "reset_at": _rate_reset_at,
        "blocked_until": _rate_blocked_until,
        "pct_used": round((1 - _rate_remaining / max(_rate_limit, 1)) * 100, 1),
    }


def _api_get(url: str) -> Any:
    """HTTP GET with rate limit tracking, retry, and exponential backoff."""
    global _rate_blocked_until
    last_err = None
    for attempt in range(_MAX_RETRIES):
        with _rate_lock:
            _wait_for_rate_limit()
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=10) as resp:
                _update_rate_limits(resp)
                return _json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 429:
                # Unauthenticated: blocked for 15 minutes
                _rate_blocked_until = time.time() + 900
                _rate_remaining = 0
                logger.error(
                    "API 429 — blocked for 15 min (unauthenticated). "
                    "Resume at %s. URL: %s",
                    time.strftime("%H:%M:%S", time.localtime(_rate_blocked_until)),
                    url,
                )
                # Don't retry 429 for unauthenticated — it's a 15 min block
                raise
            elif 500 <= e.code < 600:
                wait = [1, 2, 5][min(attempt, 2)]
                logger.warning("API %d for %s — retrying in %ds", e.code, url, wait)
                time.sleep(wait)
            else:
                raise
        except urllib.error.HTTPError:
            raise
        except Exception as e:
            last_err = e
            if attempt < _MAX_RETRIES - 1:
                time.sleep([1, 2, 5][min(attempt, 2)])
            else:
                raise
    raise last_err  # type: ignore


# Live price cache for paper trading (symbol → (timestamp, price))
_live_price_cache: Dict[str, tuple] = {}
_PRICE_CACHE_TTL = 5  # seconds


@dataclass
class TickerData:
    symbol: str
    price: float
    bid: float
    ask: float
    volume: float
    timestamp: datetime


@dataclass
class Balance:
    asset: str
    available: float
    in_order: float


@dataclass
class CandleData:
    symbol: str
    interval: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class MarketInfo:
    symbol: str
    base: str
    quote: str
    min_order_size: float
    price_precision: int
    volume_24h: float


_PAPER_STATE_FILE = "paper_state.json"


class BitvavoClient:
    """
    Thin wrapper around python-bitvavo-api.
    In paper trading mode, place_order returns a simulated fill.
    """

    def __init__(self, api_key: str, api_secret: str, paper_trading: bool = True) -> None:
        self.paper_trading = paper_trading
        self._paper_balance: Dict[str, float] = self._load_paper_state()

        if api_key and api_secret:
            try:
                from python_bitvavo_api.bitvavo import Bitvavo  # type: ignore

                self._client = Bitvavo({"APIKEY": api_key, "APISECRET": api_secret})
                logger.info("Bitvavo REST client initialised (live=%s)", not paper_trading)
            except ImportError:
                logger.warning("python-bitvavo-api not installed; using stub REST client")
                self._client = None
        else:
            logger.info("No API credentials — paper trading only")
            self._client = None

    # ── Markets ───────────────────────────────────────────────────────────────

    def get_markets(self) -> List[MarketInfo]:
        """Return all EUR-quoted markets sorted by 24h volume."""
        if self._client is None:
            return self._stub_markets()
        try:
            raw = self._client.markets({})
            tickers = {t["market"]: t for t in self._client.ticker24h({})}
            result = []
            for m in raw:
                sym = m.get("market", "")
                if not sym.endswith("-EUR"):
                    continue
                t = tickers.get(sym, {})
                result.append(
                    MarketInfo(
                        symbol=sym,
                        base=m.get("base", ""),
                        quote=m.get("quote", "EUR"),
                        min_order_size=float(m.get("minOrderInBaseAsset", 0.0001)),
                        price_precision=int(m.get("pricePrecision", 5)),
                        volume_24h=float(t.get("volumeQuote", 0)),
                    )
                )
            result.sort(key=lambda x: x.volume_24h, reverse=True)
            return result
        except Exception as e:
            logger.error("get_markets failed: %s", e)
            return []

    def get_ticker(self, symbol: str) -> Optional[TickerData]:
        if self._client is None:
            return self._stub_ticker(symbol)
        try:
            t = self._client.tickerPrice({"market": symbol})
            book = self._client.book({"market": symbol, "depth": 1})
            bids = book.get("bids", [[0]])
            asks = book.get("asks", [[0]])
            t24 = self._client.ticker24h({"market": symbol})
            return TickerData(
                symbol=symbol,
                price=float(t.get("price", 0)),
                bid=float(bids[0][0]) if bids else 0.0,
                ask=float(asks[0][0]) if asks else 0.0,
                volume=float(t24.get("volumeQuote", 0)),
                timestamp=datetime.now(timezone.utc),
            )
        except Exception as e:
            logger.error("get_ticker(%s) failed: %s", symbol, e)
            return None

    def get_candles(
        self, symbol: str, interval: str = "5m", limit: int = 500
    ) -> List[CandleData]:
        if self._client is None:
            return self._public_candles(symbol, interval, limit)
        try:
            raw = self._client.candles(symbol, interval, {"limit": limit})
            return self._parse_candles(symbol, interval, raw)
        except Exception as e:
            logger.error("get_candles(%s) failed: %s", symbol, e)
            return []

    def _public_candles(self, symbol: str, interval: str, limit: int) -> List[CandleData]:
        """Fetch candles from Bitvavo public REST API (no credentials needed)."""
        url = f"{_API_BASE}/{symbol}/candles?interval={interval}&limit={limit}"
        try:
            raw = _api_get(url)
            return self._parse_candles(symbol, interval, raw)
        except Exception as e:
            logger.error("public_candles(%s) failed: %s", symbol, e)
            return []

    def _parse_candles(self, symbol: str, interval: str, raw: list) -> List[CandleData]:
        candles = []
        for c in raw:
            # Bitvavo format: [timestamp_ms, open, high, low, close, volume]
            candles.append(
                CandleData(
                    symbol=symbol,
                    interval=interval,
                    timestamp=datetime.fromtimestamp(c[0] / 1000, tz=timezone.utc),
                    open=float(c[1]),
                    high=float(c[2]),
                    low=float(c[3]),
                    close=float(c[4]),
                    volume=float(c[5]),
                )
            )
        return candles

    # ── Account ───────────────────────────────────────────────────────────────

    def get_balance(self) -> List[Balance]:
        if self._client is None or self.paper_trading:
            return [
                Balance(asset=k, available=v, in_order=0.0)
                for k, v in self._paper_balance.items()
            ]
        try:
            raw = self._client.balance({})
            return [
                Balance(
                    asset=b["symbol"],
                    available=float(b["available"]),
                    in_order=float(b["inOrder"]),
                )
                for b in raw
            ]
        except Exception as e:
            logger.error("get_balance failed: %s", e)
            return []

    def get_balance_eur(self) -> float:
        balances = self.get_balance()
        return next((b.available for b in balances if b.asset == "EUR"), 0.0)

    # ── Orders ────────────────────────────────────────────────────────────────

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Returns an order dict matching Bitvavo's response schema.
        In paper mode, fills immediately at current (stub) price.
        """
        if self.paper_trading or self._client is None:
            return self._paper_fill(symbol, side, order_type, amount, price)

        body: Dict[str, Any] = {
            "market": symbol,
            "side": side,
            "orderType": order_type,
            "amount": str(round(amount, 8)),
        }
        if price and order_type != "market":
            body["price"] = str(round(price, 8))

        try:
            resp = self._client.placeOrder(symbol, side, order_type, body)
            return resp
        except Exception as e:
            logger.error("place_order(%s %s %s) failed: %s", side, amount, symbol, e)
            raise

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        if self.paper_trading or self._client is None:
            return True
        try:
            self._client.cancelOrder(symbol, order_id)
            return True
        except Exception as e:
            logger.error("cancel_order(%s) failed: %s", order_id, e)
            return False

    def get_order(self, symbol: str, order_id: str) -> Optional[Dict[str, Any]]:
        if self._client is None:
            return None
        try:
            return self._client.getOrder(symbol, order_id)
        except Exception:
            return None

    # ── Paper trading helpers ─────────────────────────────────────────────────

    def _paper_fill(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: Optional[float],
    ) -> Dict[str, Any]:
        """Simulate an immediate fill for paper trading."""
        from bot.risk.fees import get_taker_fee
        ticker = self._stub_ticker(symbol)
        fill_price = price if (price and order_type != "market") else (ticker.price if ticker else 1.0)
        fee = amount * fill_price * get_taker_fee(symbol)
        order_id = str(uuid.uuid4())

        base = symbol.split("-")[0]
        if side == "buy":
            cost = amount * fill_price + fee
            self._paper_balance["EUR"] = max(0.0, self._paper_balance.get("EUR", 0) - cost)
            self._paper_balance[base] = self._paper_balance.get(base, 0.0) + amount
        else:
            proceeds = amount * fill_price - fee
            self._paper_balance["EUR"] = self._paper_balance.get("EUR", 0.0) + proceeds
            self._paper_balance[base] = max(0.0, self._paper_balance.get(base, 0.0) - amount)

        self._save_paper_state()

        return {
            "orderId": order_id,
            "market": symbol,
            "side": side,
            "orderType": order_type,
            "status": "filled",
            "amount": str(amount),
            "amountRemaining": "0",
            "price": str(fill_price),
            "amountQuote": str(amount * fill_price),
            "filledAmount": str(amount),
            "filledAmountQuote": str(amount * fill_price),
            "feePaid": str(fee),
            "feeCurrency": "EUR",
            "paper": True,
        }

    def _load_paper_state(self) -> Dict[str, float]:
        """Load paper balance from disk, or start with €10k."""
        import os
        if os.path.exists(_PAPER_STATE_FILE):
            try:
                with open(_PAPER_STATE_FILE, "r") as f:
                    data = _json.load(f)
                logger.info("Loaded paper state: EUR=%.2f + %d assets", data.get("EUR", 0), len(data) - 1)
                return data
            except Exception as e:
                logger.warning("Failed to load paper state: %s — starting fresh", e)
        return {"EUR": 10000.0}

    def _save_paper_state(self) -> None:
        """Persist paper balance to disk atomically."""
        try:
            import os
            tmp_file = _PAPER_STATE_FILE + ".tmp"
            with open(tmp_file, "w") as f:
                _json.dump(self._paper_balance, f, indent=2)
            os.replace(tmp_file, _PAPER_STATE_FILE)
        except Exception as e:
            logger.warning("Failed to save paper state: %s", e)

    def _stub_ticker(self, symbol: str) -> TickerData:
        """Fetch real price from Bitvavo public API with caching."""
        now = time.time()
        cached = _live_price_cache.get(symbol)
        if cached and (now - cached[0]) < _PRICE_CACHE_TTL:
            return cached[1]
        try:
            data = _api_get(f"{_API_BASE}/ticker/price?market={symbol}")
            price = float(data.get("price", 0))
            if price > 0:
                ticker = TickerData(
                    symbol=symbol,
                    price=price,
                    bid=price * 0.999,
                    ask=price * 1.001,
                    volume=1_000_000.0,
                    timestamp=datetime.now(timezone.utc),
                )
                _live_price_cache[symbol] = (now, ticker)
                return ticker
        except Exception as e:
            logger.debug("Live ticker fetch failed for %s: %s", symbol, e)
        # Fallback to cached or default
        if cached:
            return cached[1]
        return TickerData(
            symbol=symbol, price=1.0, bid=0.999, ask=1.001,
            volume=1_000_000.0, timestamp=datetime.now(timezone.utc),
        )

    def _stub_markets(self) -> List[MarketInfo]:
        """Fetch all EUR markets from Bitvavo public API (no credentials needed)."""
        try:
            markets = _api_get(f"{_API_BASE}/markets")
            tickers = _api_get(f"{_API_BASE}/ticker/24h")
            vol_map = {t["market"]: float(t.get("volumeQuote") or 0) for t in tickers}

            # Extract fee categories for the fee module
            from bot.risk.fees import set_market_categories
            fee_cats = {}
            result = []
            for m in markets:
                sym = m.get("market", "")
                if not sym.endswith("-EUR") or m.get("status") != "trading":
                    continue
                fee_cats[sym] = m.get("feeCategory", "A")
                result.append(MarketInfo(
                    symbol=sym,
                    base=sym.split("-")[0],
                    quote="EUR",
                    min_order_size=float(m.get("minOrderInBaseAsset") or 0.0001),
                    price_precision=int(m.get("pricePrecision") or 5),
                    volume_24h=vol_map.get(sym, 0),
                ))
            set_market_categories(fee_cats)
            result.sort(key=lambda x: x.volume_24h, reverse=True)
            logger.info("Fetched %d EUR markets from Bitvavo public API", len(result))
            return result
        except Exception as e:
            logger.error("Public markets fetch failed: %s — using minimal fallback", e)
            symbols = [
                ("BTC-EUR", "BTC"), ("ETH-EUR", "ETH"), ("SOL-EUR", "SOL"),
                ("ADA-EUR", "ADA"), ("DOT-EUR", "DOT"),
            ]
            return [
                MarketInfo(symbol=s, base=b, quote="EUR", min_order_size=0.0001, price_precision=5, volume_24h=1_000_000)
                for s, b in symbols
            ]
