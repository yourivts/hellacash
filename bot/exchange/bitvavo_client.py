"""Bitvavo REST API wrapper with paper-trading support."""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Bitvavo fee: 0.25% taker, 0.15% maker (Starter tier)
TAKER_FEE = 0.0025


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


class BitvavoClient:
    """
    Thin wrapper around python-bitvavo-api.
    In paper trading mode, place_order returns a simulated fill.
    """

    def __init__(self, api_key: str, api_secret: str, paper_trading: bool = True) -> None:
        self.paper_trading = paper_trading
        self._paper_balance: Dict[str, float] = {"EUR": 10000.0}  # start with €10k

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
            return []
        try:
            raw = self._client.candles(symbol, interval, {"limit": limit})
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
        except Exception as e:
            logger.error("get_candles(%s) failed: %s", symbol, e)
            return []

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
        ticker = self._stub_ticker(symbol)
        fill_price = price if (price and order_type != "market") else (ticker.price if ticker else 1.0)
        fee = amount * fill_price * TAKER_FEE
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

    def _stub_ticker(self, symbol: str) -> TickerData:
        prices = {
            "BTC-EUR": 55000.0, "ETH-EUR": 3200.0, "SOL-EUR": 130.0,
            "ADA-EUR": 0.45, "DOT-EUR": 7.0, "LINK-EUR": 14.0,
        }
        price = prices.get(symbol, 1.0)
        return TickerData(
            symbol=symbol,
            price=price,
            bid=price * 0.999,
            ask=price * 1.001,
            volume=1_000_000.0,
            timestamp=datetime.now(timezone.utc),
        )

    def _stub_markets(self) -> List[MarketInfo]:
        symbols = [
            ("BTC-EUR", "BTC"), ("ETH-EUR", "ETH"), ("SOL-EUR", "SOL"),
            ("ADA-EUR", "ADA"), ("DOT-EUR", "DOT"),
        ]
        return [
            MarketInfo(symbol=s, base=b, quote="EUR", min_order_size=0.0001, price_precision=5, volume_24h=1_000_000)
            for s, b in symbols
        ]
