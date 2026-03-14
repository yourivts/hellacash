"""On-chain metrics signal provider — Binance funding/OI + Blockchair whales."""
from __future__ import annotations

import json
import logging
import time
import urllib.request
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

BITVAVO_TO_BINANCE: Dict[str, str] = {
    "BTC-EUR": "BTCUSDT", "ETH-EUR": "ETHUSDT", "SOL-EUR": "SOLUSDT",
    "XRP-EUR": "XRPUSDT", "ADA-EUR": "ADAUSDT", "DOGE-EUR": "DOGEUSDT",
    "DOT-EUR": "DOTUSDT", "LINK-EUR": "LINKUSDT", "AVAX-EUR": "AVAXUSDT",
    "MATIC-EUR": "MATICUSDT", "ATOM-EUR": "ATOMUSDT", "UNI-EUR": "UNIUSDT",
    "LTC-EUR": "LTCUSDT", "BCH-EUR": "BCHUSDT", "FIL-EUR": "FILUSDT",
}

# Scoring weights per metric
METRIC_WEIGHTS = {"funding": 0.35, "oi": 0.35, "whale": 0.20, "reserves": 0.10}
_BINANCE_FAPI = "https://fapi.binance.com"
_BLOCKCHAIR = "https://api.blockchair.com"
_TIMEOUT = 10


def _parse_funding_rate(rate: float) -> float:
    if rate > 0.0001:
        return -0.5  # crowded longs → contrarian bearish
    elif rate < -0.0001:
        return 0.5   # crowded shorts → contrarian bullish
    return 0.0


def _parse_open_interest(
    current_oi: float, prev_oi: float,
    current_price: float, prev_price: float,
) -> float:
    if prev_oi <= 0 or prev_price <= 0:
        return 0.0
    oi_change = (current_oi - prev_oi) / prev_oi
    price_change = (current_price - prev_price) / prev_price
    if abs(oi_change) < 0.01:
        return 0.0
    if oi_change > 0 and price_change > 0:
        return 0.5   # trend confirmation
    elif oi_change > 0 and price_change < 0:
        return -0.5  # bearish pressure
    elif oi_change < 0 and price_change > 0:
        return 0.3   # short squeeze potential
    elif oi_change < 0 and price_change < 0:
        return -0.3  # long liquidation
    return 0.0


def _fetch_json(url: str) -> Optional[Any]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "hellacash/1.0"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        logger.warning("On-chain fetch failed %s: %s", url, e)
        return None


class OnchainProvider:
    """On-chain metrics signal provider. Caches results, polled async."""

    name = "onchain"

    def __init__(self) -> None:
        self._cache: Dict[str, Dict[str, float]] = {}
        self._prev_oi: Dict[str, float] = {}
        self._prev_price: Dict[str, float] = {}
        self._has_data = False
        self._failure_count = 0
        self._disabled_until = 0.0

    def score(self, symbol: str, **kwargs) -> float:
        if not self._has_data or symbol not in self._cache:
            return 0.0
        metrics = self._cache[symbol]
        if not metrics:
            return 0.0
        active = {k: v for k, v in metrics.items() if k in METRIC_WEIGHTS}
        if not active:
            return 0.0
        weights = {k: METRIC_WEIGHTS[k] for k in active}
        total_w = sum(weights.values())
        return max(-1.0, min(1.0,
            sum(active[k] * weights[k] / total_w for k in active)
        ))

    def is_available(self) -> bool:
        return self._has_data and time.time() > self._disabled_until

    async def poll(self, symbols: list[str], current_prices: Dict[str, float]) -> None:
        if time.time() < self._disabled_until:
            return

        import asyncio
        loop = asyncio.get_event_loop()

        for symbol in symbols:
            binance_sym = BITVAVO_TO_BINANCE.get(symbol)
            if not binance_sym:
                continue

            try:
                metrics: Dict[str, float] = {}

                # Funding rate
                data = await loop.run_in_executor(None, _fetch_json,
                    f"{_BINANCE_FAPI}/fapi/v1/fundingRate?symbol={binance_sym}&limit=1")
                if data and len(data) > 0:
                    rate = float(data[0].get("fundingRate", 0))
                    metrics["funding"] = _parse_funding_rate(rate)

                # Open interest
                data = await loop.run_in_executor(None, _fetch_json,
                    f"{_BINANCE_FAPI}/fapi/v1/openInterest?symbol={binance_sym}")
                if data:
                    oi = float(data.get("openInterest", 0))
                    price = current_prices.get(symbol, 0)
                    prev_oi = self._prev_oi.get(symbol, oi)
                    prev_price = self._prev_price.get(symbol, price)
                    metrics["oi"] = _parse_open_interest(oi, prev_oi, price, prev_price)
                    self._prev_oi[symbol] = oi
                    self._prev_price[symbol] = price

                self._cache[symbol] = metrics
                self._has_data = True
                self._failure_count = 0

            except Exception as e:
                logger.error("On-chain poll error for %s: %s", symbol, e)
                self._failure_count += 1
                if self._failure_count >= 3:
                    self._disabled_until = time.time() + 600
                    logger.warning("On-chain circuit breaker: disabled for 10 min")
