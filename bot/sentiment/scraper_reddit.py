"""Reddit scraper using public JSON API (no credentials required)."""
from __future__ import annotations

import logging
import time
import urllib.request
import json
from typing import Dict, List, Tuple

from bot.sentiment.keywords import get_keywords

logger = logging.getLogger(__name__)

# General crypto subs — always scraped (cover all assets)
GENERAL_SUBS = [
    "CryptoCurrency",
    "CryptoMarkets",
    "SatoshiStreetBets",
    "CryptoMoonShots",
    "BitcoinMarkets",
    "ethtrader",
    "ethfinance",
    "defi",
    "altcoin",
    "Daytrading",
    "wallstreetbetscrypto",
]

# Coin-specific subs — only scraped if the base asset is tradeable
_COIN_SUBS: Dict[str, List[str]] = {
    "BTC": ["Bitcoin"],
    "ETH": ["ethereum"],
    "SOL": ["solana"],
    "ADA": ["cardano"],
    "XRP": ["Ripple"],
    "DOGE": ["dogecoin"],
    "DOT": ["polkadot"],
    "LINK": ["Chainlink"],
    "ATOM": ["cosmosnetwork"],
    "AVAX": ["avax"],
    "MATIC": ["0xPolygon"],
    "ARB": ["Arbitrum"],
    "OP": ["optimismCollective"],
    "UNI": ["UniSwap"],
    "AAVE": ["Aave"],
    "MKR": ["MakerDAO"],
    "LDO": ["LidoFinance"],
    "NEAR": ["nearprotocol"],
    "SHIB": ["SHIBArmy"],
    "LTC": ["litecoin"],
    "XMR": ["Monero"],
    "IOTA": ["Iota"],
    "HBAR": ["Hedera"],
    "VET": ["VeChain"],
    "ALGO": ["algorand"],
    "XTZ": ["Tezos"],
    "XLM": ["Stellar"],
    "FLOKI": ["Floki"],
    "BONK": ["BONK"],
    "TRX": ["Tronix"],
    "CAKE": ["pancakeswap"],
    "IMX": ["ImmutableX"],
    "STRK": ["starknet"],
}

# Built dynamically based on tradeable symbols
_active_subs: List[str] = []


def _build_sub_list(tradeable_symbols: List[str]) -> List[str]:
    """Build subreddit list from general subs + coin subs for tradeable assets."""
    subs = list(GENERAL_SUBS)
    for sym in tradeable_symbols:
        base = sym.split("-")[0].upper()
        for sub in _COIN_SUBS.get(base, []):
            if sub not in subs:
                subs.append(sub)
    return subs

# Cache subreddit data for 5 minutes
CACHE_TTL = 300
_sub_cache: Dict[str, Tuple[float, list]] = {}  # sub_name → (timestamp, posts)

# Resolved at first use from settings
_user_agent: str = ""


def _get_user_agent() -> str:
    global _user_agent
    if not _user_agent:
        from bot.config import get_settings
        _user_agent = get_settings().reddit_user_agent
        logger.info("Reddit User-Agent: %s", _user_agent)
    return _user_agent


def _fetch_sub(sub_name: str) -> list:
    """Fetch a subreddit's hot posts, with caching and rate-limit backoff."""
    now = time.time()
    cached = _sub_cache.get(sub_name)
    if cached and (now - cached[0]) < CACHE_TTL:
        return cached[1]

    url = f"https://www.reddit.com/r/{sub_name}/hot.json?limit=100&raw_json=1"
    headers = {
        "User-Agent": _get_user_agent(),
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode())

    posts = data.get("data", {}).get("children", [])
    _sub_cache[sub_name] = (now, posts)
    return posts


BACKOFF_CYCLES = 3  # skip this many cycles after a 403/429


class RedditScraper:
    def __init__(self, client_id: str = "", client_secret: str = "", user_agent: str = "") -> None:
        self._subs: List[str] = list(GENERAL_SUBS)
        self._backoff_until: float = 0.0  # timestamp when backoff expires

    def set_tradeable_symbols(self, symbols: List[str]) -> None:
        """Rebuild the subreddit list based on current tradeable symbols."""
        self._subs = _build_sub_list(symbols)
        logger.info("Reddit scraper active subs: %d (%d general + %d coin-specific)",
                     len(self._subs), len(GENERAL_SUBS), len(self._subs) - len(GENERAL_SUBS))

    def prefetch_subs(self) -> None:
        """Fetch all active subreddits once, populating the cache."""
        now = time.time()
        if now < self._backoff_until:
            remaining = int(self._backoff_until - now)
            logger.info("Reddit backoff active — skipping prefetch (%d s remaining)", remaining)
            return

        fetched = 0
        skipped = 0
        blocked = False
        total = len(self._subs)
        for sub_name in self._subs:
            cached = _sub_cache.get(sub_name)
            if cached and (time.time() - cached[0]) < CACHE_TTL:
                skipped += 1
                continue
            try:
                _fetch_sub(sub_name)
                fetched += 1
                if fetched % 10 == 0:
                    logger.info("Reddit prefetch progress: %d/%d fetched, %d cached", fetched, total - skipped, skipped)
            except urllib.error.HTTPError as e:
                if e.code in (429, 403):
                    self._backoff_until = time.time() + BACKOFF_CYCLES * CACHE_TTL
                    logger.warning(
                        "Reddit %d for r/%s after %d fetched — backing off for %d s",
                        e.code, sub_name, fetched, BACKOFF_CYCLES * CACHE_TTL,
                    )
                    blocked = True
                    break
                logger.error("Reddit prefetch error r/%s: %s", sub_name, e)
            except Exception as e:
                logger.error("Reddit prefetch error r/%s: %s", sub_name, e)
            time.sleep(1)

        if not blocked:
            logger.info("Reddit prefetch done — %d subs cached (%d new, %d already cached)", len(_sub_cache), fetched, skipped)

    def fetch_posts(self, asset: str, limit: int = 50) -> List[str]:
        """Filter cached subreddit posts by asset keywords. No HTTP requests."""
        keywords = get_keywords(asset)
        texts: List[str] = []

        for sub_name in self._subs:
            cached = _sub_cache.get(sub_name)
            if not cached:
                continue
            for post in cached[1]:
                p = post["data"]
                content = (p.get("title", "") + " " + p.get("selftext", "")).lower()
                if any(kw in content for kw in keywords):
                    texts.append(p["title"][:280])
                    if len(texts) >= limit:
                        return texts

        return texts[:limit]

    @property
    def available(self) -> bool:
        return True
