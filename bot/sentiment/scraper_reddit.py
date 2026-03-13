"""Reddit scraper using public JSON API (no credentials required)."""
from __future__ import annotations

import logging
import time
import urllib.request
import json
from typing import Dict, List, Tuple

from bot.sentiment.keywords import get_keywords

logger = logging.getLogger(__name__)

SUBREDDITS = [
    # Major crypto subs
    "CryptoCurrency",
    "Bitcoin",
    "ethereum",
    "altcoin",
    "CryptoMarkets",
    "defi",
    "SatoshiStreetBets",
    "CryptoMoonShots",
    # Trading & analysis
    "CryptoTechnology",
    "BitcoinMarkets",
    "ethtrader",
    "ethfinance",
    "CryptoCurrencyTrading",
    "Daytrading",
    "wallstreetbetscrypto",
    # Major L1/L2 coins
    "solana",
    "cardano",
    "Ripple",
    "dogecoin",
    "polkadot",
    "Chainlink",
    "cosmosnetwork",
    "algorand",
    "Tezos",
    "Stellar",
    "Hedera",
    "avax",
    "FantomFoundation",
    "Tronix",
    "NEO",
    "VeChain",
    "harmony_one",
    "nearprotocol",
    "Elrond",
    "IOStoken",
    "Ravencoin",
    # DeFi & L2
    "UniSwap",
    "Aave",
    "PancakeSwapOfficial",
    "SushiSwapOfficial",
    "MakerDAO",
    "LidoFinance",
    "Arbitrum",
    "optimismCollective",
    "polygonnetwork",
    "0xPolygon",
    "starknet",
    # NFT & Gaming
    "NFT",
    "AxieInfinity",
    "TheSandboxGaming",
    "decentraland",
    "ImmutableX",
    # Memecoins
    "SHIBArmy",
    "Floki",
    "pepecoin",
    "dogelon",
    "BONK",
    # Privacy & misc
    "Monero",
    "litecoin",
    "EOS",
    "Iota",
]

HEADERS = {"User-Agent": "hellacash/1.0 (sentiment bot; +https://github.com/hellacash)"}

# Cache subreddit data for 30 minutes to avoid 429s
CACHE_TTL = 1800
_sub_cache: Dict[str, Tuple[float, list]] = {}  # sub_name → (timestamp, posts)


def _fetch_sub(sub_name: str) -> list:
    """Fetch a subreddit's hot posts, with caching and rate-limit backoff."""
    now = time.time()
    cached = _sub_cache.get(sub_name)
    if cached and (now - cached[0]) < CACHE_TTL:
        return cached[1]

    # Use old.reddit.com — different rate limits than www.reddit.com
    url = f"https://old.reddit.com/r/{sub_name}/hot.json?limit=100"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode())

    posts = data.get("data", {}).get("children", [])
    _sub_cache[sub_name] = (now, posts)
    return posts


class RedditScraper:
    def __init__(self, client_id: str = "", client_secret: str = "", user_agent: str = "") -> None:
        pass  # credentials no longer needed

    def prefetch_subs(self) -> None:
        """Fetch all subreddits once, populating the cache. Call before scoring assets."""
        fetched = 0
        skipped = 0
        total = len(SUBREDDITS)
        for i, sub_name in enumerate(SUBREDDITS):
            # Skip if still cached
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
                if e.code == 429:
                    logger.warning("Reddit 429 for r/%s after %d fetched — will continue next cycle", sub_name, fetched)
                    break
                logger.error("Reddit prefetch error r/%s: %s", sub_name, e)
            except Exception as e:
                logger.error("Reddit prefetch error r/%s: %s", sub_name, e)
            # 1s delay — old.reddit.com is more lenient
            time.sleep(1)
        logger.info("Reddit prefetch done — %d subs cached (%d new, %d already cached)", len(_sub_cache), fetched, skipped)

    def fetch_posts(self, asset: str, limit: int = 50) -> List[str]:
        """Filter cached subreddit posts by asset keywords. No HTTP requests."""
        keywords = get_keywords(asset)
        texts: List[str] = []

        for sub_name in SUBREDDITS:
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
