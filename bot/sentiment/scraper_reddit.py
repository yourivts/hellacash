"""Reddit scraper using PRAW."""
from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)

SUBREDDITS = ["CryptoCurrency", "Bitcoin", "ethereum", "altcoin", "CryptoMarkets"]

ASSET_KEYWORDS = {
    "BTC": ["bitcoin", "btc", "$btc"],
    "ETH": ["ethereum", "eth", "$eth"],
    "SOL": ["solana", "sol", "$sol"],
    "ADA": ["cardano", "ada", "$ada"],
    "DOT": ["polkadot", "dot", "$dot"],
    "LINK": ["chainlink", "link", "$link"],
    "AVAX": ["avalanche", "avax", "$avax"],
    "MATIC": ["polygon", "matic", "$matic"],
    "XRP": ["ripple", "xrp", "$xrp"],
    "DOGE": ["dogecoin", "doge", "$doge"],
}


def _build_reddit(client_id: str, client_secret: str, user_agent: str):
    try:
        import praw  # type: ignore
        return praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )
    except ImportError:
        logger.warning("praw not installed; Reddit scraper disabled")
        return None


class RedditScraper:
    def __init__(self, client_id: str, client_secret: str, user_agent: str) -> None:
        self._reddit = _build_reddit(client_id, client_secret, user_agent)

    def fetch_posts(self, asset: str, limit: int = 50) -> List[str]:
        """Fetch hot post titles + text mentioning the asset."""
        if self._reddit is None or not client_id_present(asset):
            return []

        keywords = ASSET_KEYWORDS.get(asset.upper(), [asset.lower()])
        texts: List[str] = []

        try:
            for sub_name in SUBREDDITS:
                sub = self._reddit.subreddit(sub_name)
                for post in sub.hot(limit=limit // len(SUBREDDITS) + 5):
                    content = (post.title + " " + (post.selftext or "")).lower()
                    if any(kw in content for kw in keywords):
                        texts.append(post.title[:280])
                        if len(texts) >= limit:
                            break
                if len(texts) >= limit:
                    break
        except Exception as e:
            logger.error("Reddit fetch error: %s", e)

        return texts[:limit]

    @property
    def available(self) -> bool:
        return self._reddit is not None


def client_id_present(asset: str) -> bool:
    return asset.upper() in ASSET_KEYWORDS or len(asset) <= 5
