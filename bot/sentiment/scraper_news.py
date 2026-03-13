"""RSS/Atom news feed scraper."""
from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)

NEWS_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
    "https://feeds.reuters.com/reuters/businessNews",
]

ASSET_KEYWORDS = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth"],
    "SOL": ["solana", "sol"],
    "ADA": ["cardano", "ada"],
    "DOT": ["polkadot", "dot"],
    "LINK": ["chainlink"],
    "AVAX": ["avalanche", "avax"],
    "MATIC": ["polygon", "matic"],
    "XRP": ["ripple", "xrp"],
    "DOGE": ["dogecoin", "doge"],
}


class NewsScraper:
    def fetch_articles(self, asset: str, limit: int = 30) -> List[str]:
        """Fetch news article titles mentioning the asset from RSS feeds."""
        keywords = ASSET_KEYWORDS.get(asset.upper(), [asset.lower()])
        texts: List[str] = []

        try:
            import feedparser  # type: ignore
        except ImportError:
            logger.warning("feedparser not installed; news scraper disabled")
            return []

        for feed_url in NEWS_FEEDS:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries:
                    title = entry.get("title", "")
                    summary = entry.get("summary", "")
                    content = (title + " " + summary).lower()
                    if any(kw in content for kw in keywords):
                        texts.append(title[:500])
                        if len(texts) >= limit:
                            break
            except Exception as e:
                logger.debug("Feed parse error %s: %s", feed_url, e)
            if len(texts) >= limit:
                break

        return texts[:limit]
