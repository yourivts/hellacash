"""RSS/Atom news feed scraper with caching."""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Tuple

from bot.sentiment.keywords import get_keywords

logger = logging.getLogger(__name__)

NEWS_FEEDS = [
    # Major crypto outlets
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
    "https://www.theblock.co/rss.xml",
    "https://bitcoinmagazine.com/feed",
    "https://www.newsbtc.com/feed/",
    "https://ambcrypto.com/feed/",
    # General finance with crypto coverage
    "https://feeds.reuters.com/reuters/businessNews",
]

# Cache parsed feeds for 30 minutes — articles don't change that fast
CACHE_TTL = 1800
_feed_cache: Dict[str, Tuple[float, list]] = {}  # url → (timestamp, entries)


def _fetch_feed(url: str) -> list:
    """Fetch and cache an RSS feed."""
    now = time.time()
    cached = _feed_cache.get(url)
    if cached and (now - cached[0]) < CACHE_TTL:
        return cached[1]

    try:
        import feedparser  # type: ignore
    except ImportError:
        return []

    feed = feedparser.parse(url)
    entries = [
        {
            "title": e.get("title", ""),
            "summary": e.get("summary", ""),
        }
        for e in feed.entries
    ]
    _feed_cache[url] = (now, entries)
    return entries


class NewsScraper:
    def prefetch_feeds(self) -> None:
        """Fetch all RSS feeds once, populating the cache."""
        fetched = 0
        for url in NEWS_FEEDS:
            cached = _feed_cache.get(url)
            if cached and (time.time() - cached[0]) < CACHE_TTL:
                continue
            try:
                _fetch_feed(url)
                fetched += 1
            except Exception as e:
                logger.debug("Feed prefetch error %s: %s", url, e)
        logger.info("News prefetch done — %d feeds cached (%d new)", len(_feed_cache), fetched)

    def fetch_articles(self, asset: str, limit: int = 30) -> List[str]:
        """Filter cached news articles by asset keywords. No HTTP requests."""
        keywords = get_keywords(asset)
        texts: List[str] = []

        for url in NEWS_FEEDS:
            cached = _feed_cache.get(url)
            if not cached:
                # Try fetching if not cached yet
                try:
                    entries = _fetch_feed(url)
                except Exception:
                    continue
            else:
                entries = cached[1]

            for entry in entries:
                content = (entry["title"] + " " + entry["summary"]).lower()
                if any(kw in content for kw in keywords):
                    texts.append(entry["title"][:500])
                    if len(texts) >= limit:
                        return texts

        return texts[:limit]
