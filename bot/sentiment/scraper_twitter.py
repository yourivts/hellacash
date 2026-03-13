"""Twitter/X scraper using tweepy v2 (snscrape fallback)."""
from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)

ASSET_CASHTAGS = {
    "BTC": ["$BTC", "bitcoin"],
    "ETH": ["$ETH", "ethereum"],
    "SOL": ["$SOL", "solana"],
    "ADA": ["$ADA", "cardano"],
    "DOT": ["$DOT", "polkadot"],
    "LINK": ["$LINK", "chainlink"],
    "AVAX": ["$AVAX", "avalanche"],
    "MATIC": ["$MATIC", "polygon"],
    "XRP": ["$XRP", "ripple"],
    "DOGE": ["$DOGE", "dogecoin"],
}


class TwitterScraper:
    def __init__(self, bearer_token: str = "") -> None:
        self._client = None
        self._bearer_token = bearer_token

        if bearer_token:
            try:
                import tweepy  # type: ignore
                self._client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)
                logger.info("Tweepy v2 client initialized")
            except ImportError:
                logger.warning("tweepy not installed; trying snscrape fallback")

    def fetch_tweets(self, asset: str, limit: int = 50) -> List[str]:
        """Fetch recent tweets mentioning the asset."""
        cashtags = ASSET_CASHTAGS.get(asset.upper(), [asset])
        query = " OR ".join(cashtags) + " lang:en -is:retweet"

        # Try tweepy first
        if self._client is not None:
            return self._fetch_tweepy(query, limit)

        # Fallback: snscrape (no auth needed)
        return self._fetch_snscrape(query, limit)

    def _fetch_tweepy(self, query: str, limit: int) -> List[str]:
        try:
            resp = self._client.search_recent_tweets(
                query=query,
                max_results=min(limit, 100),
                tweet_fields=["text"],
            )
            if resp.data:
                return [t.text[:280] for t in resp.data]
        except Exception as e:
            logger.error("Tweepy fetch error: %s", e)
        return []

    def _fetch_snscrape(self, query: str, limit: int) -> List[str]:
        try:
            import snscrape.modules.twitter as sntwitter  # type: ignore
            tweets = []
            for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
                if i >= limit:
                    break
                tweets.append(tweet.rawContent[:280])
            return tweets
        except ImportError:
            logger.debug("snscrape not installed; no Twitter data")
        except Exception as e:
            logger.error("snscrape error: %s", e)
        return []

    @property
    def available(self) -> bool:
        return True  # always at least try snscrape
