"""Sentiment aggregator: fuses Reddit, Twitter, and news into per-asset scores."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from bot.config import Settings
from bot.data.database import get_session
from bot.data.repositories import save_sentiment
from bot.events.bus import TOPIC_SENTIMENT_UPDATED, get_bus
from bot.sentiment.nlp import FinBERTScorer, VADERScorer
from bot.sentiment.scraper_news import NewsScraper
from bot.sentiment.scraper_reddit import RedditScraper
from bot.sentiment.scraper_twitter import TwitterScraper

logger = logging.getLogger(__name__)

# Source weights for aggregate score
SOURCE_WEIGHTS = {"news": 0.40, "reddit": 0.35, "twitter": 0.25}


class SentimentAggregator:
    """
    Orchestrates all sentiment scrapers and fuses scores per asset.
    Maintains an in-memory cache of the latest score per asset.
    """

    def __init__(self, settings: Settings) -> None:
        self._reddit = RedditScraper(
            settings.reddit_client_id,
            settings.reddit_client_secret,
            settings.reddit_user_agent,
        )
        self._twitter = TwitterScraper(settings.twitter_bearer_token)
        self._news = NewsScraper()
        self._vader = VADERScorer()
        self._finbert = FinBERTScorer()
        self._cache: Dict[str, float] = {}  # symbol → latest aggregate score
        self._bus = get_bus()

    def get_score(self, symbol: str) -> float:
        """Return cached aggregate sentiment score for a symbol (base asset)."""
        base = symbol.split("-")[0].upper()
        return self._cache.get(base, 0.0)

    async def run_cycle(self, assets: List[str]) -> None:
        """Scrape and score all assets. Call this periodically (e.g., every 10 min)."""
        for symbol in assets:
            base = symbol.split("-")[0].upper()
            await self._score_asset(base)

    async def _score_asset(self, asset: str) -> None:
        loop = asyncio.get_event_loop()

        # Run blocking scrapers in thread pool
        news_texts, reddit_texts, twitter_texts = await asyncio.gather(
            loop.run_in_executor(None, self._news.fetch_articles, asset, 30),
            loop.run_in_executor(None, self._reddit.fetch_posts, asset, 50),
            loop.run_in_executor(None, self._twitter.fetch_tweets, asset, 50),
        )

        # Score each source
        news_score = await loop.run_in_executor(None, self._finbert.score_batch, news_texts)
        reddit_score = await loop.run_in_executor(None, self._vader.score_batch, reddit_texts)
        twitter_score = await loop.run_in_executor(None, self._vader.score_batch, twitter_texts)

        # Weighted aggregate
        agg = (
            news_score * SOURCE_WEIGHTS["news"]
            + reddit_score * SOURCE_WEIGHTS["reddit"]
            + twitter_score * SOURCE_WEIGHTS["twitter"]
        )

        self._cache[asset] = agg

        logger.info(
            "Sentiment %s: news=%.3f reddit=%.3f twitter=%.3f → agg=%.3f",
            asset, news_score, reddit_score, twitter_score, agg,
        )

        # Persist to DB
        now = datetime.now(timezone.utc)
        async with get_session() as session:
            for source, score, count, samples in [
                ("news", news_score, len(news_texts), news_texts[:3]),
                ("reddit", reddit_score, len(reddit_texts), reddit_texts[:3]),
                ("twitter", twitter_score, len(twitter_texts), twitter_texts[:3]),
                ("aggregate", agg, len(news_texts) + len(reddit_texts) + len(twitter_texts), []),
            ]:
                await save_sentiment(
                    session,
                    symbol=f"{asset}-EUR",
                    source=source,
                    score=score,
                    post_count=count,
                    sample_texts={"samples": samples},
                    computed_at=now,
                )

        # Publish event
        await self._bus.publish(TOPIC_SENTIMENT_UPDATED, {
            "asset": asset,
            "symbol": f"{asset}-EUR",
            "aggregate": agg,
            "news": news_score,
            "reddit": reddit_score,
            "twitter": twitter_score,
            "timestamp": now.isoformat(),
        })
