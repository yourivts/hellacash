"""Sentiment aggregator: fuses Reddit, news, and Fear & Greed into per-asset scores."""
from __future__ import annotations

import asyncio
import json
import logging
import urllib.request
import time
from datetime import datetime
from typing import Dict, List, Tuple

from bot.config import Settings
from bot.data.database import get_session
from bot.data.repositories import save_sentiment
from bot.events.bus import TOPIC_SENTIMENT_UPDATED, get_bus
from bot.sentiment.nlp import FinBERTScorer, VADERScorer
from bot.sentiment.scraper_news import NewsScraper
from bot.sentiment.scraper_reddit import RedditScraper
from bot.sentiment.scraper_fear_greed import FearGreedScraper

logger = logging.getLogger(__name__)

# Source weights for aggregate score
SOURCE_WEIGHTS = {"news": 0.40, "reddit": 0.35, "fear_greed": 0.25}

# Cache all Bitvavo EUR markets for 1 hour
_all_markets_cache: Dict[str, object] = {"symbols": [], "ts": 0}
_MARKETS_TTL = 3600


def _fetch_all_eur_symbols() -> List[str]:
    """Fetch all EUR trading markets from Bitvavo."""
    now = time.time()
    if _all_markets_cache["symbols"] and (now - _all_markets_cache["ts"]) < _MARKETS_TTL:
        return _all_markets_cache["symbols"]
    try:
        req = urllib.request.Request(
            "https://api.bitvavo.com/v2/markets",
            headers={"User-Agent": "hellacash/1.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            markets = json.loads(resp.read().decode())
        symbols = [
            m["market"] for m in markets
            if m.get("market", "").endswith("-EUR") and m.get("status") == "trading"
        ]
        _all_markets_cache["symbols"] = symbols
        _all_markets_cache["ts"] = now
        logger.info("Fetched %d EUR markets from Bitvavo for sentiment", len(symbols))
        return symbols
    except Exception as e:
        logger.error("Failed to fetch Bitvavo markets: %s", e)
        return _all_markets_cache["symbols"] or []


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
        self._fear_greed = FearGreedScraper()
        self._news = NewsScraper()
        self._vader = VADERScorer()
        self._finbert = FinBERTScorer()
        self._cache: Dict[str, float] = {}  # base asset → latest aggregate score
        self._bus = get_bus()

    def get_score(self, symbol: str) -> float:
        """Return cached aggregate sentiment score for a symbol (base asset)."""
        base = symbol.split("-")[0].upper()
        return self._cache.get(base, 0.0)

    async def run_cycle(self, assets: List[str]) -> None:
        """Scrape and score ALL Bitvavo EUR markets, not just active trading symbols."""
        loop = asyncio.get_event_loop()

        # Get all Bitvavo EUR markets
        all_symbols = await loop.run_in_executor(None, _fetch_all_eur_symbols)
        if not all_symbols:
            all_symbols = assets  # fallback to active symbols

        # Step 1: Pre-fetch ALL shared data sources ONCE
        fear_greed_score, _, _ = await asyncio.gather(
            loop.run_in_executor(None, self._fear_greed.get_score),
            loop.run_in_executor(None, self._reddit.prefetch_subs),
            loop.run_in_executor(None, self._news.prefetch_feeds),
        )

        # Step 2: Pre-filter — find which assets actually have mentions
        # This avoids running NLP on 400+ assets with zero mentions
        assets_with_mentions: List[Tuple[str, List[str], List[str]]] = []
        assets_without_mentions: List[str] = []

        for symbol in all_symbols:
            base = symbol.split("-")[0].upper()
            news_texts = self._news.fetch_articles(base, 30)
            reddit_texts = self._reddit.fetch_posts(base, 50)
            if news_texts or reddit_texts:
                assets_with_mentions.append((base, news_texts, reddit_texts))
            else:
                assets_without_mentions.append(base)

        logger.info(
            "Sentiment cycle: %d assets with mentions, %d with fear_greed only",
            len(assets_with_mentions), len(assets_without_mentions),
        )

        # Step 3: Score assets WITH mentions (run NLP)
        for base, news_texts, reddit_texts in assets_with_mentions:
            try:
                news_score = await loop.run_in_executor(None, self._finbert.score_batch, news_texts)
                reddit_score = await loop.run_in_executor(None, self._vader.score_batch, reddit_texts)

                agg = (
                    news_score * SOURCE_WEIGHTS["news"]
                    + reddit_score * SOURCE_WEIGHTS["reddit"]
                    + fear_greed_score * SOURCE_WEIGHTS["fear_greed"]
                )
                self._cache[base] = agg

                logger.info(
                    "Sentiment %s: news=%.3f(%d) reddit=%.3f(%d) fg=%.3f → agg=%.3f",
                    base, news_score, len(news_texts), reddit_score, len(reddit_texts),
                    fear_greed_score, agg,
                )

                await self._persist_and_publish(
                    base, news_score, reddit_score, fear_greed_score, agg,
                    news_texts, reddit_texts,
                )
            except Exception as e:
                logger.error("Sentiment scoring failed for %s: %s", base, e)

        # Step 4: Assets WITHOUT mentions just get Fear & Greed score
        now = datetime.utcnow()
        for base in assets_without_mentions:
            self._cache[base] = fear_greed_score * SOURCE_WEIGHTS["fear_greed"]

        logger.info("Sentiment cycle complete — %d total assets scored", len(self._cache))

    async def _persist_and_publish(
        self,
        asset: str,
        news_score: float,
        reddit_score: float,
        fear_greed_score: float,
        agg: float,
        news_texts: List[str],
        reddit_texts: List[str],
    ) -> None:
        now = datetime.utcnow()
        async with get_session() as session:
            for source, score, count, samples in [
                ("news", news_score, len(news_texts), news_texts[:3]),
                ("reddit", reddit_score, len(reddit_texts), reddit_texts[:3]),
                ("fear_greed", fear_greed_score, 1, []),
                ("aggregate", agg, len(news_texts) + len(reddit_texts) + 1, []),
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

        await self._bus.publish(TOPIC_SENTIMENT_UPDATED, {
            "asset": asset,
            "symbol": f"{asset}-EUR",
            "aggregate": agg,
            "news": news_score,
            "reddit": reddit_score,
            "fear_greed": fear_greed_score,
            "timestamp": now.isoformat(),
        })
