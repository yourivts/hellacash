"""Sentiment aggregator: fuses Reddit, news, and Fear & Greed into per-asset scores."""
from __future__ import annotations

import asyncio
import json
import logging
import urllib.request
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Set, Tuple

from bot.config import Settings
from bot.data.database import get_session
from bot.data.repositories import save_sentiment
from bot.events.bus import TOPIC_SENTIMENT_UPDATED, TOPIC_SENTIMENT_DEGRADED, get_bus
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
        self._source_failures: Dict[str, int] = {}
        self._source_disabled_until: Dict[str, datetime] = {}
        self._max_failures = 3
        self._cooldown_secs = 600

    def _record_failure(self, source: str) -> None:
        self._source_failures[source] = self._source_failures.get(source, 0) + 1
        if self._source_failures[source] >= self._max_failures:
            until = datetime.now(timezone.utc) + timedelta(seconds=self._cooldown_secs)
            self._source_disabled_until[source] = until
            logger.warning("Sentiment source '%s' disabled until %s", source, until.isoformat())
            try:
                active = [s for s in SOURCE_WEIGHTS if not self._is_source_disabled(s)]
                asyncio.get_running_loop().create_task(
                    self._bus.publish(TOPIC_SENTIMENT_DEGRADED, {
                        "source": source, "status": "disabled",
                        "active_sources": active,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                )
            except RuntimeError:
                pass

    def _record_success(self, source: str) -> None:
        was_disabled = source in self._source_disabled_until
        self._source_failures[source] = 0
        self._source_disabled_until.pop(source, None)
        if was_disabled:
            try:
                active = [s for s in SOURCE_WEIGHTS if not self._is_source_disabled(s)]
                asyncio.get_running_loop().create_task(
                    self._bus.publish(TOPIC_SENTIMENT_DEGRADED, {
                        "source": source, "status": "recovered",
                        "active_sources": active,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                )
            except RuntimeError:
                pass

    def _is_source_disabled(self, source: str) -> bool:
        until = self._source_disabled_until.get(source)
        if until is None:
            return False
        if datetime.now(timezone.utc) >= until:
            self._source_disabled_until.pop(source, None)
            self._source_failures[source] = 0
            logger.info("Sentiment source '%s' recovered (cooldown expired)", source)
            return False
        return True

    def _get_active_weights(self, disabled: set) -> Dict[str, float]:
        active = {k: v for k, v in SOURCE_WEIGHTS.items() if k not in disabled}
        if not active:
            return {}
        total = sum(active.values())
        return {k: v / total for k, v in active.items()}

    def get_score(self, symbol: str) -> float:
        """Return cached aggregate sentiment score for a symbol (base asset)."""
        base = symbol.split("-")[0].upper()
        return self._cache.get(base, 0.0)

    async def run_cycle(self, assets: List[str]) -> None:
        """Scrape and score only tradeable symbols."""
        loop = asyncio.get_event_loop()

        all_symbols = assets

        # Update Reddit scraper's sub list to match tradeable symbols
        self._reddit.set_tradeable_symbols(all_symbols)

        # Step 1: Pre-fetch ALL shared data sources ONCE, guarded by circuit breakers
        fear_greed_score = 0.0
        disabled_sources: Set[str] = set()

        if not self._is_source_disabled("fear_greed"):
            try:
                fear_greed_score = await loop.run_in_executor(None, self._fear_greed.get_score)
                self._record_success("fear_greed")
            except Exception as e:
                logger.error("Fear & Greed fetch failed: %s", e)
                self._record_failure("fear_greed")
                disabled_sources.add("fear_greed")
        else:
            disabled_sources.add("fear_greed")
            logger.warning("Sentiment source 'fear_greed' is disabled (circuit breaker open)")

        if not self._is_source_disabled("reddit"):
            try:
                await loop.run_in_executor(None, self._reddit.prefetch_subs)
                self._record_success("reddit")
            except Exception as e:
                logger.error("Reddit prefetch failed: %s", e)
                self._record_failure("reddit")
                disabled_sources.add("reddit")
        else:
            disabled_sources.add("reddit")
            logger.warning("Sentiment source 'reddit' is disabled (circuit breaker open)")

        if not self._is_source_disabled("news"):
            try:
                await loop.run_in_executor(None, self._news.prefetch_feeds)
                self._record_success("news")
            except Exception as e:
                logger.error("News prefetch failed: %s", e)
                self._record_failure("news")
                disabled_sources.add("news")
        else:
            disabled_sources.add("news")
            logger.warning("Sentiment source 'news' is disabled (circuit breaker open)")

        # Step 2: Pre-filter — find which assets actually have mentions
        # This avoids running NLP on 400+ assets with zero mentions
        assets_with_mentions: List[Tuple[str, List[str], List[str]]] = []
        assets_without_mentions: List[str] = []

        for symbol in all_symbols:
            base = symbol.split("-")[0].upper()
            news_texts = [] if "news" in disabled_sources else self._news.fetch_articles(base, 30)
            reddit_texts = [] if "reddit" in disabled_sources else self._reddit.fetch_posts(base, 50)
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
                source_scores: Dict[str, float] = {}

                if "news" not in disabled_sources and news_texts:
                    try:
                        news_score = await loop.run_in_executor(
                            None, self._finbert.score_batch, news_texts
                        )
                        source_scores["news"] = news_score
                    except Exception as e:
                        logger.error("News NLP scoring failed for %s: %s", base, e)
                        self._record_failure("news")
                        disabled_sources.add("news")

                if "reddit" not in disabled_sources and reddit_texts:
                    try:
                        reddit_score = await loop.run_in_executor(
                            None, self._vader.score_batch, reddit_texts
                        )
                        source_scores["reddit"] = reddit_score
                    except Exception as e:
                        logger.error("Reddit NLP scoring failed for %s: %s", base, e)
                        self._record_failure("reddit")
                        disabled_sources.add("reddit")

                if "fear_greed" not in disabled_sources:
                    source_scores["fear_greed"] = fear_greed_score

                active_weights = self._get_active_weights(disabled=disabled_sources)
                if not active_weights:
                    logger.warning("All sentiment sources disabled for %s, skipping", base)
                    continue

                agg = sum(
                    source_scores.get(src, 0.0) * w
                    for src, w in active_weights.items()
                )
                self._cache[base] = agg

                news_score_log = source_scores.get("news", float("nan"))
                reddit_score_log = source_scores.get("reddit", float("nan"))
                logger.info(
                    "Sentiment %s: news=%.3f(%d) reddit=%.3f(%d) fg=%.3f → agg=%.3f",
                    base, news_score_log, len(news_texts), reddit_score_log, len(reddit_texts),
                    fear_greed_score, agg,
                )

                await self._persist_and_publish(
                    base,
                    source_scores.get("news", 0.0),
                    source_scores.get("reddit", 0.0),
                    fear_greed_score,
                    agg,
                    news_texts,
                    reddit_texts,
                )
            except Exception as e:
                logger.error("Sentiment scoring failed for %s: %s", base, e)

        # Step 4: Assets WITHOUT mentions just get Fear & Greed score
        now = datetime.utcnow()
        fg_weight = self._get_active_weights(disabled=disabled_sources).get("fear_greed", 0.0)
        for base in assets_without_mentions:
            self._cache[base] = fear_greed_score * fg_weight

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
