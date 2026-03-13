"""NLP sentiment scoring: VADER for social + optional FinBERT for news."""
from __future__ import annotations

import logging
import re
from typing import List

logger = logging.getLogger(__name__)

# Crypto-specific keyword boosters
BULL_WORDS = {"bullish", "moon", "mooning", "pump", "breakout", "ath", "bull", "accumulate", "hodl"}
BEAR_WORDS = {"bearish", "crash", "dump", "rug", "scam", "sell", "liquidation", "bear", "short", "panic"}


def _apply_crypto_boost(text: str, score: float) -> float:
    words = set(re.findall(r"\b\w+\b", text.lower()))
    bull_hits = len(words & BULL_WORDS)
    bear_hits = len(words & BEAR_WORDS)
    boost = (bull_hits - bear_hits) * 0.05
    return max(-1.0, min(1.0, score + boost))


class VADERScorer:
    def __init__(self) -> None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
            self._analyzer = SentimentIntensityAnalyzer()
            logger.info("VADER sentiment analyzer loaded")
        except ImportError:
            self._analyzer = None
            logger.warning("vaderSentiment not installed; returning neutral scores")

    def score(self, text: str) -> float:
        """Returns compound score: -1.0 (very negative) to +1.0 (very positive)."""
        if self._analyzer is None:
            return 0.0
        raw = self._analyzer.polarity_scores(text)["compound"]
        return _apply_crypto_boost(text, raw)

    def score_batch(self, texts: List[str]) -> float:
        """Average score across a list of texts."""
        if not texts:
            return 0.0
        scores = [self.score(t) for t in texts]
        return sum(scores) / len(scores)


_finbert_pipe = None
_finbert_loaded = False


def _get_finbert_pipe():
    global _finbert_pipe, _finbert_loaded
    if _finbert_loaded:
        return _finbert_pipe
    _finbert_loaded = True
    try:
        from transformers import pipeline  # type: ignore
        _finbert_pipe = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            truncation=True,
            max_length=512,
        )
        logger.info("FinBERT pipeline loaded")
    except Exception as e:
        logger.warning("FinBERT unavailable (%s); falling back to VADER for news", e)
    return _finbert_pipe


class FinBERTScorer:
    """
    Uses ProsusAI/finbert for news articles (finance-domain sentiment).
    Falls back to VADER if transformers is not installed.
    """

    def __init__(self) -> None:
        self._pipe = _get_finbert_pipe()
        if self._pipe is None:
            self._vader = VADERScorer()

    def score(self, text: str) -> float:
        if self._pipe is None:
            return self._vader.score(text)
        try:
            result = self._pipe(text[:512])[0]
            label = result["label"].lower()
            conf = result["score"]
            if label == "positive":
                return conf
            elif label == "negative":
                return -conf
            return 0.0
        except Exception:
            return 0.0

    def score_batch(self, texts: List[str]) -> float:
        if not texts:
            return 0.0
        scores = [self.score(t[:512]) for t in texts]
        return sum(scores) / len(scores)
