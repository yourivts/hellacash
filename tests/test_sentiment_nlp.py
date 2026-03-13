"""Tests for bot.sentiment.nlp sentiment scoring."""
from __future__ import annotations

import pytest

from bot.sentiment.nlp import VADERScorer, _apply_crypto_boost


class TestVADERScorer:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.scorer = VADERScorer()
        if self.scorer._analyzer is None:
            pytest.skip("vaderSentiment not installed")

    def test_vader_scorer_positive(self):
        score = self.scorer.score("This is absolutely amazing and wonderful!")
        assert score > 0.0

    def test_vader_scorer_negative(self):
        score = self.scorer.score("This is terrible, awful, and disgusting.")
        assert score < 0.0


class TestCryptoBoost:
    def test_crypto_boost_bull_words(self):
        base_score = 0.0
        boosted = _apply_crypto_boost("Bitcoin is mooning, very bullish breakout!", base_score)
        assert boosted > 0.0

    def test_crypto_boost_bear_words(self):
        base_score = 0.0
        boosted = _apply_crypto_boost("crash dump rug scam panic sell", base_score)
        assert boosted < 0.0

    def test_crypto_boost_clamps_to_range(self):
        # Many bull words should not exceed 1.0
        boosted = _apply_crypto_boost(
            "moon mooning pump breakout ath bull bullish accumulate hodl",
            0.9,
        )
        assert boosted <= 1.0

        # Many bear words should not go below -1.0
        boosted = _apply_crypto_boost(
            "crash dump rug scam sell liquidation bear short panic bearish",
            -0.9,
        )
        assert boosted >= -1.0
