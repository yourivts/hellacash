"""Tests for sentiment source circuit breakers."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from bot.sentiment.aggregator import SentimentAggregator, SOURCE_WEIGHTS


def _make_aggregator() -> SentimentAggregator:
    settings = MagicMock()
    settings.reddit_client_id = ""
    settings.reddit_client_secret = ""
    settings.reddit_user_agent = "test"
    agg = SentimentAggregator(settings)
    return agg


class TestCircuitBreakerDisablesAfterFailures:
    def test_source_disabled_after_3_failures(self):
        agg = _make_aggregator()
        for _ in range(3):
            agg._record_failure("reddit")
        assert agg._is_source_disabled("reddit") is True

    def test_source_not_disabled_after_2_failures(self):
        agg = _make_aggregator()
        for _ in range(2):
            agg._record_failure("reddit")
        assert agg._is_source_disabled("reddit") is False


class TestCircuitBreakerRecovery:
    def test_source_recovers_after_cooldown(self):
        agg = _make_aggregator()
        for _ in range(3):
            agg._record_failure("reddit")
        assert agg._is_source_disabled("reddit") is True
        agg._source_disabled_until["reddit"] = datetime.now(timezone.utc) - timedelta(seconds=1)
        assert agg._is_source_disabled("reddit") is False


class TestReweighting:
    def test_reweight_with_one_source_disabled(self):
        agg = _make_aggregator()
        weights = agg._get_active_weights(disabled={"reddit"})
        assert abs(weights["news"] - 0.40 / 0.65) < 0.01
        assert abs(weights["fear_greed"] - 0.25 / 0.65) < 0.01
        assert "reddit" not in weights

    def test_all_sources_disabled_returns_empty(self):
        agg = _make_aggregator()
        weights = agg._get_active_weights(disabled={"news", "reddit", "fear_greed"})
        assert weights == {}


class TestRecordSuccess:
    def test_success_resets_failure_count(self):
        agg = _make_aggregator()
        for _ in range(2):
            agg._record_failure("news")
        agg._record_success("news")
        assert agg._source_failures.get("news", 0) == 0
