# tests/test_config_validation.py
"""Tests for config field validation and credential masking."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from bot.config import Settings


def _base_settings(**overrides) -> dict:
    defaults = {
        "bitvavo_api_key": "",
        "bitvavo_api_secret": "",
        "database_url": "postgresql+asyncpg://x:x@localhost/x",
    }
    defaults.update(overrides)
    return defaults


class TestCredentialMasking:
    def test_api_key_not_in_repr(self):
        s = Settings(**_base_settings(bitvavo_api_key="secret123"))
        assert "secret123" not in repr(s)

    def test_api_secret_not_in_repr(self):
        s = Settings(**_base_settings(bitvavo_api_secret="topsecret"))
        assert "topsecret" not in repr(s)

    def test_reddit_client_secret_not_in_repr(self):
        s = Settings(**_base_settings(reddit_client_secret="redditsecret"))
        assert "redditsecret" not in repr(s)

    def test_twitter_token_not_in_repr(self):
        s = Settings(**_base_settings(twitter_bearer_token="twittertoken"))
        assert "twittertoken" not in repr(s)


class TestRangeValidation:
    def test_kelly_fraction_too_high(self):
        with pytest.raises(ValidationError):
            Settings(**_base_settings(kelly_fraction=1.5))

    def test_kelly_fraction_negative(self):
        with pytest.raises(ValidationError):
            Settings(**_base_settings(kelly_fraction=-0.1))

    def test_kelly_fraction_valid(self):
        s = Settings(**_base_settings(kelly_fraction=0.25))
        assert s.kelly_fraction == 0.25

    def test_min_signal_confidence_too_high(self):
        with pytest.raises(ValidationError):
            Settings(**_base_settings(min_signal_confidence=1.5))

    def test_max_drawdown_pct_too_high(self):
        with pytest.raises(ValidationError):
            Settings(**_base_settings(max_drawdown_pct=101.0))

    def test_max_daily_loss_negative(self):
        with pytest.raises(ValidationError):
            Settings(**_base_settings(max_daily_loss_eur=-50.0))

    def test_max_position_size_pct_valid(self):
        s = Settings(**_base_settings(max_position_size_pct=20.0))
        assert s.max_position_size_pct == 20.0


class TestNewConfigFields:
    def test_mtf_weights_defaults(self):
        s = Settings(database_url="postgresql+asyncpg://x:x@localhost/x")
        assert s.mtf_tech_weight == 0.55
        assert s.mtf_sent_weight == 0.20
        assert s.mtf_onchain_weight == 0.15
        assert s.mtf_book_weight == 0.10

    def test_onchain_defaults(self):
        s = Settings(database_url="postgresql+asyncpg://x:x@localhost/x")
        assert s.onchain_enabled is True
        assert s.onchain_poll_interval_secs == 300

    def test_orderbook_defaults(self):
        s = Settings(database_url="postgresql+asyncpg://x:x@localhost/x")
        assert s.orderbook_enabled is True
        assert s.orderbook_depth_levels == 25
