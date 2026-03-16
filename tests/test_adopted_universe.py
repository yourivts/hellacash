"""Tests for bot.strategy.adopted_universe."""
from __future__ import annotations

import unittest.mock
import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from bot.strategy.adopted_universe import (
    AdoptedUniverse, SymbolConfig, StrategyConfig,
    CHAMPION_DEFAULTS, ALL_STRATEGIES,
)


def _mock_wf_result(adopted=True, sharpe=1.0, pnl=100.0, params=None):
    """Create a mock WFResult."""
    r = MagicMock()
    r.adopted = adopted
    r.recommended_params = params or dict(CHAMPION_DEFAULTS)
    r.avg_oos_sharpe = sharpe
    r.avg_oos_pnl = pnl
    return r


class TestEmptyUniverse:
    def test_empty_blocks_all(self):
        u = AdoptedUniverse()
        assert u.is_adopted("BTC-EUR") is False
        assert u.get_enabled_strategies("BTC-EUR") == []
        assert u.adopted_symbols() == []
        assert u.count() == 0

    def test_get_strategy_params_fallback(self):
        u = AdoptedUniverse()
        params = u.get_strategy_params("BTC-EUR", "orderflow")
        assert params == CHAMPION_DEFAULTS

    def test_get_regime_params_fallback(self):
        u = AdoptedUniverse()
        rp = u.get_regime_params("BTC-EUR")
        assert rp["quiet_atr_threshold"] == CHAMPION_DEFAULTS["quiet_atr_threshold"]
        assert rp["regime_adx_threshold"] == CHAMPION_DEFAULTS["regime_adx_threshold"]


class TestUpdate:
    def test_mixed_adopted_rejected(self):
        u = AdoptedUniverse()
        results = {
            "BTC-EUR": {
                "orderflow": _mock_wf_result(adopted=True, sharpe=1.5, pnl=3000),
                "range": _mock_wf_result(adopted=False),
                "squeeze": _mock_wf_result(adopted=True, sharpe=0.8, pnl=500),
                "funding_contrarian": _mock_wf_result(adopted=False),
            },
            "DOGE-EUR": {
                "orderflow": _mock_wf_result(adopted=False),
                "range": _mock_wf_result(adopted=False),
                "squeeze": _mock_wf_result(adopted=False),
                "funding_contrarian": _mock_wf_result(adopted=False),
            },
        }
        u.update(results)
        assert u.is_adopted("BTC-EUR") is True
        assert u.is_adopted("DOGE-EUR") is False
        assert set(u.get_enabled_strategies("BTC-EUR")) == {"orderflow", "squeeze"}
        assert u.count() == 1

    def test_per_strategy_params(self):
        of_params = dict(CHAMPION_DEFAULTS, atr_multiplier=3.0, rr_ratio=2.5)
        sq_params = dict(CHAMPION_DEFAULTS, atr_multiplier=4.5, rr_ratio=3.5)
        u = AdoptedUniverse()
        results = {
            "BTC-EUR": {
                "orderflow": _mock_wf_result(adopted=True, params=of_params),
                "squeeze": _mock_wf_result(adopted=True, params=sq_params),
                "range": _mock_wf_result(adopted=False),
                "funding_contrarian": _mock_wf_result(adopted=False),
            },
        }
        u.update(results)
        assert u.get_strategy_params("BTC-EUR", "orderflow")["atr_multiplier"] == 3.0
        assert u.get_strategy_params("BTC-EUR", "squeeze")["atr_multiplier"] == 4.5

    def test_regime_params_from_best_strategy(self):
        params_high = dict(CHAMPION_DEFAULTS, quiet_atr_threshold=1.3, regime_adx_threshold=26)
        params_low = dict(CHAMPION_DEFAULTS, quiet_atr_threshold=0.9, regime_adx_threshold=22)
        u = AdoptedUniverse()
        results = {
            "BTC-EUR": {
                "orderflow": _mock_wf_result(adopted=True, sharpe=2.0, params=params_high),
                "range": _mock_wf_result(adopted=True, sharpe=0.5, params=params_low),
                "squeeze": _mock_wf_result(adopted=False),
                "funding_contrarian": _mock_wf_result(adopted=False),
            },
        }
        u.update(results)
        rp = u.get_regime_params("BTC-EUR")
        assert rp["quiet_atr_threshold"] == 1.3  # from orderflow (higher sharpe)
        assert rp["regime_adx_threshold"] == 26

    def test_zero_adoption_preserves_previous(self):
        u = AdoptedUniverse()
        # First update with adoption
        results1 = {
            "BTC-EUR": {
                "orderflow": _mock_wf_result(adopted=True),
                "range": _mock_wf_result(adopted=False),
                "squeeze": _mock_wf_result(adopted=False),
                "funding_contrarian": _mock_wf_result(adopted=False),
            },
        }
        u.update(results1)
        assert u.count() == 1

        # Second update with zero adoptions
        results2 = {
            "BTC-EUR": {
                "orderflow": _mock_wf_result(adopted=False),
                "range": _mock_wf_result(adopted=False),
                "squeeze": _mock_wf_result(adopted=False),
                "funding_contrarian": _mock_wf_result(adopted=False),
            },
        }
        u.update(results2)
        assert u.count() == 1  # preserved previous

    def test_symbol_zero_strategies_not_adopted(self):
        u = AdoptedUniverse()
        results = {
            "BTC-EUR": {
                "orderflow": _mock_wf_result(adopted=False),
                "range": _mock_wf_result(adopted=False),
                "squeeze": _mock_wf_result(adopted=False),
                "funding_contrarian": _mock_wf_result(adopted=False),
            },
        }
        u.update(results)
        assert u.is_adopted("BTC-EUR") is False


class TestDiffLogging:
    def test_update_logs_added_removed(self):
        import logging
        u = AdoptedUniverse()
        # First: adopt BTC
        results1 = {
            "BTC-EUR": {
                "orderflow": _mock_wf_result(adopted=True),
                "range": _mock_wf_result(adopted=False),
                "squeeze": _mock_wf_result(adopted=False),
                "funding_contrarian": _mock_wf_result(adopted=False),
            },
        }
        u.update(results1)
        # Second: drop BTC, add ETH
        results2 = {
            "ETH-EUR": {
                "orderflow": _mock_wf_result(adopted=True),
                "range": _mock_wf_result(adopted=False),
                "squeeze": _mock_wf_result(adopted=False),
                "funding_contrarian": _mock_wf_result(adopted=False),
            },
        }
        with unittest.mock.patch("bot.strategy.adopted_universe.logger") as mock_logger:
            u.update(results2)
            log_msg = mock_logger.info.call_args[0][0] % mock_logger.info.call_args[0][1:]
            assert "+ETH-EUR" in log_msg
            assert "-BTC-EUR" in log_msg


class TestAdoptedSymbols:
    def test_returns_correct_list(self):
        u = AdoptedUniverse()
        results = {
            "BTC-EUR": {
                "orderflow": _mock_wf_result(adopted=True),
                "range": _mock_wf_result(adopted=False),
                "squeeze": _mock_wf_result(adopted=False),
                "funding_contrarian": _mock_wf_result(adopted=False),
            },
            "ETH-EUR": {
                "orderflow": _mock_wf_result(adopted=False),
                "range": _mock_wf_result(adopted=True),
                "squeeze": _mock_wf_result(adopted=False),
                "funding_contrarian": _mock_wf_result(adopted=False),
            },
        }
        u.update(results)
        assert sorted(u.adopted_symbols()) == ["BTC-EUR", "ETH-EUR"]
