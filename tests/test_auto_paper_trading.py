"""Tests for auto-enabling paper trading after walk-forward completes."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.learning.walk_forward import WFResult, WFWindow
from bot.strategy.adopted_universe import AdoptedUniverse


@pytest.fixture
def mock_settings():
    s = MagicMock()
    s.paper_trading = True
    return s


@pytest.fixture
def mock_discord():
    d = AsyncMock()
    d.send_walk_forward_report = AsyncMock()
    return d


@pytest.fixture
def universe():
    return AdoptedUniverse()


def _make_wf_result(adopted=True, avg_sharpe=1.0, avg_pnl=50.0):
    return WFResult(
        windows=[
            WFWindow(sharpe=avg_sharpe, pnl=avg_pnl, params={"entry_threshold": 0.40}),
            WFWindow(sharpe=avg_sharpe, pnl=avg_pnl, params={"entry_threshold": 0.40}),
            WFWindow(sharpe=avg_sharpe, pnl=avg_pnl, params={"entry_threshold": 0.40}),
        ],
        avg_oos_sharpe=avg_sharpe,
        avg_oos_pnl=avg_pnl,
        sharpe_stability=0.1,
        recommended_params={"entry_threshold": 0.40} if adopted else None,
        adopted=adopted,
    )


def _make_nested_results(adopted=True):
    """Create nested {symbol: {strategy: WFResult}} format."""
    return {
        "BTC-EUR": {
            "orderflow": _make_wf_result(adopted=adopted),
            "range": _make_wf_result(adopted=False),
            "squeeze": _make_wf_result(adopted=False),
            "funding_contrarian": _make_wf_result(adopted=False),
        },
    }


class TestAutoEnablePaperTrading:

    @pytest.mark.asyncio
    async def test_auto_starts_paper_trading_after_walk_forward(self, mock_settings, mock_discord, universe):
        """Paper trading is auto-enabled when walk-forward finishes and bot is not running."""
        from bot.main import _process_walk_forward_results

        results = _make_nested_results(adopted=True)

        with patch("bot.main.is_running", return_value=False), \
             patch("bot.main.start_bot", new_callable=AsyncMock) as mock_start:
            await _process_walk_forward_results(
                results, get_router_fn=MagicMock, trading_loop=None,
                discord=mock_discord, settings=mock_settings, universe=universe,
            )
            mock_start.assert_called_once()

    @pytest.mark.asyncio
    async def test_does_not_start_if_already_running(self, mock_settings, mock_discord, universe):
        """If bot is already running, don't call start_bot again."""
        from bot.main import _process_walk_forward_results

        results = _make_nested_results(adopted=True)

        with patch("bot.main.is_running", return_value=True), \
             patch("bot.main.start_bot", new_callable=AsyncMock) as mock_start:
            await _process_walk_forward_results(
                results, get_router_fn=MagicMock, trading_loop=None,
                discord=mock_discord, settings=mock_settings, universe=universe,
            )
            mock_start.assert_not_called()

    @pytest.mark.asyncio
    async def test_does_not_start_in_live_mode(self, mock_discord, universe):
        """Live trading mode should NOT auto-start."""
        from bot.main import _process_walk_forward_results

        live_settings = MagicMock()
        live_settings.paper_trading = False

        results = _make_nested_results(adopted=True)

        with patch("bot.main.is_running", return_value=False), \
             patch("bot.main.start_bot", new_callable=AsyncMock) as mock_start:
            await _process_walk_forward_results(
                results, get_router_fn=MagicMock, trading_loop=None,
                discord=mock_discord, settings=live_settings, universe=universe,
            )
            mock_start.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_starts_even_when_no_params_adopted(self, mock_settings, mock_discord, universe):
        """Auto-start happens regardless of whether params were adopted."""
        from bot.main import _process_walk_forward_results

        results = _make_nested_results(adopted=False)

        with patch("bot.main.is_running", return_value=False), \
             patch("bot.main.start_bot", new_callable=AsyncMock) as mock_start:
            await _process_walk_forward_results(
                results, get_router_fn=MagicMock, trading_loop=None,
                discord=mock_discord, settings=mock_settings, universe=universe,
            )
            mock_start.assert_called_once()
