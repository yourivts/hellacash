"""Tests for auto-enabling paper trading after walk-forward completes."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.learning.walk_forward import WFResult, WFWindow


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


class TestAutoEnablePaperTrading:

    @pytest.mark.asyncio
    async def test_auto_starts_paper_trading_after_walk_forward(self, mock_settings, mock_discord):
        """Paper trading is auto-enabled when walk-forward finishes and bot is not running."""
        from bot.main import _process_walk_forward_results

        results = {"BTC-EUR": _make_wf_result(adopted=True)}

        with patch("bot.main.is_running", return_value=False), \
             patch("bot.main.start_bot", new_callable=AsyncMock) as mock_start:
            await _process_walk_forward_results(
                results, get_router_fn=MagicMock, trading_loop=None,
                discord=mock_discord, settings=mock_settings,
            )
            mock_start.assert_called_once()

    @pytest.mark.asyncio
    async def test_does_not_start_if_already_running(self, mock_settings, mock_discord):
        """If bot is already running, don't call start_bot again."""
        from bot.main import _process_walk_forward_results

        results = {"BTC-EUR": _make_wf_result(adopted=True)}

        with patch("bot.main.is_running", return_value=True), \
             patch("bot.main.start_bot", new_callable=AsyncMock) as mock_start:
            await _process_walk_forward_results(
                results, get_router_fn=MagicMock, trading_loop=None,
                discord=mock_discord, settings=mock_settings,
            )
            mock_start.assert_not_called()

    @pytest.mark.asyncio
    async def test_does_not_start_in_live_mode(self, mock_discord):
        """Live trading mode should NOT auto-start."""
        from bot.main import _process_walk_forward_results

        live_settings = MagicMock()
        live_settings.paper_trading = False

        results = {"BTC-EUR": _make_wf_result(adopted=True)}

        with patch("bot.main.is_running", return_value=False), \
             patch("bot.main.start_bot", new_callable=AsyncMock) as mock_start:
            await _process_walk_forward_results(
                results, get_router_fn=MagicMock, trading_loop=None,
                discord=mock_discord, settings=live_settings,
            )
            mock_start.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_starts_even_when_no_params_adopted(self, mock_settings, mock_discord):
        """Auto-start happens regardless of whether params were adopted."""
        from bot.main import _process_walk_forward_results

        results = {"BTC-EUR": _make_wf_result(adopted=False)}

        with patch("bot.main.is_running", return_value=False), \
             patch("bot.main.start_bot", new_callable=AsyncMock) as mock_start:
            await _process_walk_forward_results(
                results, get_router_fn=MagicMock, trading_loop=None,
                discord=mock_discord, settings=mock_settings,
            )
            mock_start.assert_called_once()
