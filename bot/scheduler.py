"""BotScheduler — manages all periodic async loops."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)


class BotScheduler:
    """
    Creates and manages all periodic background loops.

    Uses callbacks (not concrete types) for loose coupling. Each callback
    is a coroutine or callable provided by the orchestrator.
    """

    def __init__(
        self,
        run_cycle: Callable,
        sentiment_cycle: Callable,
        portfolio_snapshot: Callable,
        optimizer_run: Callable,
        is_running: Callable[[], bool],
        active_symbols: Callable[[], List[str]],
        settings: Any,
        discord_run: Optional[Callable] = None,
        walk_forward_run: Optional[Callable] = None,
    ) -> None:
        self._run_cycle = run_cycle
        self._sentiment_cycle = sentiment_cycle
        self._portfolio_snapshot = portfolio_snapshot
        self._optimizer_run = optimizer_run
        self._is_running = is_running
        self._active_symbols = active_symbols
        self._settings = settings
        self._discord_run = discord_run
        self._walk_forward_run = walk_forward_run
        self._tasks: List[asyncio.Task] = []

    async def run_all(self) -> None:
        """Create all periodic loop tasks and wait for them."""
        settings = self._settings
        coros = [
            self._strategy_loop(settings.strategy_cycle_secs),
            self._sentiment_loop(settings.sentiment_poll_interval_secs),
            self._portfolio_snapshot_loop(300),
            self._optimizer_loop(24),
        ]
        if self._discord_run is not None:
            coros.append(self._discord_run())
        if self._walk_forward_run is not None:
            coros.append(self._walk_forward_loop(168))  # weekly
        self._tasks = [asyncio.create_task(c) for c in coros]
        await asyncio.gather(*self._tasks)

    def shutdown(self) -> None:
        """Cancel all running tasks."""
        for task in self._tasks:
            task.cancel()
        logger.info("BotScheduler: all tasks cancelled")

    # ── Private loop methods ──────────────────────────────────────────────────

    async def _strategy_loop(self, interval_secs: int) -> None:
        while True:
            if self._is_running():
                await self._run_cycle()
            await asyncio.sleep(interval_secs)

    async def _sentiment_loop(self, interval_secs: int) -> None:
        while True:
            symbols = self._active_symbols()
            if symbols:  # always collect sentiment, even before trading activated
                try:
                    await self._sentiment_cycle(symbols)
                except Exception as e:
                    logger.error("Sentiment loop error: %s", e)
            await asyncio.sleep(interval_secs)

    async def _portfolio_snapshot_loop(self, interval_secs: int) -> None:
        while True:
            if self._is_running():
                try:
                    await self._portfolio_snapshot()
                except Exception as e:
                    logger.error("Snapshot error: %s", e)
            await asyncio.sleep(interval_secs)

    async def _optimizer_loop(self, interval_hours: int) -> None:
        while True:
            await asyncio.sleep(interval_hours * 3600)
            if self._is_running():
                try:
                    await self._optimizer_run()
                except Exception as e:
                    logger.error("Optimizer error: %s", e)

    async def _walk_forward_loop(self, interval_hours: int) -> None:
        # Run once on startup (after 60s warmup for candles to load), then weekly
        await asyncio.sleep(60)
        while True:
            try:
                logger.info("Walk-forward optimization starting...")
                await self._walk_forward_run()
            except Exception as e:
                logger.error("Walk-forward error: %s", e)
            await asyncio.sleep(interval_hours * 3600)
