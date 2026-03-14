"""Walk-forward optimization — rolling out-of-sample parameter validation."""
from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

TRAIN_DAYS = 60
TEST_DAYS = 14
STEP_DAYS = 14
MIN_WINDOWS = 3
MIN_OOS_SHARPE = 0.5
MAX_SHARPE_STD = 0.5


@dataclass
class WFWindow:
    sharpe: float = 0.0
    pnl: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)
    train_start: Optional[datetime] = None
    train_end: Optional[datetime] = None
    test_start: Optional[datetime] = None
    test_end: Optional[datetime] = None


@dataclass
class WFResult:
    windows: List[WFWindow] = field(default_factory=list)
    avg_oos_sharpe: float = 0.0
    avg_oos_pnl: float = 0.0
    sharpe_stability: float = 0.0
    recommended_params: Optional[Dict[str, Any]] = None
    adopted: bool = False


class WalkForwardOptimizer:
    """Rolling walk-forward optimization using BacktestEngine."""

    def __init__(self) -> None:
        self._latest_result: Optional[WFResult] = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def latest_result(self) -> Optional[WFResult]:
        return self._latest_result

    def _generate_windows(self, total_days: int) -> List[dict]:
        windows = []
        offset = 0
        while offset + TRAIN_DAYS + TEST_DAYS <= total_days:
            windows.append({
                "train_start_day": offset,
                "train_end_day": offset + TRAIN_DAYS,
                "test_start_day": offset + TRAIN_DAYS,
                "test_end_day": offset + TRAIN_DAYS + TEST_DAYS,
            })
            offset += STEP_DAYS
        return windows

    def _should_adopt(self, windows: List[WFWindow]) -> bool:
        if len(windows) < MIN_WINDOWS:
            return False
        sharpes = [w.sharpe for w in windows]
        pnls = [w.pnl for w in windows]
        avg_sharpe = statistics.mean(sharpes)
        avg_pnl = statistics.mean(pnls)
        sharpe_std = statistics.stdev(sharpes) if len(sharpes) > 1 else 999
        if avg_sharpe < MIN_OOS_SHARPE:
            return False
        if sharpe_std > MAX_SHARPE_STD:
            return False
        if avg_pnl <= 0:
            return False
        return True

    async def run(self, candle_fetcher=None) -> WFResult:
        """Execute full walk-forward optimization."""
        self._running = True
        try:
            result = await self._execute(candle_fetcher)
            self._latest_result = result
            return result
        finally:
            self._running = False

    async def _execute(self, candle_fetcher) -> WFResult:
        import asyncio
        from bot.backtest.engine import BacktestEngine

        total_days = 120
        windows_spec = self._generate_windows(total_days)

        if len(windows_spec) < MIN_WINDOWS:
            logger.warning("Walk-forward: insufficient data for %d windows", MIN_WINDOWS)
            return WFResult()

        candidates = [
            {"sentiment_weight": 0.15, "entry_threshold": 0.30},
            {"sentiment_weight": 0.20, "entry_threshold": 0.35},
            {"sentiment_weight": 0.25, "entry_threshold": 0.40},
            {"sentiment_weight": 0.30, "entry_threshold": 0.35},
            {"sentiment_weight": 0.10, "entry_threshold": 0.30},
        ]

        loop = asyncio.get_running_loop()
        wf_windows: List[WFWindow] = []

        for i, ws in enumerate(windows_spec):
            if candle_fetcher is None:
                break

            train_candles = await candle_fetcher(ws["train_start_day"], ws["train_end_day"])
            test_candles = await candle_fetcher(ws["test_start_day"], ws["test_end_day"])

            if not train_candles or not test_candles:
                continue

            logger.info("Walk-forward: running window %d/%d (%d train, %d test candles, %d candidates in parallel)...",
                        i + 1, len(windows_spec), len(train_candles), len(test_candles), len(candidates))

            # Run all candidate backtests in parallel across CPU cores
            from concurrent.futures import ProcessPoolExecutor

            def _run_single_backtest(candles, params):
                from bot.backtest.engine import BacktestEngine as _BE
                engine = _BE(candles, strategy_params=params, slippage_pct=0.001)
                result = engine.run()
                return params, result.total_pnl

            def _run_window_parallel(train, test, cands):
                import os
                workers = min(len(cands), os.cpu_count() or 4)
                best_pnl = float("-inf")
                best_params = cands[0]
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    futures = [pool.submit(_run_single_backtest, train, p) for p in cands]
                    for f in futures:
                        params, pnl = f.result()
                        if pnl > best_pnl:
                            best_pnl = pnl
                            best_params = params
                # Test the winner on out-of-sample data
                test_engine = BacktestEngine(test, strategy_params=best_params, slippage_pct=0.001)
                test_result = test_engine.run()
                return best_params, test_result

            best_params, test_result = await loop.run_in_executor(
                None, _run_window_parallel, train_candles, test_candles, candidates
            )

            logger.info("Walk-forward: window %d/%d done — sharpe=%.2f, pnl=€%.2f, params=%s",
                        i + 1, len(windows_spec), test_result.sharpe_ratio, test_result.total_pnl, best_params)

            wf_windows.append(WFWindow(
                sharpe=test_result.sharpe_ratio,
                pnl=test_result.total_pnl,
                params=best_params,
            ))

        adopted = self._should_adopt(wf_windows)
        avg_sharpe = statistics.mean(w.sharpe for w in wf_windows) if wf_windows else 0
        avg_pnl = statistics.mean(w.pnl for w in wf_windows) if wf_windows else 0
        sharpe_std = statistics.stdev(w.sharpe for w in wf_windows) if len(wf_windows) > 1 else 0

        best_params = None
        if adopted and wf_windows:
            best_window = max(wf_windows, key=lambda w: w.pnl)
            best_params = best_window.params

        return WFResult(
            windows=wf_windows,
            avg_oos_sharpe=avg_sharpe,
            avg_oos_pnl=avg_pnl,
            sharpe_stability=sharpe_std,
            recommended_params=best_params,
            adopted=adopted,
        )
