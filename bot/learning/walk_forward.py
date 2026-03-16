"""Walk-forward optimization — rolling out-of-sample parameter validation."""
from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from bot.backtest.engine import BacktestResult
from bot.strategy.adopted_universe import CHAMPION_DEFAULTS

logger = logging.getLogger(__name__)

TRAIN_DAYS = 90
TEST_DAYS = 14
STEP_DAYS = 14
MIN_WINDOWS = 4


def _run_single_backtest(candles, params, target_strategy=None):
    """Top-level function so ProcessPoolExecutor can pickle it."""
    import time
    t0 = time.perf_counter()
    from bot.backtest.engine import BacktestEngine
    engine = BacktestEngine(candles, strategy_params=params, slippage_pct=0.001, target_strategy=target_strategy)
    result = engine.run()
    elapsed = time.perf_counter() - t0
    pf = result.profit_factor if hasattr(result, 'profit_factor') else 1.0
    ppf = result.profit_per_fee if hasattr(result, 'profit_per_fee') else 0.0
    logging.getLogger(__name__).info(
        "Backtest done: %d candles, pnl=€%.2f, sharpe=%.2f, pf=%.2f, ppf=%.2f, %.1fs",
        len(candles), result.total_pnl, result.sharpe_ratio, pf, ppf, elapsed
    )
    return params, result.total_pnl, result.sharpe_ratio, pf, ppf



@dataclass
class WFWindow:
    sharpe: float = 0.0
    pnl: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)
    profit_per_fee: float = 0.0
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

    def __init__(self, rl_optimizer=None, candle_store=None) -> None:
        self._rl_optimizer = rl_optimizer
        self._candle_store = candle_store
        self._latest_result: Optional[WFResult] = None
        self._results_by_symbol: Dict[str, WFResult] = {}
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def latest_result(self) -> Optional[WFResult]:
        return self._latest_result

    def result_for_symbol(self, symbol: str) -> Optional[WFResult]:
        """Backward-compat: return first result matching this symbol."""
        # Check composite keys first (new format)
        for key, val in self._results_by_symbol.items():
            if key.startswith(f"{symbol}:"):
                return val
        # Fall back to direct key (old format)
        return self._results_by_symbol.get(symbol)

    def results_for_symbol(self, symbol: str) -> Dict[str, WFResult]:
        """Return all per-strategy results for a symbol."""
        results = {}
        for key, val in self._results_by_symbol.items():
            if key.startswith(f"{symbol}:"):
                strategy = key.split(":", 1)[1]
                results[strategy] = val
        return results

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
        ppfs = [w.profit_per_fee for w in windows]

        median_sharpe = statistics.median(sharpes)
        profitable = sum(1 for p in pnls if p > 0)
        sharpe_std = statistics.stdev(sharpes) if len(sharpes) > 1 else 999
        avg_ppf = statistics.mean(ppfs) if ppfs else 0.0

        return (
            median_sharpe > 0.3
            and profitable >= len(windows) * 0.70
            and statistics.mean(pnls) > 0
            and sharpe_std < 1.5
            and avg_ppf > 1.5
        )

    def _run_rl_window(self, train_candles_5m, test_candles_5m,
                        target_strategy=None):
        """RL-driven parameter optimization for a single window."""
        from bot.learning.feature_extractor import extract_features

        if self._rl_optimizer is None or not self._rl_optimizer.has_model(target_strategy or "orderflow"):
            return dict(CHAMPION_DEFAULTS), BacktestResult()

        # Get 1m candles for feature extraction from CandleStore
        features = np.zeros(27, dtype=np.float32)
        if self._candle_store is not None and train_candles_5m:
            try:
                # Determine date range from 5m candles
                if isinstance(train_candles_5m[0], dict):
                    first_ts = train_candles_5m[0].get("timestamp", "")
                    last_ts = train_candles_5m[-1].get("timestamp", "")
                else:
                    first_ts = train_candles_5m[0].timestamp
                    last_ts = train_candles_5m[-1].timestamp

                start = datetime.fromisoformat(str(first_ts).replace("Z", "+00:00"))
                end = datetime.fromisoformat(str(last_ts).replace("Z", "+00:00"))

                symbol = (train_candles_5m[0].get("symbol", "BTC-EUR")
                          if isinstance(train_candles_5m[0], dict)
                          else getattr(train_candles_5m[0], "symbol", "BTC-EUR"))

                candles_1m = self._candle_store.get_candles(symbol, start, end, resample="1m")
                btc_1m = None
                if symbol != "BTC-EUR":
                    btc_1m = self._candle_store.get_candles("BTC-EUR", start, end, resample="1m")
                features = extract_features(candles_1m, btc_1m)
            except Exception as e:
                logger.warning("Feature extraction failed in walk-forward: %s", e)

        # Predict parameters
        params = self._rl_optimizer.predict(features, target_strategy or "orderflow")

        # Run backtest on test candles (OOS validation)
        try:
            from bot.backtest.engine import BacktestEngine
            test_engine = BacktestEngine(
                test_candles_5m,
                strategy_params=params,
                target_strategy=target_strategy,
            )
            test_result = test_engine.run()
        except Exception:
            test_result = BacktestResult()

        return params, test_result

    async def run(self, candle_fetcher=None, symbol: str = "BTC-EUR") -> WFResult:
        """Execute full walk-forward optimization for a single symbol."""
        self._running = True
        try:
            result = await self._execute(candle_fetcher, symbol)
            self._latest_result = result
            self._results_by_symbol[symbol] = result
            return result
        finally:
            self._running = False

    async def run_multi(self, candle_fetcher_factory=None, symbols: List[str] = None) -> Dict[str, WFResult]:
        """Run walk-forward optimization for multiple symbols sequentially."""
        if not symbols:
            symbols = ["BTC-EUR"]
        self._running = True
        results: Dict[str, WFResult] = {}
        try:
            for symbol in symbols:
                logger.info("Walk-forward: starting optimization for %s", symbol)
                fetcher = candle_fetcher_factory(symbol) if candle_fetcher_factory else None
                result = await self._execute(fetcher, symbol)
                results[symbol] = result
                self._results_by_symbol[symbol] = result
                self._latest_result = result
            return results
        finally:
            self._running = False

    async def run_multi_per_strategy(
        self, candle_fetcher_factory=None, symbols: List[str] = None, strategies: List[str] = None,
    ) -> Dict[str, Dict[str, WFResult]]:
        """Run walk-forward for each strategy-symbol combo.

        Fetches candles once per symbol, then runs optimization for each strategy.
        Returns: {symbol: {strategy_name: WFResult}}
        """
        from bot.strategy.adopted_universe import ALL_STRATEGIES

        if not symbols:
            symbols = ["BTC-EUR"]
        if not strategies:
            strategies = ALL_STRATEGIES

        self._running = True
        results: Dict[str, Dict[str, WFResult]] = {}

        try:
            for symbol in symbols:
                logger.info("Walk-forward: fetching full candle range for %s", symbol)

                total_days = TRAIN_DAYS + TEST_DAYS * MIN_WINDOWS
                if candle_fetcher_factory:
                    fetcher = candle_fetcher_factory(symbol)
                    all_candles = await fetcher(0, total_days)
                else:
                    all_candles = []

                if not all_candles:
                    logger.warning("Walk-forward: no candles for %s — skipping", symbol)
                    continue

                logger.info(
                    "Walk-forward: got %d candles for %s — running %d strategies",
                    len(all_candles), symbol, len(strategies),
                )

                results[symbol] = {}
                for strategy in strategies:
                    result = await self._execute_from_candles(
                        all_candles, symbol, target_strategy=strategy,
                    )
                    results[symbol][strategy] = result
                    self._results_by_symbol[f"{symbol}:{strategy}"] = result
                    self._latest_result = result

            return results
        finally:
            self._running = False

    async def _execute(self, candle_fetcher, symbol: str = "BTC-EUR") -> WFResult:
        import asyncio

        total_days = TRAIN_DAYS + TEST_DAYS * MIN_WINDOWS
        windows_spec = self._generate_windows(total_days)

        if len(windows_spec) < MIN_WINDOWS:
            logger.warning("Walk-forward: insufficient data for %d windows", MIN_WINDOWS)
            return WFResult()

        logger.info("Walk-forward [%s]: RL optimization, %d windows",
                     symbol, len(windows_spec))

        loop = asyncio.get_running_loop()
        wf_windows: List[WFWindow] = []

        for i, ws in enumerate(windows_spec):
            if candle_fetcher is None:
                break

            train_candles = await candle_fetcher(ws["train_start_day"], ws["train_end_day"])
            test_candles = await candle_fetcher(ws["test_start_day"], ws["test_end_day"])

            if not train_candles or not test_candles:
                continue

            logger.info("Walk-forward [%s]: window %d/%d (%d train, %d test candles)...",
                        symbol, i + 1, len(windows_spec), len(train_candles), len(test_candles))

            best_params, test_result = await loop.run_in_executor(
                None, self._run_rl_window, train_candles, test_candles, None
            )

            logger.info("Walk-forward [%s]: window %d/%d done — sharpe=%.2f, pnl=€%.2f, params=%s",
                        symbol, i + 1, len(windows_spec), test_result.sharpe_ratio, test_result.total_pnl, best_params)

            wf_windows.append(WFWindow(
                sharpe=test_result.sharpe_ratio,
                pnl=test_result.total_pnl,
                params=best_params,
                profit_per_fee=getattr(test_result, 'profit_per_fee', 0.0),
            ))

        adopted = self._should_adopt(wf_windows)
        avg_sharpe = statistics.mean(w.sharpe for w in wf_windows) if wf_windows else 0
        avg_pnl = statistics.mean(w.pnl for w in wf_windows) if wf_windows else 0
        sharpe_std = statistics.stdev(w.sharpe for w in wf_windows) if len(wf_windows) > 1 else 0

        best_params = None
        if adopted and wf_windows:
            best_window = max(wf_windows, key=lambda w: w.sharpe)
            best_params = best_window.params

        return WFResult(
            windows=wf_windows,
            avg_oos_sharpe=avg_sharpe,
            avg_oos_pnl=avg_pnl,
            sharpe_stability=sharpe_std,
            recommended_params=best_params,
            adopted=adopted,
        )

    async def _execute_from_candles(
        self, all_candles, symbol: str = "BTC-EUR", target_strategy: str = None,
    ) -> WFResult:
        """Execute walk-forward using pre-fetched candles (no API calls)."""
        import asyncio

        total_days = TRAIN_DAYS + TEST_DAYS * MIN_WINDOWS
        windows_spec = self._generate_windows(total_days)

        if len(windows_spec) < MIN_WINDOWS:
            logger.warning("Walk-forward: insufficient windows for %s", symbol)
            return WFResult()

        candles_per_day = len(all_candles) / total_days if total_days > 0 else 288

        loop = asyncio.get_running_loop()
        wf_windows: List[WFWindow] = []

        for i, ws in enumerate(windows_spec):
            train_start = int(ws["train_start_day"] * candles_per_day)
            train_end = int(ws["train_end_day"] * candles_per_day)
            test_start = int(ws["test_start_day"] * candles_per_day)
            test_end = int(ws["test_end_day"] * candles_per_day)

            train_candles = all_candles[train_start:train_end]
            test_candles = all_candles[test_start:test_end]

            if not train_candles or not test_candles:
                continue

            strat_label = f" [{target_strategy}]" if target_strategy else ""
            logger.info(
                "Walk-forward [%s]%s: window %d/%d (%d train, %d test candles)...",
                symbol, strat_label, i + 1, len(windows_spec),
                len(train_candles), len(test_candles),
            )

            best_params, test_result = await loop.run_in_executor(
                None, self._run_rl_window, train_candles, test_candles, target_strategy,
            )

            logger.info(
                "Walk-forward [%s]%s: window %d/%d done — sharpe=%.2f, pnl=€%.2f",
                symbol, strat_label, i + 1, len(windows_spec),
                test_result.sharpe_ratio, test_result.total_pnl,
            )

            wf_windows.append(WFWindow(
                sharpe=test_result.sharpe_ratio,
                pnl=test_result.total_pnl,
                params=best_params,
                profit_per_fee=getattr(test_result, "profit_per_fee", 0.0),
            ))

        adopted = self._should_adopt(wf_windows)
        avg_sharpe = statistics.mean(w.sharpe for w in wf_windows) if wf_windows else 0
        avg_pnl = statistics.mean(w.pnl for w in wf_windows) if wf_windows else 0
        sharpe_std = statistics.stdev(w.sharpe for w in wf_windows) if len(wf_windows) > 1 else 0

        best_params = None
        if adopted and wf_windows:
            best_window = max(wf_windows, key=lambda w: w.sharpe)
            best_params = best_window.params

        return WFResult(
            windows=wf_windows,
            avg_oos_sharpe=avg_sharpe,
            avg_oos_pnl=avg_pnl,
            sharpe_stability=sharpe_std,
            recommended_params=best_params,
            adopted=adopted,
        )
