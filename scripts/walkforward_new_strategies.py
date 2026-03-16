"""Walk-forward test for new strategies across key pairs.

Fetches all candle data upfront, then runs walk-forward in-memory.
"""
import asyncio
import statistics
import sys
import os
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.exchange.bitvavo_client import BitvavoClient
from bot.backtest.engine import BacktestEngine
from bot.learning.walk_forward import (
    TRAIN_DAYS, TEST_DAYS, STEP_DAYS, MIN_WINDOWS,
    WFWindow, WFResult, WalkForwardOptimizer,
)

PAIRS = ["BTC-EUR", "ETH-EUR", "SOL-EUR", "XRP-EUR"]
TOTAL_DAYS = 180  # 6 months of history


def fetch_all_candles(pair: str, days: int) -> list:
    """Fetch candle data upfront (synchronous)."""
    client = BitvavoClient(api_key="", api_secret="", paper_trading=True)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    print(f"  Fetching {pair} candles ({days} days)...", flush=True)
    candles = client.get_candles_range(pair, interval="5m", start=start, end=end)
    print(f"  {pair}: {len(candles)} candles loaded", flush=True)
    return candles


def generate_windows(total_days: int):
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


def slice_candles(all_candles, start_day: int, end_day: int, total_days: int):
    """Slice candle list by day offset."""
    if not all_candles:
        return []
    t0 = all_candles[0].timestamp
    t_end = all_candles[-1].timestamp
    total_span = (t_end - t0).total_seconds()
    if total_span <= 0:
        return []

    start_ts = t0 + timedelta(days=start_day)
    end_ts = t0 + timedelta(days=end_day)

    return [c for c in all_candles if start_ts <= c.timestamp <= end_ts]


def run_walkforward_pair(pair: str, all_candles: list) -> WFResult:
    """Run walk-forward optimization for one pair using pre-fetched candles."""
    windows_spec = generate_windows(TOTAL_DAYS)
    print(f"\n  {pair}: {len(windows_spec)} walk-forward windows", flush=True)

    if len(windows_spec) < MIN_WINDOWS:
        print(f"  {pair}: insufficient windows ({len(windows_spec)} < {MIN_WINDOWS})")
        return WFResult()

    wfo = WalkForwardOptimizer()
    wf_windows = []

    for i, ws in enumerate(windows_spec):
        train_candles = slice_candles(all_candles, ws["train_start_day"], ws["train_end_day"], TOTAL_DAYS)
        test_candles = slice_candles(all_candles, ws["test_start_day"], ws["test_end_day"], TOTAL_DAYS)

        if len(train_candles) < 500 or len(test_candles) < 100:
            print(f"    Window {i+1}: skipped (train={len(train_candles)}, test={len(test_candles)})")
            continue

        print(f"    Window {i+1}/{len(windows_spec)}: train={len(train_candles)}, test={len(test_candles)}, "
              f"RL optimization...", end="", flush=True)

        t0 = time.time()
        best_params, test_result = wfo._run_rl_window(train_candles, test_candles)
        elapsed = time.time() - t0

        print(f" sharpe={test_result.sharpe_ratio:.2f}, pnl=EUR{test_result.total_pnl:+.2f} ({elapsed:.0f}s)")

        # Show strategy breakdown for this window
        strat_counts = {}
        for t in test_result.trade_log:
            if t.strategy not in strat_counts:
                strat_counts[t.strategy] = {"n": 0, "pnl": 0.0}
            strat_counts[t.strategy]["n"] += 1
            strat_counts[t.strategy]["pnl"] += t.pnl_eur
        for s, sc in sorted(strat_counts.items()):
            print(f"        {s}: {sc['n']} trades, EUR{sc['pnl']:+.2f}")

        wf_windows.append(WFWindow(
            sharpe=test_result.sharpe_ratio,
            pnl=test_result.total_pnl,
            params=best_params,
        ))

    if not wf_windows:
        return WFResult()

    sharpes = [w.sharpe for w in wf_windows]
    pnls = [w.pnl for w in wf_windows]
    avg_sharpe = statistics.mean(sharpes)
    avg_pnl = statistics.mean(pnls)
    sharpe_std = statistics.stdev(sharpes) if len(sharpes) > 1 else 0

    # Adoption check
    sorted_sharpes = sorted(sharpes)
    n = len(sorted_sharpes)
    median_sharpe = (sorted_sharpes[n // 2] if n % 2 == 1
                     else (sorted_sharpes[n // 2 - 1] + sorted_sharpes[n // 2]) / 2)
    profitable = sum(1 for p in pnls if p > 0)
    adopted = (
        len(wf_windows) >= MIN_WINDOWS
        and median_sharpe > 0
        and profitable > len(wf_windows) / 2
        and avg_pnl > 0
    )

    best_params = None
    if adopted:
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


def main():
    print("=" * 80)
    print(f"WALK-FORWARD: All strategies incl. funding_contrarian + orderflow")
    print(f"  {TOTAL_DAYS} days history, {TRAIN_DAYS}d train / {TEST_DAYS}d test / {STEP_DAYS}d step")
    print(f"  RL-based optimization per window")
    print("=" * 80, flush=True)

    # Fetch all data upfront
    all_data = {}
    for pair in PAIRS:
        all_data[pair] = fetch_all_candles(pair, TOTAL_DAYS)
        time.sleep(1)  # rate limit courtesy

    # Run walk-forward per pair
    for pair in PAIRS:
        candles = all_data[pair]
        if len(candles) < 1000:
            print(f"\n  {pair}: insufficient candles ({len(candles)}), skipping")
            continue

        result = run_walkforward_pair(pair, candles)

        print(f"\n  {'='*60}")
        print(f"  {pair} SUMMARY:")
        print(f"    Windows: {len(result.windows)}")
        print(f"    Avg OOS Sharpe: {result.avg_oos_sharpe:.2f}")
        print(f"    Avg OOS P&L: EUR{result.avg_oos_pnl:+.2f}")
        print(f"    Sharpe Stability (std): {result.sharpe_stability:.2f}")
        print(f"    Adopted: {result.adopted}")
        if result.recommended_params:
            print(f"    Recommended Params: {result.recommended_params}")
        print(f"  {'='*60}")

    print(f"\n{'='*80}")
    print("Walk-forward complete.")


if __name__ == "__main__":
    main()
