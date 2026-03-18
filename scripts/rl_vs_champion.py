"""
RL Models vs Champion Defaults: 5-year backtest on BTC-EUR, ETH-EUR, XRP-EUR.
Runs each of the 4 trained RL models per-strategy and compares against champion defaults.
Uses GPU kernel for batch backtests when available, falls back to CPU.
"""
import sys, os, time, statistics
# Force unbuffered output so results appear in real-time
sys.stdout.reconfigure(line_buffering=True)
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from bot.exchange.bitvavo_client import BitvavoClient
from bot.backtest.engine import BacktestEngine
from bot.learning.rl_optimizer import RLOptimizer
from bot.learning.feature_extractor import extract_features
from bot.strategy.adopted_universe import CHAMPION_DEFAULTS

PAIRS = ["BTC-EUR", "ETH-EUR", "XRP-EUR"]
STRATEGIES = ["orderflow", "range", "squeeze", "funding_contrarian"]
YEARS = 5
DAYS = YEARS * 365
CAPITAL = 10_000.0


def _gpu_available() -> bool:
    try:
        import cupy as cp
        cp.cuda.Device(0).compute_capability
        return True
    except Exception:
        return False


def candles_to_df(candles) -> pd.DataFrame:
    """Convert CandleData list to a 5m OHLCV DataFrame."""
    df = pd.DataFrame([{
        "timestamp": c.timestamp, "open": c.open, "high": c.high,
        "low": c.low, "close": c.close, "volume": c.volume,
    } for c in candles]).set_index("timestamp").sort_index()
    return df


def run_backtest(candles, params, target_strategy=None):
    engine = BacktestEngine(
        candles, initial_capital=CAPITAL, max_open_positions=10,
        strategy_params=params, target_strategy=target_strategy,
    )
    return engine.run()


def print_result_table(results, title):
    print(f"\n{'=' * 130}")
    print(f"  {title}")
    print(f"{'=' * 130}")
    print(
        f"  {'Config':30s}  {'Pair':10s}  {'Strategy':20s}  {'Trades':>6}  {'P&L':>12}"
        f"  {'Sharpe':>8}  {'WR%':>6}  {'PF':>6}  {'MaxDD%':>7}  {'ROI%':>8}"
    )
    print("  " + "-" * 127)
    for r in results:
        print(
            f"  {r['name']:30s}  {r['pair']:10s}  {r['strategy']:20s}  {r['trades']:6d}  {r['pnl']:>+12.2f}"
            f"  {r['sharpe']:>+8.2f}  {r['wr']:>5.1f}%  {r['pf']:>6.2f}  {r['dd']:>6.1f}%  {r['roi']:>+7.2f}%"
        )


def run_gpu_batch(gpu_data, kernel, batch_configs, rl_opt, dfs_by_pair):
    """Run a batch of per-strategy backtests on GPU in a single kernel launch."""
    from bot.learning.gpu_backtest_kernel import (
        STRATEGY_NAME_TO_ID, build_batch_inputs, launch_backtest,
    )

    # Build symbol index mapping
    symbol_list = list(dfs_by_pair.keys())
    symbol_to_idx = {s: i for i, s in enumerate(symbol_list)}

    param_dicts = []
    strategy_ids = []
    symbol_indices = []

    for cfg in batch_configs:
        param_dicts.append(cfg["params"])
        strategy_ids.append(STRATEGY_NAME_TO_ID[cfg["strategy"]])
        symbol_indices.append(symbol_to_idx[cfg["pair"]])

    params_gpu, seg_info_gpu, batch_size = build_batch_inputs(
        gpu_data, param_dicts, strategy_ids, symbol_indices,
    )
    gpu_results = launch_backtest(kernel, gpu_data, params_gpu, seg_info_gpu, batch_size)

    results = []
    for i, cfg in enumerate(batch_configs):
        pnl = float(gpu_results["pnl"][i])
        trades = int(gpu_results["trades"][i])
        wins = int(gpu_results["wins"][i])
        wr = (wins / trades * 100.0) if trades > 0 else 0.0
        sum_win = float(gpu_results["sum_win_pnl"][i])
        sum_loss = abs(float(gpu_results["sum_loss_pnl"][i]))
        pf = (sum_win / sum_loss) if sum_loss > 0 else (99.0 if sum_win > 0 else 0.0)
        dd = float(gpu_results["max_dd"][i])
        sharpe = float(gpu_results["sharpe"][i])
        roi = pnl / CAPITAL * 100

        results.append({
            "name": cfg["name"],
            "pair": cfg["pair"],
            "strategy": cfg["strategy"],
            "trades": trades,
            "pnl": pnl,
            "sharpe": sharpe,
            "wr": wr,
            "pf": pf,
            "dd": dd,
            "roi": roi,
        })

    return results


def main():
    print("=" * 130)
    print(f"  RL MODELS vs CHAMPION DEFAULTS: {YEARS}-YEAR BACKTEST")
    print(f"  Pairs: {', '.join(PAIRS)} | Capital: EUR {CAPITAL:,.0f}/pair | 5m candles")
    print("=" * 130)

    use_gpu = _gpu_available()
    print(f"  GPU acceleration: {'ENABLED' if use_gpu else 'DISABLED (CPU fallback)'}")

    # Load RL models
    rl_opt = RLOptimizer()
    rl_opt.load_models()
    loaded = [s for s in STRATEGIES if rl_opt.has_model(s)]
    print(f"  RL models loaded: {loaded}")

    # Fetch candles
    candles_by_pair = {}
    dfs_by_pair = {}
    for pair in PAIRS:
        client = BitvavoClient(api_key="", api_secret="", paper_trading=True)
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=DAYS)
        print(f"  Fetching {pair}...", end="", flush=True)
        candles = client.get_candles_range(pair, interval="5m", start=start, end=end)
        if candles and len(candles) > 1000:
            days_actual = (candles[-1].timestamp - candles[0].timestamp).days
            print(f" {len(candles):,} candles ({days_actual} days)", flush=True)
            candles_by_pair[pair] = candles
            dfs_by_pair[pair] = candles_to_df(candles)
        else:
            print(f" skipped", flush=True)
        time.sleep(1)

    results = []

    # -- Prepare GPU data (once) --
    gpu_data = None
    kernel = None
    if use_gpu:
        from bot.learning.gpu_backtest_kernel import (
            compile_kernel, prepare_gpu_data_from_dfs,
        )
        print(f"\n  Preparing GPU data...", end="", flush=True)
        t0 = time.time()
        kernel = compile_kernel()
        gpu_data = prepare_gpu_data_from_dfs(dfs_by_pair)
        print(f" done ({time.time() - t0:.1f}s)", flush=True)

    # -- Champion defaults (all strategies, combined) -- always CPU --
    print(f"\n  Running Champion Defaults (all strategies combined)...")
    for pair, candles in candles_by_pair.items():
        t0 = time.time()
        bt = run_backtest(candles, dict(CHAMPION_DEFAULTS))
        elapsed = time.time() - t0
        pnl = bt.total_pnl
        roi = pnl / CAPITAL * 100
        print(f"    {pair:12s}  {bt.total_trades:>5d} trades  EUR{pnl:>+10.2f}  Sharpe={bt.sharpe_ratio:>+7.2f}  ({elapsed:.0f}s)")
        results.append({
            "name": "Champion Defaults",
            "pair": pair,
            "strategy": "all (combined)",
            "trades": bt.total_trades,
            "pnl": pnl,
            "sharpe": bt.sharpe_ratio,
            "wr": bt.win_rate,
            "pf": bt.profit_factor,
            "dd": bt.max_drawdown_pct,
            "roi": roi,
        })

    # -- Per-strategy backtests: champion + RL static (GPU batched or CPU) --
    if use_gpu and gpu_data is not None:
        # Build batch configs for all per-strategy runs
        batch_configs = []

        # Champion per-strategy
        for strat in STRATEGIES:
            for pair in candles_by_pair:
                batch_configs.append({
                    "name": f"Champion [{strat}]",
                    "pair": pair,
                    "strategy": strat,
                    "params": dict(CHAMPION_DEFAULTS),
                })

        # RL static per-strategy
        for strat in STRATEGIES:
            if not rl_opt.has_model(strat):
                continue
            for pair in candles_by_pair:
                try:
                    recent_df = dfs_by_pair[pair].iloc[-500:]
                    features = extract_features(recent_df, None)
                except Exception:
                    features = np.zeros(27, dtype=np.float32)
                params = rl_opt.predict(features, strat)
                batch_configs.append({
                    "name": f"RL Model [{strat}]",
                    "pair": pair,
                    "strategy": strat,
                    "params": params,
                })

        print(f"\n  Running {len(batch_configs)} per-strategy backtests on GPU...", end="", flush=True)
        t0 = time.time()
        gpu_results = run_gpu_batch(gpu_data, kernel, batch_configs, rl_opt, dfs_by_pair)
        elapsed = time.time() - t0
        print(f" done ({elapsed:.1f}s)", flush=True)

        # Print individual results
        print(f"\n  Champion Defaults (per strategy, isolated):")
        for r in gpu_results:
            if r["name"].startswith("Champion ["):
                print(f"    {r['strategy']:20s} {r['pair']:12s}  {r['trades']:>5d} trades  EUR{r['pnl']:>+10.2f}  Sharpe={r['sharpe']:>+7.2f}")
        print(f"\n  RL Models (per strategy, isolated):")
        for r in gpu_results:
            if r["name"].startswith("RL Model ["):
                print(f"    {r['strategy']:20s} {r['pair']:12s}  {r['trades']:>5d} trades  EUR{r['pnl']:>+10.2f}  Sharpe={r['sharpe']:>+7.2f}")

        results.extend(gpu_results)

    else:
        # CPU fallback: sequential backtests
        print(f"\n  Running Champion Defaults (per strategy, isolated)...")
        for strat in STRATEGIES:
            for pair, candles in candles_by_pair.items():
                t0 = time.time()
                bt = run_backtest(candles, dict(CHAMPION_DEFAULTS), target_strategy=strat)
                elapsed = time.time() - t0
                pnl = bt.total_pnl
                roi = pnl / CAPITAL * 100
                print(f"    {strat:20s} {pair:12s}  {bt.total_trades:>5d} trades  EUR{pnl:>+10.2f}  ({elapsed:.0f}s)")
                results.append({
                    "name": f"Champion [{strat}]",
                    "pair": pair,
                    "strategy": strat,
                    "trades": bt.total_trades,
                    "pnl": pnl,
                    "sharpe": bt.sharpe_ratio,
                    "wr": bt.win_rate,
                    "pf": bt.profit_factor,
                    "dd": bt.max_drawdown_pct,
                    "roi": roi,
                })

        print(f"\n  Running RL Models (per strategy, isolated)...")
        for strat in STRATEGIES:
            if not rl_opt.has_model(strat):
                print(f"    {strat}: NO MODEL — skipping")
                continue

            for pair, candles in candles_by_pair.items():
                try:
                    recent_df = dfs_by_pair[pair].iloc[-500:]
                    features = extract_features(recent_df, None)
                except Exception:
                    features = np.zeros(27, dtype=np.float32)

                params = rl_opt.predict(features, strat)
                if pair == list(candles_by_pair.keys())[0]:
                    print(f"    {strat} RL params ({pair}): atr_mult={params.get('atr_multiplier', '?'):.2f}, "
                          f"rr={params.get('rr_ratio', '?'):.2f}, "
                          f"sig_min={params.get('signal_strength_min', '?'):.2f}, "
                          f"min_pf={params.get('min_profit_multiple', '?'):.2f}, "
                          f"consec={params.get('consecutive_confirms', '?')}")
                t0 = time.time()
                bt = run_backtest(candles, params, target_strategy=strat)
                elapsed = time.time() - t0
                pnl = bt.total_pnl
                roi = pnl / CAPITAL * 100
                print(f"    {strat:20s} {pair:12s}  {bt.total_trades:>5d} trades  EUR{pnl:>+10.2f}  Sharpe={bt.sharpe_ratio:>+7.2f}  ({elapsed:.0f}s)")
                results.append({
                    "name": f"RL Model [{strat}]",
                    "pair": pair,
                    "strategy": strat,
                    "trades": bt.total_trades,
                    "pnl": pnl,
                    "sharpe": bt.sharpe_ratio,
                    "wr": bt.win_rate,
                    "pf": bt.profit_factor,
                    "dd": bt.max_drawdown_pct,
                    "roi": roi,
                })

    # -- Live Simulation per strategy --
    # Replicates live trading: walk-forward validation + adaptive RL predict.
    # Uses daily (288 bars) validation interval for 5yr backtest speed.
    # (Live bot uses 48 = 4h, but 5yr × 4 strategies × 3 pairs needs ~1800 validations each)
    LIVE_SIM_INTERVAL = 288  # daily in 5m bars (1 day = 288 × 5m)
    print(f"\n  Running Live Simulation (WF validation + adaptive predict every {LIVE_SIM_INTERVAL * 5 // 60}h)...")
    for strat in STRATEGIES:
        if not rl_opt.has_model(strat):
            print(f"    {strat}: NO MODEL — skipping")
            continue

        for pair, candles in candles_by_pair.items():
            params = dict(CHAMPION_DEFAULTS)
            t0 = time.time()
            engine = BacktestEngine(
                candles, initial_capital=CAPITAL, max_open_positions=10,
                strategy_params=params, target_strategy=strat,
                rl_optimizer=rl_opt, rl_strategy=strat,
                adapt_interval=LIVE_SIM_INTERVAL,
                live_sim=True,
            )
            bt = engine.run()
            elapsed = time.time() - t0
            pnl = bt.total_pnl
            roi = pnl / CAPITAL * 100
            print(f"    {strat:20s} {pair:12s}  {bt.total_trades:>5d} trades  EUR{pnl:>+10.2f}  Sharpe={bt.sharpe_ratio:>+7.2f}  ({elapsed:.0f}s)")
            results.append({
                "name": f"Live Sim [{strat}]",
                "pair": pair,
                "strategy": strat,
                "trades": bt.total_trades,
                "pnl": pnl,
                "sharpe": bt.sharpe_ratio,
                "wr": bt.win_rate,
                "pf": bt.profit_factor,
                "dd": bt.max_drawdown_pct,
                "roi": roi,
            })

    # -- Per-Signal RL with Online Learning --
    # New architecture: RL evaluates each signal individually and learns from every trade.
    # Runs all strategies combined (no target_strategy) so the model sees all signals.
    from bot.learning.rl_signal_evaluator import RLSignalEvaluator
    print(f"\n  Running Per-Signal RL with Online Learning (all strategies combined)...")
    for pair, candles in candles_by_pair.items():
        evaluator = RLSignalEvaluator()
        t0 = time.time()
        engine = BacktestEngine(
            candles, initial_capital=CAPITAL, max_open_positions=10,
            strategy_params=dict(CHAMPION_DEFAULTS),
            signal_evaluator=evaluator,
            online_learning=True,
        )
        bt = engine.run()
        elapsed = time.time() - t0
        pnl = bt.total_pnl
        roi = pnl / CAPITAL * 100
        stats = evaluator.stats
        print(
            f"    {pair:12s}  {bt.total_trades:>5d} trades  EUR{pnl:>+10.2f}  Sharpe={bt.sharpe_ratio:>+7.2f}"
            f"  PPO updates={stats['updates']}, avg_rew={stats['avg_reward_last_100']:.4f}  ({elapsed:.0f}s)"
        )
        results.append({
            "name": "RL Online (combined)",
            "pair": pair,
            "strategy": "all (combined)",
            "trades": bt.total_trades,
            "pnl": pnl,
            "sharpe": bt.sharpe_ratio,
            "wr": bt.win_rate,
            "pf": bt.profit_factor,
            "dd": bt.max_drawdown_pct,
            "roi": roi,
        })

    # Also test per-strategy RL online learning
    print(f"\n  Running Per-Signal RL with Online Learning (per strategy)...")
    for strat in STRATEGIES:
        evaluator = RLSignalEvaluator()
        for pair, candles in candles_by_pair.items():
            t0 = time.time()
            engine = BacktestEngine(
                candles, initial_capital=CAPITAL, max_open_positions=10,
                strategy_params=dict(CHAMPION_DEFAULTS),
                target_strategy=strat,
                signal_evaluator=evaluator,
                online_learning=True,
            )
            bt = engine.run()
            elapsed = time.time() - t0
            pnl = bt.total_pnl
            roi = pnl / CAPITAL * 100
            stats = evaluator.stats
            print(
                f"    {strat:20s} {pair:12s}  {bt.total_trades:>5d} trades  EUR{pnl:>+10.2f}"
                f"  Sharpe={bt.sharpe_ratio:>+7.2f}  updates={stats['updates']}  ({elapsed:.0f}s)"
            )
            results.append({
                "name": f"RL Online [{strat}]",
                "pair": pair,
                "strategy": strat,
                "trades": bt.total_trades,
                "pnl": pnl,
                "sharpe": bt.sharpe_ratio,
                "wr": bt.win_rate,
                "pf": bt.profit_factor,
                "dd": bt.max_drawdown_pct,
                "roi": roi,
            })

    # --- ML Signal Generator test ---
    ml_gen = None
    model_dir = "models/ml_signals"
    if Path(model_dir).exists() and (Path(model_dir) / "lstm.pt").exists():
        from bot.learning.ml_signal_generator import MLSignalGenerator
        ml_gen = MLSignalGenerator(model_dir=model_dir)

        print(f"\n{'='*100}")
        print(f"  ML SIGNAL GENERATOR TEST")
        print(f"{'='*100}")

        for pair in PAIRS:
            if pair not in candles_by_pair:
                print(f"  {pair} skipped (no candle data)")
                continue
            candles = candles_by_pair[pair]

            # ML without RL evaluator
            engine_ml = BacktestEngine(
                candles, initial_capital=CAPITAL,
                ml_signal_generator=ml_gen,
            )
            bt_ml = engine_ml.run()
            print(f"  {pair} ML Only:    {bt_ml.total_trades} trades, "
                  f"EUR {bt_ml.total_pnl:+.2f}, Sharpe={bt_ml.sharpe_ratio:+.2f}")
            results.append({
                "name": "ML Only",
                "pair": pair,
                "strategy": "ml_signal",
                "trades": bt_ml.total_trades,
                "pnl": bt_ml.total_pnl,
                "sharpe": bt_ml.sharpe_ratio,
                "wr": bt_ml.win_rate,
                "pf": bt_ml.profit_factor,
                "dd": bt_ml.max_drawdown_pct,
                "roi": bt_ml.total_pnl / CAPITAL * 100,
            })

            # ML with RL evaluator + online learning
            evaluator = RLSignalEvaluator()
            engine_ml_rl = BacktestEngine(
                candles, initial_capital=CAPITAL,
                ml_signal_generator=ml_gen,
                signal_evaluator=evaluator,
                online_learning=True,
            )
            bt_ml_rl = engine_ml_rl.run()
            print(f"  {pair} ML + RL:    {bt_ml_rl.total_trades} trades, "
                  f"EUR {bt_ml_rl.total_pnl:+.2f}, Sharpe={bt_ml_rl.sharpe_ratio:+.2f}")
            results.append({
                "name": "ML + RL Online",
                "pair": pair,
                "strategy": "ml_signal",
                "trades": bt_ml_rl.total_trades,
                "pnl": bt_ml_rl.total_pnl,
                "sharpe": bt_ml_rl.sharpe_ratio,
                "wr": bt_ml_rl.win_rate,
                "pf": bt_ml_rl.profit_factor,
                "dd": bt_ml_rl.max_drawdown_pct,
                "roi": bt_ml_rl.total_pnl / CAPITAL * 100,
            })
    else:
        print(f"\n  ML models not found at {model_dir} — skipping ML test")

    # -- Summary tables --
    print_result_table(results, "ALL RESULTS")

    # -- Aggregate comparison --
    print(f"\n\n{'=' * 130}")
    print(f"  AGGREGATE COMPARISON (sum across all 3 pairs)")
    print(f"{'=' * 130}")

    # Group by name
    groups = {}
    for r in results:
        key = r["name"]
        if key not in groups:
            groups[key] = {"pnl": 0, "trades": 0, "sharpes": [], "wrs": [], "pfs": [], "dds": []}
        groups[key]["pnl"] += r["pnl"]
        groups[key]["trades"] += r["trades"]
        groups[key]["sharpes"].append(r["sharpe"])
        groups[key]["wrs"].append(r["wr"])
        groups[key]["pfs"].append(r["pf"])
        groups[key]["dds"].append(r["dd"])

    print(
        f"  {'Config':30s}  {'Trades':>6}  {'Total P&L':>12}  {'Avg Sharpe':>10}"
        f"  {'Avg WR%':>8}  {'Avg PF':>7}  {'Max DD%':>8}  {'ROI%':>8}"
    )
    print("  " + "-" * 100)

    for name, g in sorted(groups.items(), key=lambda x: x[1]["pnl"], reverse=True):
        avg_sharpe = statistics.mean(g["sharpes"]) if g["sharpes"] else 0
        avg_wr = statistics.mean(g["wrs"]) if g["wrs"] else 0
        avg_pf = statistics.mean(g["pfs"]) if g["pfs"] else 0
        max_dd = max(g["dds"]) if g["dds"] else 0
        roi = g["pnl"] / (CAPITAL * len(PAIRS)) * 100
        print(
            f"  {name:30s}  {g['trades']:6d}  {g['pnl']:>+12.2f}  {avg_sharpe:>+10.2f}"
            f"  {avg_wr:>7.1f}%  {avg_pf:>7.2f}  {max_dd:>7.1f}%  {roi:>+7.2f}%"
        )


if __name__ == "__main__":
    main()
