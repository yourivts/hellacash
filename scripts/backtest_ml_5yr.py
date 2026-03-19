"""5-year backtest of the full system with ML signal generator.

Loads trained Transformer + XGBoost models and runs a realistic backtest
across all pairs, reporting per-pair and aggregate P&L, Sharpe, drawdown,
and yearly breakdown.

Batch ML inference: features are extracted once per pair (vectorized), then
Transformer + XGBoost run in batch. A lookup wrapper provides instant signal
retrieval during the backtest engine loop.
"""
import sys, os, time, statistics, gc
from datetime import datetime, timedelta, timezone
from collections import defaultdict

import numpy as np
import torch
import xgboost as xgb

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from bot.data.candle_store import CandleStore
from bot.data.external_features import ExternalDataProvider
from bot.backtest.engine import BacktestEngine
from bot.learning.ml_signal_generator import MLSignalGenerator, HORIZONS, DIRECTIONS, MIN_SIGNAL_PROB
from bot.learning.ml_features import (
    _batch_extract_tabular,
    build_lstm_sequences_batch,
    MIN_BARS_5M,
)
from bot.config import get_settings

# Use same pairs as training
PAIRS = [
    "BTC-EUR", "ETH-EUR", "XRP-EUR", "SOL-EUR", "ADA-EUR",
    "DOGE-EUR", "LINK-EUR", "AVAX-EUR", "DOT-EUR", "POL-EUR",
    "SHIB-EUR", "UNI-EUR", "LTC-EUR", "ATOM-EUR", "NEAR-EUR",
    "FIL-EUR", "ARB-EUR", "OP-EUR", "APT-EUR", "SUI-EUR",
    "PEPE-EUR", "INJ-EUR", "FET-EUR", "RENDER-EUR", "TIA-EUR",
]
YEARS = 5
DAYS = YEARS * 365
CAPITAL_PER_PAIR = 10_000.0

# Strategy parameters (champion defaults)
PARAMS = {
    "atr_multiplier": 3.0,
    "rr_ratio": 2.0,
    "kelly_fraction": 0.25,
    "min_confirmations": 2,
    "entry_threshold": 0.50,
    "cooldown_hours": 48,
    "max_hold_hours": 72,
}

# BacktestEngine signal cadence
ENGINE_WARMUP = 60
ENGINE_SIGNAL_EVERY = 12


def load_candles_db(candle_store, pair, start, end):
    """Load 5m candles from PostgreSQL (much faster than API)."""
    print(f"  Loading {pair} from DB...", end="", flush=True)
    df = candle_store.get_candles(pair, start, end, resample="5m")
    if df.empty:
        print(" NO DATA", flush=True)
        return None
    actual_days = (df.index[-1] - df.index[0]).days
    print(f" {len(df):,} candles ({actual_days} days)", flush=True)
    return df


class PrecomputedMLSignals:
    """Lookup wrapper — returns pre-computed ML predictions by DataFrame length.

    The backtest engine calls predict(df_window, symbol) where df_window is
    df.iloc[max(0, i-2000):i+1]. We identify the candle index from len(df_window)
    offset by the window start, but the simplest approach is to key on the original
    DataFrame index of the last candle in the window.
    """

    def __init__(self, signal_map):
        """signal_map: dict mapping candle_index -> prediction dict"""
        self._signals = signal_map
        self._disabled = False

    def predict(self, df_5m, symbol="BTC-EUR", **kwargs):
        """Look up pre-computed prediction for the last candle in df_5m."""
        default = {
            "probabilities": [0.5] * 12,
            "direction": None,
            "net_up": 0.5,
            "net_down": 0.5,
        }
        if self._disabled:
            return default

        # The engine passes df.iloc[max(0, i-2000):i+1]
        # The last timestamp in this window identifies the signal point
        last_ts = df_5m.index[-1]
        result = self._signals.get(last_ts)
        if result is not None:
            return result
        return default


def precompute_ml_signals(ml_gen, df, pair, btc_df=None, external_features=None):
    """Batch-compute all ML signals for a pair's full DataFrame.

    Args:
        btc_df: BTC-EUR 5m DataFrame for cross-asset features (features 62-64)
        external_features: (len(df), 32) array of external data (features 55-61, 65-89)

    Returns a dict mapping timestamp -> prediction result.
    """
    n = len(df)
    # Engine evaluates at indices where (i - WARMUP) % SIGNAL_EVERY == 0
    # and i >= MIN_BARS_5M (features need enough history)
    first_signal = max(ENGINE_WARMUP, MIN_BARS_5M)
    # Align to engine cadence
    remainder = (first_signal - ENGINE_WARMUP) % ENGINE_SIGNAL_EVERY
    if remainder != 0:
        first_signal += ENGINE_SIGNAL_EVERY - remainder

    sample_indices = np.arange(first_signal, n, ENGINE_SIGNAL_EVERY)
    if len(sample_indices) == 0:
        return {}

    print(f"    Extracting {len(sample_indices):,} feature vectors...", end="", flush=True)
    t0 = time.time()

    # 1. Batch tabular features (indicators computed once, vectorized indexing)
    # Pass BTC data for cross-asset features and external data
    btc_for_features = btc_df if pair != "BTC-EUR" else None
    tabular = _batch_extract_tabular(df, pair, btc_for_features, sample_indices, external_features)

    # 2. Batch LSTM/Transformer sequences
    sequences = build_lstm_sequences_batch(df, sample_indices)
    print(f" {time.time()-t0:.1f}s", flush=True)

    # 3. Batch Transformer embedding
    print(f"    Transformer embedding ({len(sequences)} sequences)...", end="", flush=True)
    t0 = time.time()
    device = next(ml_gen._embedder.parameters()).device
    ml_gen._embedder.eval()
    embeddings = []
    batch_size = 4096
    with torch.no_grad():
        for start_idx in range(0, len(sequences), batch_size):
            batch = torch.from_numpy(sequences[start_idx:start_idx + batch_size]).float().to(device)
            emb = ml_gen._embedder.embed(batch)
            embeddings.append(emb.cpu().numpy())
    embeddings = np.concatenate(embeddings, axis=0)  # (N, 16)
    print(f" {time.time()-t0:.1f}s", flush=True)

    # 4. Concatenate features: 90 tabular + 16 embedding = 106
    combined = np.concatenate([tabular, embeddings], axis=1).astype(np.float32)  # (N, 106)

    # 5. Batch XGBoost predictions (all 12 models)
    print(f"    XGBoost batch inference ({len(combined):,} samples x 12 models)...", end="", flush=True)
    t0 = time.time()
    all_probs = np.zeros((len(combined), 12), dtype=np.float32)
    dmat = xgb.DMatrix(combined)
    for model_idx, (h, d) in enumerate([(h, d) for h in HORIZONS for d in DIRECTIONS]):
        key = f"{h}_{d}"
        model = ml_gen._xgb_models[key]
        booster = model.get_booster()
        raw_preds = booster.predict(dmat)
        all_probs[:, model_idx] = raw_preds
    print(f" {time.time()-t0:.1f}s", flush=True)

    # 6. Build signal map: timestamp -> prediction dict
    signal_map = {}
    for j, idx in enumerate(sample_indices):
        probs = all_probs[j].tolist()
        up_probs = [probs[i] for i in range(0, 12, 2)]
        down_probs = [probs[i] for i in range(1, 12, 2)]
        net_up = float(np.mean(up_probs))
        net_down = float(np.mean(down_probs))
        direction = None
        if max(net_up, net_down) >= MIN_SIGNAL_PROB:
            direction = "LONG" if net_up > net_down else "SHORT"
        ts = df.index[idx]
        signal_map[ts] = {
            "probabilities": probs,
            "direction": direction,
            "net_up": net_up,
            "net_down": net_down,
        }

    return signal_map


def analyze_trades(trade_log):
    """Group trades by strategy."""
    by_strat = defaultdict(lambda: {
        "trades": [], "wins": 0, "losses": 0, "total_pnl": 0.0,
        "win_pnls": [], "loss_pnls": [], "directions": defaultdict(int),
        "exit_reasons": defaultdict(int), "hold_times": [],
    })
    for t in trade_log:
        s = by_strat[t.strategy]
        s["trades"].append(t)
        s["total_pnl"] += t.pnl_eur
        s["directions"][t.direction] += 1
        s["exit_reasons"][t.exit_reason] += 1
        if t.pnl_eur > 0:
            s["wins"] += 1
            s["win_pnls"].append(t.pnl_eur)
        else:
            s["losses"] += 1
            s["loss_pnls"].append(t.pnl_eur)
        try:
            entry_dt = datetime.fromisoformat(t.entry_time)
            exit_dt = datetime.fromisoformat(t.exit_time)
            s["hold_times"].append((exit_dt - entry_dt).total_seconds() / 3600)
        except Exception:
            pass
    return by_strat


def compute_sharpe(trade_pnls, periods_per_year=252):
    if len(trade_pnls) < 2:
        return 0.0
    avg = statistics.mean(trade_pnls)
    std = statistics.stdev(trade_pnls)
    if std == 0:
        return 0.0
    return (avg / std) * (periods_per_year ** 0.5)


def print_strategy_report(name, data, total_capital=CAPITAL_PER_PAIR):
    n = len(data["trades"])
    if n == 0:
        print(f"  {name:25s}: 0 trades")
        return

    pnl = data["total_pnl"]
    wins = data["wins"]
    losses = data["losses"]
    wr = wins / n * 100

    avg_win = statistics.mean(data["win_pnls"]) if data["win_pnls"] else 0
    avg_loss = statistics.mean(data["loss_pnls"]) if data["loss_pnls"] else 0
    max_win = max(data["win_pnls"]) if data["win_pnls"] else 0
    max_loss = min(data["loss_pnls"]) if data["loss_pnls"] else 0

    all_pnls = [t.pnl_eur for t in data["trades"]]
    sharpe = compute_sharpe(all_pnls)
    profit_factor = abs(sum(data["win_pnls"]) / sum(data["loss_pnls"])) if data["loss_pnls"] and sum(data["loss_pnls"]) != 0 else float('inf')

    avg_hold = statistics.mean(data["hold_times"]) if data["hold_times"] else 0
    max_hold = max(data["hold_times"]) if data["hold_times"] else 0
    expectancy = pnl / n
    roi = pnl / total_capital * 100

    print(f"\n  --- {name} ---")
    print(f"    Trades:          {n:>6d}  (W:{wins} / L:{losses})")
    print(f"    Win Rate:        {wr:>6.1f}%")
    print(f"    Total P&L:       EUR{pnl:>+10.2f}  (ROI: {roi:+.2f}%)")
    print(f"    Sharpe:          {sharpe:>+8.2f}")
    print(f"    Profit Factor:   {profit_factor:>8.2f}")
    print(f"    Expectancy:      EUR{expectancy:>+8.2f} / trade")
    print(f"    Avg Win:         EUR{avg_win:>+8.2f}  |  Avg Loss: EUR{avg_loss:>+8.2f}")
    print(f"    Best Trade:      EUR{max_win:>+8.2f}  |  Worst:    EUR{max_loss:>+8.2f}")
    print(f"    Avg Hold:        {avg_hold:>6.1f}h  |  Max Hold: {max_hold:>6.1f}h")
    print(f"    Directions:      LONG={data['directions'].get('LONG', 0)}  SHORT={data['directions'].get('SHORT', 0)}")
    exits = data["exit_reasons"]
    exit_str = ", ".join(f"{k}={v}" for k, v in sorted(exits.items(), key=lambda x: -x[1]))
    print(f"    Exit Reasons:    {exit_str}")


def main():
    print("=" * 90)
    print(f"  ML SIGNAL GENERATOR: {YEARS}-YEAR BACKTEST (BATCH MODE)")
    print(f"  Transformer + XGBoost | Capital: EUR {CAPITAL_PER_PAIR:,.0f}/pair | 5m candles")
    print("=" * 90, flush=True)

    # Load ML signal generator (for model weights)
    print("\n  Loading ML models...", end="", flush=True)
    try:
        ml_gen = MLSignalGenerator(model_dir="models/ml_signals")
        print(" OK", flush=True)
    except FileNotFoundError as e:
        print(f" FAILED: {e}")
        sys.exit(1)

    # Load candles from PostgreSQL
    settings = get_settings()
    candle_store = CandleStore(db_url=settings.database_url)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=DAYS)

    # Load BTC-EUR first (needed for cross-asset features on all other pairs)
    print("\n  Loading BTC-EUR for cross-asset features...", end="", flush=True)
    btc_df = candle_store.get_candles("BTC-EUR", start, end, resample="5m")
    print(f" {len(btc_df):,} candles", flush=True)

    # Load external data (funding rates, OI, on-chain, macro — 32 features)
    print("  Loading external data...", end="", flush=True)
    ext_provider = ExternalDataProvider(cache_dir="data/external_cache", db_url=settings.database_url)
    btc_idx = btc_df.index if btc_df.index.tz is not None else btc_df.index.tz_localize("UTC")
    ext_features_full = ext_provider.build_training_features(btc_idx)
    ext_df = pd.DataFrame(ext_features_full, index=btc_idx)
    print(f" {ext_features_full.shape}", flush=True)

    # Aggregate tracking
    agg_strats = defaultdict(lambda: {
        "trades": [], "wins": 0, "losses": 0, "total_pnl": 0.0,
        "win_pnls": [], "loss_pnls": [], "directions": defaultdict(int),
        "exit_reasons": defaultdict(int), "hold_times": [],
    })
    grand_total_pnl = 0.0
    grand_total_trades = 0
    pair_results = []

    for pair in PAIRS:
        df = load_candles_db(candle_store, pair, start, end)
        if df is None or len(df) < 10000:
            print(f"  {pair}: insufficient data ({len(df) if df is not None else 0}), skipping")
            continue

        actual_days = (df.index[-1] - df.index[0]).days

        # Align external features to this pair's index
        pair_idx = df.index if df.index.tz is not None else df.index.tz_localize("UTC")
        pair_ext = ext_df.reindex(pair_idx, method="ffill").values.astype(np.float32)

        # Pre-compute all ML signals in batch (with BTC + external features)
        t0 = time.time()
        signal_map = precompute_ml_signals(ml_gen, df, pair, btc_df=btc_df, external_features=pair_ext)
        precomp_time = time.time() - t0

        n_signals = sum(1 for v in signal_map.values() if v["direction"] is not None)
        print(f"    Pre-computed {len(signal_map):,} predictions "
              f"({n_signals:,} with direction) in {precomp_time:.1f}s", flush=True)

        # Create lookup wrapper
        lookup = PrecomputedMLSignals(signal_map)

        # Run backtest with instant signal lookup
        t0 = time.time()
        engine = BacktestEngine(
            df=df,
            symbol=pair,
            initial_capital=CAPITAL_PER_PAIR,
            max_open_positions=3,
            strategy_params=PARAMS,
            ml_signal_generator=lookup,
        )
        result = engine.run()
        bt_time = time.time() - t0

        print(f"\n{'='*90}")
        print(f"  {pair} ({actual_days} days, {len(df):,} candles)")
        print(f"  Precompute: {precomp_time:.0f}s | Backtest: {bt_time:.0f}s | Total: {precomp_time+bt_time:.0f}s")
        print(f"  P&L=EUR{result.total_pnl:+,.2f} | Trades={result.total_trades} | "
              f"WR={result.win_rate:.0%} | Sharpe={result.sharpe_ratio:.2f} | "
              f"MaxDD={result.max_drawdown_pct:.1f}% | Fees=EUR{result.total_fees_paid:.2f}")
        print(f"  Signals: {result.signals_generated} generated, {result.signals_filled} filled "
              f"({result.fill_rate:.0%} fill rate)")

        by_strat = analyze_trades(result.trade_log)
        for strat_name in sorted(by_strat.keys()):
            print_strategy_report(strat_name, by_strat[strat_name])

        pair_results.append({
            "pair": pair, "pnl": result.total_pnl, "trades": result.total_trades,
            "win_rate": result.win_rate, "sharpe": result.sharpe_ratio,
            "max_dd": result.max_drawdown_pct, "fees": result.total_fees_paid,
            "signals": result.signals_generated, "fills": result.signals_filled,
        })

        grand_total_pnl += result.total_pnl
        grand_total_trades += result.total_trades

        for strat_name, data in by_strat.items():
            a = agg_strats[strat_name]
            a["trades"].extend(data["trades"])
            a["wins"] += data["wins"]
            a["losses"] += data["losses"]
            a["total_pnl"] += data["total_pnl"]
            a["win_pnls"].extend(data["win_pnls"])
            a["loss_pnls"].extend(data["loss_pnls"])
            for k, v in data["directions"].items():
                a["directions"][k] += v
            for k, v in data["exit_reasons"].items():
                a["exit_reasons"][k] += v
            a["hold_times"].extend(data["hold_times"])

        # Free memory
        del signal_map, lookup, df
        gc.collect()

    # === PAIR SUMMARY TABLE ===
    print(f"\n\n{'='*90}")
    print(f"  PAIR SUMMARY")
    print(f"{'='*90}")
    print(f"  {'Pair':>12s} {'P&L':>12s} {'Trades':>7s} {'WR':>6s} {'Sharpe':>7s} {'MaxDD':>7s} {'Fees':>10s}")
    print(f"  {'-'*12} {'-'*12} {'-'*7} {'-'*6} {'-'*7} {'-'*7} {'-'*10}")
    for r in sorted(pair_results, key=lambda x: x["pnl"], reverse=True):
        print(f"  {r['pair']:>12s} EUR{r['pnl']:>+9.2f} {r['trades']:>7d} {r['win_rate']:>5.0%} "
              f"{r['sharpe']:>+7.2f} {r['max_dd']:>6.1f}% EUR{r['fees']:>7.2f}")

    # === AGGREGATE ===
    total_capital = CAPITAL_PER_PAIR * len(pair_results)
    total_roi = grand_total_pnl / total_capital * 100 if total_capital > 0 else 0

    print(f"\n{'='*90}")
    print(f"  AGGREGATE RESULTS ACROSS {len(pair_results)} PAIRS")
    print(f"  Total Capital: EUR {total_capital:,.0f}")
    print(f"  Total P&L: EUR{grand_total_pnl:+,.2f} (ROI: {total_roi:+.2f}%)")
    print(f"  Total Trades: {grand_total_trades}")
    print(f"{'='*90}")

    for strat_name in sorted(agg_strats.keys()):
        print_strategy_report(strat_name, agg_strats[strat_name], total_capital)

    # === YEARLY BREAKDOWN ===
    print(f"\n\n{'='*90}")
    print(f"  YEARLY P&L BREAKDOWN")
    print(f"{'='*90}")

    all_trades = []
    for data in agg_strats.values():
        all_trades.extend(data["trades"])

    by_year = defaultdict(lambda: {"pnl": 0.0, "trades": 0, "wins": 0})
    for t in all_trades:
        try:
            year = datetime.fromisoformat(t.entry_time).year
            by_year[year]["pnl"] += t.pnl_eur
            by_year[year]["trades"] += 1
            if t.pnl_eur > 0:
                by_year[year]["wins"] += 1
        except Exception:
            pass

    print(f"  {'Year':>6s} {'P&L':>14s} {'ROI':>8s} {'Trades':>7s} {'WR':>6s}")
    print(f"  {'-'*6} {'-'*14} {'-'*8} {'-'*7} {'-'*6}")
    for year in sorted(by_year.keys()):
        y = by_year[year]
        yr_roi = y["pnl"] / total_capital * 100
        yr_wr = y["wins"] / y["trades"] * 100 if y["trades"] > 0 else 0
        print(f"  {year:>6d} EUR{y['pnl']:>+11.2f} {yr_roi:>+7.2f}% {y['trades']:>7d} {yr_wr:>5.1f}%")

    # === MONTHLY BREAKDOWN (last 2 years) ===
    print(f"\n\n{'='*90}")
    print(f"  MONTHLY P&L (LAST 2 YEARS)")
    print(f"{'='*90}")

    by_month = defaultdict(lambda: {"pnl": 0.0, "trades": 0})
    cutoff = datetime.now(timezone.utc) - timedelta(days=730)
    for t in all_trades:
        try:
            dt = datetime.fromisoformat(t.entry_time)
            if dt.replace(tzinfo=timezone.utc) >= cutoff:
                key = f"{dt.year}-{dt.month:02d}"
                by_month[key]["pnl"] += t.pnl_eur
                by_month[key]["trades"] += 1
        except Exception:
            pass

    for month in sorted(by_month.keys()):
        m = by_month[month]
        print(f"  {month:>7s}  EUR{m['pnl']:>+10.2f}  ({m['trades']} trades)")

    print(f"\n  Done!")


if __name__ == "__main__":
    main()
