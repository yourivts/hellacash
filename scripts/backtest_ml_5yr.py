"""5-year backtest of the full system with ML signal generator.

Loads trained Transformer + XGBoost models and runs a realistic backtest
across all pairs, reporting per-pair and aggregate P&L, Sharpe, drawdown,
and yearly breakdown.
"""
import sys, os, time, statistics
from datetime import datetime, timedelta, timezone
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.exchange.bitvavo_client import BitvavoClient
from bot.backtest.engine import BacktestEngine
from bot.learning.ml_signal_generator import MLSignalGenerator

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


def fetch_candles(pair, days):
    client = BitvavoClient(api_key="", api_secret="", paper_trading=True)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    print(f"  Fetching {pair}...", end="", flush=True)
    candles = client.get_candles_range(pair, interval="5m", start=start, end=end)
    if candles:
        actual_days = (candles[-1].timestamp - candles[0].timestamp).days
        print(f" {len(candles):,} candles ({actual_days} days)", flush=True)
    else:
        print(" FAILED", flush=True)
    return candles


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
    print(f"  ML SIGNAL GENERATOR: {YEARS}-YEAR BACKTEST")
    print(f"  Transformer + XGBoost | Capital: EUR {CAPITAL_PER_PAIR:,.0f}/pair | 5m candles")
    print("=" * 90, flush=True)

    # Load ML signal generator
    print("\n  Loading ML models...", end="", flush=True)
    try:
        ml_gen = MLSignalGenerator(model_dir="models/ml_signals")
        print(" OK", flush=True)
    except FileNotFoundError as e:
        print(f" FAILED: {e}")
        sys.exit(1)

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
        candles = fetch_candles(pair, DAYS)
        if not candles or len(candles) < 10000:
            print(f"  {pair}: insufficient data ({len(candles) if candles else 0}), skipping")
            continue

        actual_days = (candles[-1].timestamp - candles[0].timestamp).days

        t0 = time.time()
        engine = BacktestEngine(
            candles,
            initial_capital=CAPITAL_PER_PAIR,
            max_open_positions=3,
            strategy_params=PARAMS,
            ml_signal_generator=ml_gen,
        )
        result = engine.run()
        elapsed = time.time() - t0

        print(f"\n{'='*90}")
        print(f"  {pair} ({actual_days} days, {len(candles):,} candles, {elapsed:.0f}s)")
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

        time.sleep(1)

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
