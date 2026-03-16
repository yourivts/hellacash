"""5-year comprehensive backtest with detailed per-strategy breakdown."""
import sys
import os
import time
import statistics
from datetime import datetime, timedelta, timezone
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.exchange.bitvavo_client import BitvavoClient
from bot.backtest.engine import BacktestEngine

PAIRS = ["BTC-EUR", "ETH-EUR", "SOL-EUR", "XRP-EUR", "ADA-EUR", "DOGE-EUR", "LINK-EUR", "AVAX-EUR"]
YEARS = 5
DAYS = YEARS * 365


def fetch_candles(pair, days):
    client = BitvavoClient(api_key="", api_secret="", paper_trading=True)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    print(f"  Fetching {pair} ({days} days)...", end="", flush=True)
    candles = client.get_candles_range(pair, interval="5m", start=start, end=end)
    if candles:
        actual_days = (candles[-1].timestamp - candles[0].timestamp).days
        print(f" {len(candles)} candles ({actual_days} days)", flush=True)
    else:
        print(" FAILED", flush=True)
    return candles


def analyze_trades(trade_log):
    """Detailed breakdown by strategy."""
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

        # Parse hold time from entry/exit timestamps
        try:
            entry_dt = datetime.fromisoformat(t.entry_time)
            exit_dt = datetime.fromisoformat(t.exit_time)
            hold_hours = (exit_dt - entry_dt).total_seconds() / 3600
            s["hold_times"].append(hold_hours)
        except Exception:
            pass

    return by_strat


def compute_sharpe(trade_pnls, periods_per_year=252):
    """Annualized Sharpe from trade P&Ls."""
    if len(trade_pnls) < 2:
        return 0.0
    avg = statistics.mean(trade_pnls)
    std = statistics.stdev(trade_pnls)
    if std == 0:
        return 0.0
    return (avg / std) * (periods_per_year ** 0.5)


def print_strategy_report(name, data, total_capital=10000.0):
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

    # Expectancy per trade
    expectancy = pnl / n

    # Return on capital
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
    print(f"  COMPREHENSIVE BACKTEST: {YEARS}-YEAR HISTORY, ALL STRATEGIES")
    print(f"  Capital: EUR 10,000 | Max positions: 3 | 5m candles")
    print("=" * 90, flush=True)

    # Current best parameters
    params = {
        "atr_multiplier": 3.0,
        "rr_ratio": 2.0,
        "kelly_fraction": 0.25,
        "min_confirmations": 2,
        "entry_threshold": 0.50,
        "cooldown_hours": 48,
        "max_hold_hours": 72,
    }

    # Aggregate across all pairs
    agg_strats = defaultdict(lambda: {
        "trades": [], "wins": 0, "losses": 0, "total_pnl": 0.0,
        "win_pnls": [], "loss_pnls": [], "directions": defaultdict(int),
        "exit_reasons": defaultdict(int), "hold_times": [],
    })
    grand_total_pnl = 0.0
    grand_total_trades = 0

    for pair in PAIRS:
        candles = fetch_candles(pair, DAYS)
        if not candles or len(candles) < 1000:
            print(f"  {pair}: insufficient data, skipping")
            continue

        actual_days = (candles[-1].timestamp - candles[0].timestamp).days

        t0 = time.time()
        engine = BacktestEngine(
            candles,
            initial_capital=10_000.0,
            max_open_positions=3,
            strategy_params=params,
        )
        result = engine.run()
        elapsed = time.time() - t0

        print(f"\n{'='*90}")
        print(f"  {pair} ({actual_days} days, {len(candles)} candles, {elapsed:.0f}s)")
        print(f"  Overall: P&L=EUR{result.total_pnl:+.2f} | Trades={result.total_trades} | "
              f"WR={result.win_rate:.0%} | Sharpe={result.sharpe_ratio:.2f} | MaxDD={result.max_drawdown_pct:.1f}%")

        by_strat = analyze_trades(result.trade_log)
        for strat_name in sorted(by_strat.keys()):
            print_strategy_report(strat_name, by_strat[strat_name])

        grand_total_pnl += result.total_pnl
        grand_total_trades += result.total_trades

        # Merge into aggregate
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

        time.sleep(1)  # rate limit

    # Grand summary
    print(f"\n{'='*90}")
    print(f"{'='*90}")
    print(f"  AGGREGATE RESULTS ACROSS ALL {len(PAIRS)} PAIRS")
    print(f"  Total P&L: EUR{grand_total_pnl:+.2f} | Total Trades: {grand_total_trades}")
    print(f"{'='*90}")

    for strat_name in sorted(agg_strats.keys()):
        print_strategy_report(strat_name, agg_strats[strat_name])

    # Yearly breakdown
    print(f"\n\n{'='*90}")
    print(f"  YEARLY P&L BREAKDOWN BY STRATEGY")
    print(f"{'='*90}")

    all_trades = []
    for data in agg_strats.values():
        all_trades.extend(data["trades"])

    by_year_strat = defaultdict(lambda: defaultdict(float))
    by_year_total = defaultdict(float)

    for t in all_trades:
        try:
            year = datetime.fromisoformat(t.entry_time).year
            by_year_strat[year][t.strategy] += t.pnl_eur
            by_year_total[year] += t.pnl_eur
        except Exception:
            pass

    strat_names = sorted(agg_strats.keys())
    header = f"  {'Year':>6s}"
    for s in strat_names:
        header += f"  {s:>18s}"
    header += f"  {'TOTAL':>12s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for year in sorted(by_year_strat.keys()):
        row = f"  {year:>6d}"
        for s in strat_names:
            pnl = by_year_strat[year].get(s, 0)
            row += f"  EUR{pnl:>+12.2f}"
        row += f"  EUR{by_year_total[year]:>+8.2f}"
        print(row)


if __name__ == "__main__":
    main()
