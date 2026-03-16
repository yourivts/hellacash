"""
10-Config Showdown: Backtest 10 strategy configs on 5 years of data.
Optimized: one backtest run per (pair, config), no duplicates.
New engine: 4 strategies (orderflow, funding_contrarian, range, squeeze),
confluence gate, fixed fractional sizing.
"""
import sys, os, time, statistics
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.exchange.bitvavo_client import BitvavoClient
from bot.backtest.engine import BacktestEngine

PAIRS = ["BTC-EUR", "ETH-EUR", "XRP-EUR"]
YEARS = 5
DAYS = YEARS * 365

# ---------------------------------------------------------------------------
# 10 strategy configurations: (display_name, params)
# All 4 strategies run in each backtest; configs vary shared risk/timing params.
# ---------------------------------------------------------------------------
CONFIGS = [
    # Standard (matching spec defaults)
    ("Standard Default",    dict(atr_multiplier=3.0, rr_ratio=2.0, base_risk_pct=3.0, min_profit_multiple=3.0, cooldown_hours=48, max_hold_hours=72)),

    # Conservative
    ("Conservative",        dict(atr_multiplier=4.0, rr_ratio=3.0, base_risk_pct=2.0, min_profit_multiple=4.0, cooldown_hours=72, max_hold_hours=168)),

    # Aggressive
    ("Aggressive",          dict(atr_multiplier=2.5, rr_ratio=1.5, base_risk_pct=4.5, min_profit_multiple=2.0, cooldown_hours=24, max_hold_hours=48)),

    # Wide stops
    ("Wide Stops",          dict(atr_multiplier=5.0, rr_ratio=2.5, base_risk_pct=3.0, min_profit_multiple=3.0, cooldown_hours=48, max_hold_hours=120)),

    # Tight stops
    ("Tight Stops",         dict(atr_multiplier=2.5, rr_ratio=2.0, base_risk_pct=3.0, min_profit_multiple=3.0, cooldown_hours=48, max_hold_hours=72)),

    # High RR
    ("High RR",             dict(atr_multiplier=3.5, rr_ratio=4.0, base_risk_pct=3.0, min_profit_multiple=3.0, cooldown_hours=48, max_hold_hours=120)),

    # Low cooldown
    ("Low Cooldown",        dict(atr_multiplier=3.0, rr_ratio=2.0, base_risk_pct=3.0, min_profit_multiple=3.0, cooldown_hours=12, max_hold_hours=72)),

    # High cooldown
    ("High Cooldown",       dict(atr_multiplier=3.0, rr_ratio=2.0, base_risk_pct=3.0, min_profit_multiple=3.0, cooldown_hours=96, max_hold_hours=168)),

    # Quiet filter aggressive
    ("Quiet Filter Tight",  dict(atr_multiplier=3.0, rr_ratio=2.0, base_risk_pct=3.0, min_profit_multiple=3.0, cooldown_hours=48, max_hold_hours=72, quiet_atr_threshold=1.5)),

    # Relaxed fee gate
    ("Relaxed Fee Gate",    dict(atr_multiplier=3.0, rr_ratio=2.0, base_risk_pct=3.0, min_profit_multiple=2.0, cooldown_hours=48, max_hold_hours=72)),
]


@dataclass
class Result:
    name: str
    strategy: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    sharpe: float = 0.0
    max_dd: float = 0.0
    win_rate: float = 0.0
    avg_profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    expectancy: float = 0.0
    roi_pct: float = 0.0
    avg_hold_h: float = 0.0
    longs: int = 0
    shorts: int = 0
    long_pnl: float = 0.0
    short_pnl: float = 0.0
    pnl_list: List[float] = field(default_factory=list)
    # New metrics
    total_fees: float = 0.0
    fill_rate_sum: float = 0.0
    profit_factor_sum: float = 0.0
    pair_count: int = 0
    avg_fill_rate: float = 0.0
    profit_per_fee: float = 0.0


def run_config(candles_by_pair: Dict, display_name: str, params: dict) -> Result:
    """Run one config across all pairs, collecting all strategy trades."""
    r = Result(name=display_name, strategy="all")

    for pair, candles in candles_by_pair.items():
        engine = BacktestEngine(candles, initial_capital=10_000.0, max_open_positions=10, strategy_params=params)
        bt = engine.run()

        for t in bt.trade_log:
            r.trades += 1
            r.total_pnl += t.pnl_eur
            r.pnl_list.append(t.pnl_eur)
            if t.pnl_eur > 0:
                r.wins += 1
            else:
                r.losses += 1
            if t.direction == "LONG":
                r.longs += 1
                r.long_pnl += t.pnl_eur
            else:
                r.shorts += 1
                r.short_pnl += t.pnl_eur
            if t.pnl_eur > r.best_trade:
                r.best_trade = t.pnl_eur
            if t.pnl_eur < r.worst_trade:
                r.worst_trade = t.pnl_eur

        if bt.max_drawdown_pct > r.max_dd:
            r.max_dd = bt.max_drawdown_pct

        # New metrics
        r.total_fees += bt.total_fees_paid
        r.fill_rate_sum += bt.fill_rate
        r.profit_factor_sum += bt.profit_factor
        r.pair_count += 1

    if r.trades == 0:
        return r

    win_pnls = [p for p in r.pnl_list if p > 0]
    loss_pnls = [p for p in r.pnl_list if p <= 0]

    r.win_rate = r.wins / r.trades * 100
    r.avg_win = statistics.mean(win_pnls) if win_pnls else 0.0
    r.avg_loss = statistics.mean(loss_pnls) if loss_pnls else 0.0
    r.expectancy = statistics.mean(r.pnl_list)
    r.roi_pct = r.total_pnl / (10_000 * len(candles_by_pair)) * 100

    if len(r.pnl_list) >= 2:
        avg = statistics.mean(r.pnl_list)
        std = statistics.stdev(r.pnl_list)
        if std > 0:
            r.sharpe = (avg / std) * (252 ** 0.5)

    # Aggregate new metrics
    if r.pair_count > 0:
        r.avg_fill_rate = r.fill_rate_sum / r.pair_count
        r.avg_profit_factor = r.profit_factor_sum / r.pair_count
    r.profit_per_fee = r.total_pnl / r.total_fees if r.total_fees > 0 else 0.0

    return r


def main():
    n_configs = len(CONFIGS)
    print("=" * 120, flush=True)
    print(f"  {n_configs}-CONFIG SHOWDOWN: 5-YEAR BACKTEST (2021-2026)")
    print(f"  Pairs: {', '.join(PAIRS)} | Capital: EUR 10,000/pair | 5m candles")
    print(f"  Engine: 4 strategies (orderflow, funding_contrarian, range, squeeze) | Confluence gate | Fixed fractional sizing")
    print("=" * 120, flush=True)

    # Fetch data
    candles_by_pair = {}
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
        else:
            print(f" skipped (insufficient data)", flush=True)
        time.sleep(1)

    print(f"\n  Testing {n_configs} configs across {len(candles_by_pair)} pairs...\n", flush=True)

    results = []
    for i, (name, params) in enumerate(CONFIGS, 1):
        print(f"  [{i:2d}/{n_configs}] {name:25s}", end="", flush=True)
        t0 = time.time()
        r = run_config(candles_by_pair, name, params)
        elapsed = time.time() - t0
        print(
            f"  {r.trades:5d} trades  EUR{r.total_pnl:>+10.2f}"
            f"  Sharpe={r.sharpe:>+7.2f}  FillRate={r.avg_fill_rate:>5.1f}%  PF={r.avg_profit_factor:>5.2f}  ({elapsed:.0f}s)",
            flush=True,
        )
        results.append(r)

    # Rank by Sharpe
    ranked = sorted(results, key=lambda r: r.sharpe, reverse=True)

    print(f"\n\n{'=' * 120}")
    print(f"  TOP {n_configs} CONFIGS RANKED BY SHARPE RATIO")
    print(f"  5-year backtest across {len(candles_by_pair)} major crypto pairs")
    print(f"{'=' * 120}")
    print(
        f"  {'#':>3}  {'Config':25s}  {'Trades':>6}  {'P&L (EUR)':>12}  {'Sharpe':>8}"
        f"  {'WR%':>6}  {'PF':>6}  {'FillRate%':>10}  {'P/Fee':>8}  {'MaxDD%':>7}  {'ROI%':>8}"
    )
    print("  " + "-" * 117)

    for rank, r in enumerate(ranked, 1):
        print(
            f"  {rank:3d}  {r.name:25s}  {r.trades:6d}  {r.total_pnl:>+12.2f}  {r.sharpe:>+8.2f}"
            f"  {r.win_rate:>5.1f}%  {r.avg_profit_factor:>6.2f}  {r.avg_fill_rate:>9.1f}%  {r.profit_per_fee:>+8.2f}"
            f"  {r.max_dd:>6.1f}%  {r.roi_pct:>+7.2f}%"
        )

    # Top 5 detailed
    print(f"\n\n{'=' * 120}")
    print(f"  DETAILED BREAKDOWN: TOP 5")
    print(f"{'=' * 120}")
    for rank, r in enumerate(ranked[:5], 1):
        print(f"\n  #{rank} {r.name}")
        print(f"  {'='*60}")
        print(f"    Total P&L:       EUR{r.total_pnl:>+10.2f}  (ROI: {r.roi_pct:+.2f}%)")
        print(f"    Sharpe:          {r.sharpe:>+8.2f}")
        print(f"    Trades:          {r.trades:>6d}  (W:{r.wins} / L:{r.losses})")
        print(f"    Win Rate:        {r.win_rate:>6.1f}%")
        print(f"    Avg Profit Factor: {r.avg_profit_factor:>6.2f}")
        print(f"    Avg Fill Rate:   {r.avg_fill_rate:>6.1f}%")
        print(f"    Profit/Fee:      {r.profit_per_fee:>+8.2f}")
        print(f"    Expectancy:      EUR{r.expectancy:>+8.2f} / trade")
        print(f"    Avg Win:         EUR{r.avg_win:>+8.2f}  |  Avg Loss: EUR{r.avg_loss:>+8.2f}")
        print(f"    Best Trade:      EUR{r.best_trade:>+8.2f}  |  Worst:    EUR{r.worst_trade:>+8.2f}")
        print(f"    Max Drawdown:    {r.max_dd:>6.1f}%")
        print(f"    Total Fees Paid: EUR{r.total_fees:>+10.2f}")
        print(f"    LONG:  {r.longs:>5d} trades, EUR{r.long_pnl:>+10.2f}")
        print(f"    SHORT: {r.shorts:>5d} trades, EUR{r.short_pnl:>+10.2f}")

    # Bottom 5
    print(f"\n\n{'=' * 120}")
    print(f"  BOTTOM 5 (WORST PERFORMERS)")
    print(f"{'=' * 120}")
    for rank, r in enumerate(ranked[-5:], max(1, len(ranked) - 4)):
        print(f"  #{rank} {r.name:25s}  EUR{r.total_pnl:>+10.2f}  Sharpe={r.sharpe:>+8.2f}  {r.trades} trades  WR={r.win_rate:.1f}%")


if __name__ == "__main__":
    main()
