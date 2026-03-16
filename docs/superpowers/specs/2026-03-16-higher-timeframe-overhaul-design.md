# Higher Timeframe Overhaul — Design Spec

## Problem Statement

A 25-strategy showdown over 5 years of 5m candle data (BTC-EUR, ETH-EUR, XRP-EUR) showed that 22 of 25 strategy configurations lost money. Only 3 were marginally profitable — all with swing-style parameters (fewer trades, wider stops). The root causes:

1. **Fee drag** — At 0.25% taker fees round-trip (0.50%), the bot needs a 0.50%+ move just to break even. High-frequency strategies (2,000-3,900 trades) bleed out.
2. **Signal noise** — 5m candle signals are noisy. Standard indicators (RSI, MACD, EMA crossover) on 5m produce too many false entries.
3. **No fee awareness** — The bot enters trades regardless of whether the expected move covers fees.
4. **Wrong strategies** — Trend following and breakout lost money in every configuration tested.

## Goal

Redesign the trading engine to be consistently profitable on a sub-EUR 1,000 account by:
- Reducing trade frequency (fewer, higher-quality entries)
- Cutting fee costs by 40% (maker orders)
- Only trading when expected profit significantly exceeds fees
- Keeping only strategies with demonstrated or sound theoretical edge

## Constraints

- **Capital:** Under EUR 1,000
- **Exchange:** Bitvavo primary (open to Kraken later)
- **Drawdown limit:** None — user accepts full risk for maximum returns
- **Max open positions:** 10
- **Uptime:** Mostly up (local machine), occasional restarts
- **Pairs:** All available on Bitvavo

## Architecture

### Dual-Timeframe Architecture

- **Signal generation on 1h candles** — Entry decisions happen once per hour using 1h/4h/1d indicator data. Eliminates 5m noise from entry logic entirely.
- **Execution and stop monitoring on 5m candles** — Once a trade is open, stops/trailing/exits are checked every 5m for timely risk management.

### Maker-First Order Execution

- **Entry:** Limit order at or slightly inside the bid/ask (maker fee 0.15%).
- **Exit (stop-loss):** Market order — safety first.
- **Exit (take-profit):** Limit order at TP price (maker fee 0.15%).
- **Fill timeout:** 2 x 5m candles (10 min). Cancel and re-evaluate if not filled.
- **Price moved away:** Cancel if price moves > 0.3% from limit before fill.

### Fee Impact

| Scenario | Entry | Exit | Round Trip |
|----------|-------|------|------------|
| Current (taker/taker) | 0.25% | 0.25% | 0.50% |
| New (maker entry + maker TP) | 0.15% | 0.15% | 0.30% |
| New (maker entry + taker SL) | 0.15% | 0.25% | 0.40% |

---

## Strategy Roster

### Killed (negative expectancy in every configuration)

**Trend Following** — 2,374-3,895 trades across configs, lost money every time. EMA crossover + MACD + ADX generates too many signals in crypto's choppy markets.

**Breakout** — 696-2,339 trades, negative across all configs. Volume-confirmed breakouts in crypto are mostly fakeouts.

### Kept & Redesigned

#### 1. Orderflow Absorption

Best performer in the showdown (+15.95 EUR, Sharpe +0.28 in swing config, 231 trades).

**1h signal logic:**
- 1h candle with body < 30% of range
- Lower or upper wick > 40% of range
- Volume > 1.5x 20-period average
- OBV divergence over 3 candles
- CMF confirmation (same direction as signal)
- RSI between 30-70 (absorption works in the middle, not at extremes)

**Direction:**
- Long lower wick + small body = buying absorption (LONG)
- Long upper wick + small body = selling absorption (SHORT)

#### 2. Funding Contrarian

Profitable in swing config (+2.50 EUR, Sharpe +1.36, 14 trades). Sound theoretical basis — trading against crowded leverage.

**Live mode:** On-chain score from Binance funding rates + OI (unchanged).

**Backtest proxy (loosened for more trades):**
- 1h RSI > 72 OR < 28 (loosened from 78/22 on 5m)
- MACD histogram fading (momentum exhaustion)
- 4h RSI confirming the extreme (multi-TF agreement)
- Volume requirement removed (funding extremes don't need volume confirmation)

#### 3. Range Mean-Reversion

Lost money but the concept is sound — crypto spends ~70% of time ranging. Problem was tight stops on 5m getting hit by noise.

**1h signal logic:**
- 1h BB bandwidth between 1.5% and 8% (range confirmed)
- 4h ADX < 22 (not trending on higher TF)
- Price within 0.5% of 1h lower/upper Bollinger Band
- 1h RSI < 35 for longs, > 65 for shorts
- TP at mid-band, with dynamic shift to opposite band if momentum continues
- Max hold: 72h (3 days)

### New Strategy

#### 4. Volatility Squeeze

Observed in the data: profitable trades clustered around Bollinger Band squeeze events.

**1h signal logic:**
- 1h BB bandwidth contracts below 2% (compression)
- Expands above 3% in same or next candle (expansion)
- Direction: price breaks above upper BB = LONG, below lower BB = SHORT
- Volume surge > 1.5x on expansion candle
- 4h EMA50 slope direction confirms
- TP: 2x the BB width at squeeze point (measured move)
- Expected frequency: 2-5 per pair per month (rare but high-conviction)

**Note:** Range and Squeeze are mutually exclusive via regime filtering. Range runs in RANGING (ADX < 20), Squeeze runs in TRENDING/VOLATILE (ADX > 25 or ATR > 4%). When BB bandwidth is contracting (pre-squeeze), Range may fire in RANGING conditions, but the actual squeeze expansion only fires in TRENDING/VOLATILE. The confluence gate will not combine Range + Squeeze because they never run in the same regime.

### Meta-Strategy: Confluence Gate (replaces Hybrid)

The current Hybrid strategy produced 0-3 trades in 5 years because its weighted-average approach cancels signals out. Replace with a confluence detector:

- When 2+ of the **active strategies for the current regime** fire in the same direction on the same 1h candle:
  - Position size multiplied by 1.5x
  - Use wider stops (4x ATR instead of 3.5x)
  - TP at the wider of the two strategies' targets
- This rewards agreement between independent signal sources rather than averaging them into nothing.

---

## Signal Generation Flow

```
1h candle closes
    -> Compute/update 1h, 4h, 1d indicators
    -> Detect regime (4h ADX/ATR)
    -> If QUIET regime (4h ATR% < 1.0): skip, log "quiet market"
    -> Run applicable strategies for this regime
    -> Fee-aware gate: reject if expected_profit < 3x total_fees
    -> Confluence check: if 2+ strategies agree, flag for bigger size
    -> Place limit entry orders (maker)
```

### Regime Detection (4h — more stable than current 1h)

**Evaluation order** (first match wins):

| Priority | Regime | Condition | Strategies Run |
|----------|--------|-----------|---------------|
| 1 | QUIET | 4h ATR% < 1.0 | **None — skip entirely** |
| 2 | VOLATILE | 4h ATR% > 4.0 | Funding Contrarian, Squeeze |
| 3 | TRENDING | 4h ADX > 25 | Orderflow, Funding Contrarian, Squeeze |
| 4 | RANGING | 4h ADX < 20 | Range, Orderflow, Funding Contrarian |
| 5 | NEUTRAL | ADX 20-25, ATR 1.0-4.0 | Funding Contrarian, Orderflow |

The QUIET regime is new and critical. With sub-EUR 1,000 capital, trades in low-volatility periods where moves can't cover fees are guaranteed losers.

The NEUTRAL regime covers the ~20-30% of time when ADX is between 20-25 (neither clearly trending nor ranging). Only the two most versatile strategies run in this ambiguous state.

**Code change:** `detect_regime()` in `bot/strategy/router.py` changes from accepting `df_1h` to `df_4h`. Both the live trading loop and backtest engine must pass 4h data instead of 1h data to this function.

### Fee-Aware Entry Gate

```python
min_profit_multiple = 3.0  # configurable via walk-forward

entry_fee = position_size * maker_fee      # 0.15%
exit_fee  = position_size * taker_fee      # 0.25% (worst case: SL hit)
total_fee = entry_fee + exit_fee
borrow_fee = hourly_rate * expected_hold_hours  # shorts only

expected_profit = position_size * (tp_distance_pct / 100)
fee_ratio = expected_profit / (total_fee + borrow_fee)

if fee_ratio < min_profit_multiple:
    REJECT
```

At 0.40% round-trip on a EUR 50 position, total fees = EUR 0.20. Gate requires expected profit > EUR 0.60, meaning TP must be >= 1.2% away. This naturally filters low-conviction setups.

### Computation Order

The entry evaluation follows this exact sequence to resolve dependencies:

1. Compute 1h ATR value and ATR% (`atr_value = ATR(14) on 1h candles`)
2. Look up strategy-specific ATR multiplier (2.5x-4.0x per table above)
3. Compute stop distance: `stop_distance = atr_multiplier * atr_value`
4. Compute stop distance %: `stop_distance_pct = stop_distance / current_price * 100`
5. Compute TP distance: `tp_distance = stop_distance * rr_ratio + (total_fee_pct / 100 * current_price)`
6. Compute TP distance %: `tp_distance_pct = tp_distance / current_price * 100`
7. Compute position size: `position_size = risk_eur / stop_distance_pct` (see Position Sizing)
8. Run fee-aware gate: `expected_profit = position_size * tp_distance_pct / 100`; reject if `fee_ratio < min_profit_multiple`
9. Place limit order if gate passes

---

## Position Sizing

### Fixed Fractional with Volatility Scaling

```python
base_risk_pct = 3%  # risk 3% of equity per trade
equity = current_portfolio_value

# Volatility adjustment: bigger in volatile markets (bigger moves cover fees)
vol_scale = clamp(current_atr_pct / median_atr_pct, 0.5, 2.0)

risk_eur = equity * base_risk_pct * vol_scale
stop_distance_pct = atr_multiplier * atr_pct
position_size = risk_eur / stop_distance_pct

# Hard caps
position_size = min(position_size, equity * 0.30)  # max 30% in one trade
position_size = max(position_size, 10.0)            # Bitvavo minimum EUR 10
```

**Confluence bonus:** When 2+ strategies agree, bump `base_risk_pct` to 4.5%.

**Why not Kelly:** Kelly needs reliable win_rate and avg_win/avg_loss estimates. With fewer trades from 1h signals, stable Kelly estimates take weeks to accumulate. Fixed fractional is predictable and aggressive — matching the user's goal.

---

## Stop Loss & Take Profit

### Stops (ATR-based on 1h)

| Strategy | ATR Multiplier | Rationale |
|----------|---------------|-----------|
| Orderflow | 3.5x | Standard — 1h ATR is smoother than 5m |
| Funding Contrarian | 3.5x | Standard |
| Range | 2.5x | Tighter — range-bound moves are smaller |
| Squeeze | 3.0x | Moderate — squeeze breakouts are directional |
| Confluence | 4.0x | Wide — give high-conviction trades room |

### Take Profit

- Minimum R:R of 2.5:1 **after fees**
- Formula: `TP_distance = stop_distance * rr_ratio + total_fee_pct`
- Fees baked into TP target — the bot aims for 2.5:1 net of costs

### Trailing Stop

- No trailing until price has moved 1.5x the stop distance in profit (changed from current 1.0x activation)
- Then trail at 2x ATR behind the highest/lowest point
- Prevents getting shaken out early on normal retracements
- **Code change:** `trail_stop()` in `bot/risk/stop_loss.py` needs a new `activation_threshold` parameter (default 1.5). Both `_check_exits_fast` in `engine.py` and `_check_stops` in `trading_loop.py` must pass this parameter.

---

## Risk Management

### Kept from Current System
- Max open positions: **10** (changed from 5 in `config.py` and from 3 in `BacktestEngine.__init__` default — both must update)
- Per-symbol cooldown after trade closes (default 24h, walk-forward tunable)
- Daily loss circuit breaker: **EUR 50** (changed from EUR 200 in both `config.py:max_daily_loss_eur` and `drawdown_guard.py` constructor default)

### Removed
- Max drawdown hard halt — user explicitly wants no hard limit
- Soft drawdown position scaling — replaced by volatility scaling
- Min signal confidence gate (0.60) — fee-aware gate handles this
- Min trade ROI gate — replaced by fee-aware gate

### Added

**Correlation guard:** If 3+ positions are open in the same direction on correlated pairs (e.g., BTC-EUR and ETH-EUR both long), reduce size on the 4th by 50%. Prevents concentrated directional bets.

**Weekend filter (configurable, default off):** Optional skip of entries on Saturday/Sunday due to thinner liquidity. Config parameter: `weekend_filter_enabled: bool = False`.

---

## Limit Order Manager

New component managing the full lifecycle of limit orders.

### Entry Flow

```
Signal fires on 1h candle close
    -> Calculate limit price:
        LONG:  best_bid + 0.01%
        SHORT: best_ask - 0.01%
    -> Place limit order via Bitvavo REST API
    -> Start fill timeout: 2 x 5m candles (10 min)
    -> Monitor via WebSocket order updates
```

### Fill Scenarios

1. **Filled** — Set SL (monitored locally) and TP (limit order)
2. **Partially filled** — Keep partial position, cancel remainder, adjust SL/TP
3. **Not filled (timeout)** — Cancel. Re-evaluate next 1h candle.
4. **Price moved away (> 0.3%)** — Cancel immediately. Setup gone.

### Offline Recovery

```
1. Connect to Bitvavo REST API
2. Fetch all open orders -> reconcile with local state
3. Cancel stale entry orders (> 10 min old)
4. Verify TP limit orders are still active -> re-place if missing
5. Check if any SL levels were breached while offline -> market close
6. Fetch last 200 1h candles per pair -> warm up indicators
7. Resume normal cycle
```

### Rate Limit Budget

| Category | Points/min | Notes |
|----------|-----------|-------|
| Data (candles, tickers) | ~0 (WebSocket) | 5m candles and tickers arrive via WebSocket (0 REST points) |
| Order management | ~50 | ~5-15 trades/day across all pairs = ~30-90 order operations/day |
| Startup warm-up | ~200 (burst) | Fetching 200 1h candles per pair at startup only |
| Reserve (safety) | ~950 | Vast majority of budget is unused in steady state |
| **Total** | **1000** | |

With WebSocket delivering all market data, REST API usage is minimal in steady state. The 1000 point/min budget is more than sufficient even at 100 pairs.

---

## Backtest Engine Changes

### Signal Loop

- **SIGNAL_EVERY = 12** (every 1h, up from 4 = every 20m)
- Exit checks remain every 5m candle

### Fee Model

```python
entry_fee = position_size * 0.0015    # maker 0.15%
exit_fee_tp = position_size * 0.0015  # maker TP 0.15%
exit_fee_sl = position_size * 0.0025  # taker SL 0.25%
```

Conservative assumption: 80% of exits are SL/time-based (taker), 20% are TP fills (maker).

### Fill Rate Model

85% fill rate assumption for limit entries. 15% of signals are missed. This keeps the backtest honest.

**Implementation:** Deterministic, price-based model. For each signal, the limit entry price is computed as in live (bid + 0.01% for longs, ask - 0.01% for shorts). In backtest, approximate bid/ask as `close +/- spread/2` where spread = 0.05%. If the **next candle's low** (for longs) or **next candle's high** (for shorts) does not reach the limit price, the entry is skipped. This is more realistic than random sampling and produces reproducible results.

### Pre-computed Indicators

**Add:**
- 1h BB bandwidth (range + squeeze)
- 4h ADX and 4h ATR (regime detection)
- Squeeze detector boolean array (BB bandwidth crosses from < 2% to > 3%)

**Remove:**
- All trend_following pre-computation (EMA crossovers, 5m MACD checks)
- All breakout pre-computation (recent high/low checks)

### New Metrics in BacktestResult

- `profit_per_fee_eur` — Total profit / total fees paid. Target: > 2.0
- `regime_breakdown` — P&L split by TRENDING/RANGING/VOLATILE/QUIET
- `fill_rate` — Signals generated vs trades entered
- `quiet_hours_skipped` — Hours in QUIET regime (context metric)

---

## Walk-Forward Optimization

### Parameter Search Space

**New parameters:**
```
atr_multiplier:        2.5 - 5.0   (step 0.5)
rr_ratio:              2.0 - 4.0   (step 0.5)
base_risk_pct:         2.0 - 5.0   (step 0.5)
min_profit_multiple:   2.0 - 4.0   (step 0.5)
cooldown_hours:        12 - 72     (step 12)
max_hold_hours:        48 - 240    (step 24)
quiet_atr_threshold:   0.8 - 1.5   (step 0.1)
regime_adx_threshold:  20 - 30     (step 2)
```

**Removed parameters** (no longer in search space):
- `kelly_fraction` — Replaced by `base_risk_pct` (fixed fractional sizing)
- `min_confirmations` — Replaced by per-strategy thresholds (not tuned globally)
- `entry_threshold` — Replaced by fee-aware gate (`min_profit_multiple`)
- `sentiment_weight` — Hybrid strategy removed; sentiment not used in new strategies
- `consecutive_confirms` — Removed (1h signals are already low-frequency)

**Champion seed values** (initial trial for Optuna warm-start):
```
atr_multiplier: 3.5, rr_ratio: 2.5, base_risk_pct: 3.0,
min_profit_multiple: 3.0, cooldown_hours: 24, max_hold_hours: 120,
quiet_atr_threshold: 1.0, regime_adx_threshold: 25
```

### Window Sizing

```
TRAIN_DAYS  = 180  (6 months)
TEST_DAYS   = 30   (1 month OOS)
STEP_DAYS   = 30   (monthly roll)
MIN_WINDOWS = 4
```

On 5 years of data: ~16 windows.

### Optimization Target

```python
score = (sharpe * 0.4) + (profit_factor * 0.3) + (profit_per_fee * 0.3)
```

Pure Sharpe can reward single lucky trades. Adding profit_factor rewards consistency. Adding profit_per_fee ensures fee efficiency.

### Adoption Criteria

```
median_sharpe > 0.3
profitable_windows >= 70%
avg_pnl > 0
sharpe_stdev < 1.5
profit_per_fee > 1.5
```

### Per-Strategy Optimization

Separate Optuna studies per strategy for strategy-specific params (ATR multiplier, R:R, hold time). Global params (risk_pct, cooldown, quiet threshold) shared across one study.

---

## Live Trading Loop

### Cycle Timing

```
Every 5 minutes (5m candle close):
    -> Check stops, trailing, time exits for open positions
    -> Manage pending limit orders (cancel stale, check fills)

Every 1 hour (1h candle close):
    -> Update 1h/4h/1d indicators
    -> Detect regime on 4h
    -> If QUIET: skip, log "quiet market"
    -> Run applicable strategies across all pairs
    -> Fee-aware gate filter
    -> Confluence check
    -> Place limit entry orders

Every 24 hours:
    -> Reset daily loss counter
    -> Log daily P&L summary
    -> Prune expired cooldowns
```

### 1h Candle Detection

```python
# On every 5m candle WebSocket event:
cache.append_5m(candle)

# Robust 1h detection: track last evaluated hour, trigger on hour crossing
current_hour = candle.timestamp.replace(minute=0, second=0)
if current_hour != self._last_evaluated_hour:
    h1_candle = cache.build_1h_candle(current_hour - timedelta(hours=1))
    if h1_candle is not None:  # guard against missing data
        run_signal_evaluation(h1_candle)
    self._last_evaluated_hour = current_hour
```

**Why not `minute == 55`:** Bitvavo WebSocket candle timestamps represent the candle open time. If the :55 candle arrives late (network delay, reconnection), or is missed entirely, the hour-crossing approach ensures evaluation still triggers on the next available 5m candle.

### WebSocket Subscriptions

- **Keep:** 5m candles, ticker updates, order updates
- **Add:** Order book for pairs with active limit orders
- **Remove:** 1m candle subscription (5m sufficient with wider stops). **CRITICAL:** `trading_loop.py:on_candle` currently checks stops only when `data["interval"] == "1m"`. This condition must change to `"5m"` or stops will never trigger in live trading.

---

## Testing & Validation Plan

### Step 1: Backtest on 5 years

Compare old vs new engine head-to-head on same data. Target:
- Net positive total P&L (old: -2,955 EUR across configs)
- Sharpe > 0.5
- Profit per fee EUR > 2.0
- Fewer than 500 total trades

### Step 2: Walk-forward validation

180d/30d/30d windows on BTC-EUR, ETH-EUR, XRP-EUR. Must pass adoption criteria.

### Step 3: Paper trading (2 weeks minimum)

Compare:
- Trade frequency (target: 5-15 per pair per month)
- Win rate (target: 45%+)
- R:R ratio (target: 2.5:1+)
- Limit order fill rate (calibrate 85% assumption)

### Step 4: Live with 10% capital

EUR 100 for 1 month. Scale to full capital if profitable.

### Unit Tests

- Fee-aware gate: rejects trades where expected profit < 3x fees
- Limit order manager: full lifecycle (fill, partial, timeout, cancel)
- Regime detection on 4h: QUIET triggers below ATR threshold
- Each strategy's 1h signal generation
- Confluence gate: detects 2+ strategy agreement, bumps size
- Correlation guard: reduces size at 3+ correlated positions
- Startup recovery: correct reconciliation after offline period
- Backtest engine: SIGNAL_EVERY=12, maker fees, fill rate discount

---

## File Changes Summary

### New Files
- `bot/execution/limit_order_manager.py` — Limit order lifecycle management
- `bot/strategy/squeeze.py` — Volatility Squeeze strategy
- `bot/strategy/confluence.py` — Confluence Gate meta-strategy

### Major Modifications
- `bot/backtest/engine.py` — 1h signal loop, maker fee model, fill rate, new metrics, remove trend/breakout
- `bot/strategy/router.py` — New regime detection on 4h, QUIET regime, updated strategy roster
- `bot/strategy/orderflow.py` — Redesigned for 1h candle signals
- `bot/strategy/funding_contrarian.py` — Loosened thresholds for 1h
- `bot/strategy/range_trading.py` — Redesigned for 1h bands, wider stops, max hold changed from 24h (288 bars) to 72h (864 bars)
- `bot/risk/position_sizer.py` — Fixed fractional with volatility scaling (replaces Kelly). All call sites that import `kelly_size` must update: `trading_loop.py`, `backtest/engine.py`
- `bot/risk/stop_loss.py` — Updated ATR multipliers, fee-adjusted TP, new `activation_threshold` param on `trail_stop()`
- `bot/risk/engine.py` — Remove drawdown halt, add correlation guard, fee-aware gate
- `bot/trading_loop.py` — New cycle timing, 1h signal evaluation, limit order integration, stop-check trigger changed from `"1m"` to `"5m"`, `kelly_size` import replaced with fixed fractional sizing
- `bot/learning/walk_forward.py` — New param space, window sizing, adoption criteria, per-strategy optimization
- `bot/config.py` — New config params (base_risk_pct, min_profit_multiple, quiet_atr_threshold, weekend_filter_enabled, etc.), max_open_positions default 5 -> 10, max_daily_loss_eur default 200 -> 50
- `bot/data_loader.py` — Add `build_1h_candle()` method to CandleCache for aggregating 5m candles into 1h candles, referenced by 1h candle detection logic

### Deleted Logic (within existing files)
- `bot/strategy/trend_following.py` — Strategy disabled (file kept, router stops calling it)
- `bot/strategy/breakout.py` — Strategy disabled (file kept, router stops calling it)
- `bot/strategy/hybrid.py` — Replaced by confluence gate
- `bot/strategy/mtf_voter.py` — No longer used (strategies generate signals individually on 1h, not via MTF composite)
- Trend/breakout pre-computation in backtest engine
- Kelly criterion sizing (all call sites: `position_sizer.py`, `trading_loop.py`, `backtest/engine.py`)
- 1m candle subscription and `"1m"` stop-check trigger
- Drawdown hard halt and soft drawdown position scaling
- Walk-forward parameters: `kelly_fraction`, `min_confirmations`, `entry_threshold`, `sentiment_weight`, `consecutive_confirms`
