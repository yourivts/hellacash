# Range Trading Strategy — Design Spec

## Problem

The bot's trend-following strategies (hybrid, trend_following, breakout) lose money during ranging/choppy markets. Backtesting BTC-EUR Sep-Dec 2025 shows -€63 over 98 trades — the strategy enters on lagging signals that reverse immediately, causing whipsaw losses in both directions.

The current `mean_reversion` strategy also fails in these conditions because it uses the same ATR-based exit logic as trend strategies. It's a trend strategy wearing a mean-reversion mask.

## Goal

Build a dedicated range-trading strategy that profits from price oscillating within a range, turning the bot's worst-performing market regime into a revenue source.

## Design Decisions (from brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Replace or add alongside mean_reversion | **Replace** | mean_reversion loses money; two range strategies add complexity without benefit |
| Range detection method | **Bollinger Bands + volume clustering** | BB for range boundaries (pre-computed, adaptive), volume for entry confirmation (filters false bounces) |
| Exit logic | **Dynamic TP** | Initial TP at mid-band; shift to opposite band if momentum continues; close at mid if momentum fades |
| Timeframe | **5m entries, 1h confirmation** | 1h confirms range intact, 5m times exact entry at band boundary |
| Trade cap per range | **3 bounces max** | Ranges weaken with each test; stop before likely breakout |

## Architecture

### 1. Strategy Signal Generation

**New file: `bot/strategy/range_trading.py`**

`RangeStrategy` extends `BaseStrategy`. `name = "range"`.

Generates signals when all conditions are met:

**Range confirmation (1h timeframe):**
- Bollinger bandwidth < 0.10 (raw ratio, meaning the band width is less than 10% of the mid price)
- ADX < 20 (no trend)
- Bandwidth must be >= 0.02 (minimum 2% range to ensure enough room for profit after round-trip fees + slippage)

**Entry trigger (5m timeframe):**
- LONG: price within 1% of lower BB + RSI < 40 + volume profile score > 0.3 (volume support confirmed, see Section 3 for threshold definition)
- SHORT: price within 1% of upper BB + RSI > 60 + volume profile score < -0.3 (volume resistance confirmed)
- RSI thresholds (40/60) are intentionally more relaxed than traditional mean-reversion (30/70) to generate more entry opportunities in ranges where price oscillates in a narrower RSI band

**Bounce counter:**
- `Dict[str, int]` on the strategy instance, keyed by symbol
- Tracks completed round trips (entry + exit) per symbol in the current range
- After 3 bounces, stops signaling until range resets
- **Reset condition**: bounce counter resets to 0 when BB bandwidth > 0.15 OR ADX > 25 on the 1h timeframe
- Dead zone (bandwidth 0.10-0.15, ADX 20-25): strategy won't enter new trades (range not confirmed) but bounce counter is preserved. If the range tightens back below 0.10, the old bounce count still applies. This is intentional — a range that briefly wobbled but didn't break is the same range.
- Failed orders do NOT count toward the bounce limit — only closed trades with strategy_name "range" increment the counter

**Signal output:**
- `indicator_snapshot` includes `range_mid`, `range_upper`, `range_lower`, `bounce_count`, `confirming_count`
- `confirming_count` based on: RSI extreme (1) + volume support (1) + BB position (1) = max 3

### 2. Exit Logic (Dynamic Take-Profit)

Different from trend strategies — uses range-relative levels, not ATR-based stops.

**Stop loss:**
- LONG: `lower_band - 0.5 * bandwidth_price` where `bandwidth_price = upper_band - lower_band`. Just below the range.
- SHORT: `upper_band + 0.5 * bandwidth_price`. Just above the range.
- Tighter than trend stops — ranging markets have smaller moves

**Take profit — two phases:**
1. **Phase 1**: TP at middle band (the mean). Conservative, high hit-rate target.
2. **Phase 2 (dynamic shift)**: When price reaches mid-band, evaluate RSI momentum:
   - RSI still moving in trade's favor (for LONG: current RSI > RSI from 3 bars ago; for SHORT: current RSI < RSI from 3 bars ago) → shift TP to opposite band, activate tight trailing stop (1% from highest price for LONG, 1% from lowest for SHORT)
   - RSI flattening or reversing → close entire position at mid-band

**Max hold time:** 24 hours. Uses a range-specific hold limit of `24 * 12 = 288` 5m bars (separate from the trend strategies' `_max_hold_bars`). This is hardcoded, not walk-forward optimized — range trades should always resolve quickly.

**Regime change mid-trade:** If the regime shifts to "trending" or "volatile" while a range trade is open, the trade continues with its original range levels. The stop loss and TP were set based on the range that existed at entry. Force-closing on regime change would lock in losses unnecessarily — let the existing SL/TP/max-hold handle the exit.

**Position state tracking (additions to `_OpenPosition` in backtest):**
- `tp_shifted: bool = False` — whether dynamic TP has activated
- `range_mid: float = 0.0` — mid-band at entry time
- `range_upper: float = 0.0` — upper band at entry time
- `range_lower: float = 0.0` — lower band at entry time

### 3. Volume Profile Indicator

**New function in `bot/indicators/volume.py`:**

`volume_profile_support(close, volume, high, low, lookback=200) -> pd.Series`

Returns a score from -1.0 to +1.0 per bar:
- Positive = volume concentrated near current price from below (support)
- Negative = volume concentrated from above (resistance)
- Near zero = no clear volume clustering

**Implementation:**
- Divide the recent price range (lookback window) into buckets (0.1% width)
- Sum volume per bucket
- Compare volume in the lower 20% of range vs upper 20%
- Score = (lower_volume - upper_volume) / total_volume

**Backtest optimization:**
- Pre-computed once on the full series in `_precompute_signals`
- Stored as numpy array for O(1) per-bar lookup

### 4. Integration Points

**`bot/strategy/router.py`:**
- Replace `MeanReversionStrategy` with `RangeStrategy`
- `self._range = RangeStrategy()` replaces `self._mean_rev`
- Ranging regime: `[self._hybrid, self._range]`
- Position size modifier stays at 0.5 for ranging

**`bot/backtest/engine.py`:**

*Pre-computation additions to `_precompute_signals`:*
- `bb_upper`, `bb_lower`, `bb_mid`, `bb_bandwidth` — full Bollinger band arrays (currently only `bb_pct_b` is stored)
- `vol_profile` — volume profile support score array
- `rsi` is already pre-computed

*New `"range"` branch in `_evaluate_precomputed`:*
- Check 1h BB bandwidth + ADX for range confirmation
- Check 5m price vs BB bands + RSI + volume profile for entry
- Track bounce count via `self._range_bounces: Dict[str, int]` on the engine instance
- Reset bounce count when 1h BB bandwidth > 0.15 or ADX > 25

*Range-specific exit handling in `_check_exits_fast`:*
- Method signature change: add `precomp_5m` dict and `idx_5m` int as optional parameters (defaulting to None). Only passed when range positions exist, avoiding performance cost when no range trades are open.
- Dynamic TP logic: when `pos.strategy == "range"` and price crosses `pos.range_mid`:
  - Look up `precomp_5m["rsi"][idx_5m]` and compare to `precomp_5m["rsi"][idx_5m - 3]` (3 bars back at 5m = 15 min)
  - If RSI trending favorably → set `pos.tp_shifted = True`, update TP to `pos.range_upper` (LONG) or `pos.range_lower` (SHORT), activate trailing stop
  - If RSI flat/reversing → close at mid-band
- 24h max hold: `if current_bar - pos.entry_bar >= 288: close`

**`bot/trading_loop.py`:**

*Entry changes:*
- When opening a range position, store `range_mid`, `range_upper`, `range_lower` from the signal's `indicator_snapshot` onto the position dict

*Exit changes in `_check_stops`:*
- When `pos.get("strategy_name") == "range"`:
  - Compute RSI from the last 20 5m candles (already available in candle cache, computed once per stop check — every 1m candle but RSI computation on 20 bars is negligible)
  - If price crosses `pos["range_mid"]` and `tp_shifted` is not set:
    - Compare current RSI to RSI 3 candles ago
    - Shift TP or close, same logic as backtest
  - If `tp_shifted`, apply tight 1% trailing stop

**`bot/indicators/volume.py`:**
- Add `volume_profile_support()` function

**Deleted:**
- `bot/strategy/mean_reversion.py` — fully replaced
- Remove import from `bot/strategy/router.py`
- Remove `"mean_reversion"` branch from `_evaluate_precomputed` in backtest engine

### 5. Walk-Forward Compatibility

No changes to walk-forward optimizer needed. The backtest engine's regime detection automatically selects the range strategy during ranging periods. Optuna's parameter search tests entry thresholds and other params that affect whether and when the range strategy fires.

The range strategy will be tested in the walk-forward windows where the market is ranging — exactly the periods where the current strategy loses money.

### 6. Bounce Counter State Management

**Backtest engine:** `self._range_bounces: Dict[str, int]` on the `BacktestEngine` instance. Initialized to `{}` in `__init__`. Incremented in `_close_position` when `pos.strategy == "range"`. Reset checked in `_evaluate_precomputed` before generating range signals. Automatically clean between backtest runs since each run creates a new engine instance.

**Live trading:** `RangeStrategy` instance holds `self._bounce_counts: Dict[str, int]`. The strategy's `generate_signal` method checks and respects the counter. The counter is incremented when the trading loop closes a range position (via a callback or by the strategy polling closed trades). Reset is checked each signal evaluation cycle based on 1h BB bandwidth and ADX.

## Success Criteria

- Backtest BTC-EUR Sep-Dec 2025 (ranging period): P&L improves from -€63 to >= €0
- Backtest BTC-EUR full 180 days: total P&L remains positive (range strategy does not hurt trending periods)
- Ranging-period win rate >= 45% (validates the "high hit-rate conservative targets" thesis)
- Ranging-period max drawdown <= 0.5%
- Minimum 10 trades during the ranging backtest (ensures the strategy is actually trading, not just avoiding losses by inactivity)
- Sharpe ratio during ranging periods improves from current -5.28 to > -1.0
