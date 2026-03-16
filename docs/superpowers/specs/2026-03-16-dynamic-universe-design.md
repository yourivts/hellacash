# Dynamic Tradeable Universe — Design Spec

## Goal

Only trade symbols where walk-forward optimization proves an edge, and only with strategies that are individually profitable on each symbol. Each strategy-symbol combination gets its own optimized parameters. Non-adopted symbols are monitored (candles, stops on existing positions) but no new trades are opened.

## Problem

Currently walk-forward runs on all ~25 tradeable symbols, finds per-symbol optimal params, but:
1. Only the single best-adopted symbol's params are applied globally to the router
2. All symbols are traded regardless of adoption status — losing symbols still receive trades
3. All strategies run on all symbols — even if 3 strategies lose money on a symbol and only 1 wins, all 4 still trade
4. Params are optimized for all strategies combined — a winning strategy might carry losers, producing compromise params that are suboptimal for every individual strategy
5. Per-symbol params like `atr_multiplier`, `rr_ratio`, `min_profit_multiple` are never threaded to the live trading loop calls (`initial_stops()`, `check_fee_gate()` use their function defaults instead)
6. Inline regime detection in both trading_loop.py and backtest/engine.py hardcodes thresholds instead of using walk-forward-optimized values
7. Analytics API `POST /walk-forward/run` calls dead method `router.update_hybrid_params()` (existing bug)

## Architecture

Walk-forward optimizes params **per strategy per symbol** by running backtests with a single strategy at a time. Each strategy-symbol combination is independently adopted or rejected. The `AdoptedUniverse` stores the results; the trading loop consults it to decide which strategies to run on which symbols with which params.

```
Walk-Forward (per strategy per symbol)
  ├─ BacktestEngine(target_strategy="orderflow") → optimize params
  ├─ BacktestEngine(target_strategy="range")     → optimize params
  ├─ BacktestEngine(target_strategy="squeeze")   → optimize params
  ├─ BacktestEngine(target_strategy="funding_contrarian") → optimize params
  └─ adopt/reject each combo independently
       ↓
AdoptedUniverse
  symbol → strategy → params
       ↓
Trading Loop (both run_cycle + _evaluate_1h_signals)
  ├─ skip non-adopted symbols
  ├─ only run adopted strategies for that symbol
  └─ use per-strategy-per-symbol params for stops, sizing, fees
```

Walk-forward continues to run on ALL `tradeable_symbols`. The universe is a filter on top of that set, not a replacement.

## Components

### 1. AdoptedUniverse (`bot/strategy/adopted_universe.py`)

New class that stores adopted strategy-symbol combinations with per-combo parameters. This is the **canonical source of truth** for champion default values — walk-forward imports `CHAMPION_DEFAULTS` from here for its seed trial.

```python
CHAMPION_DEFAULTS = {
    "atr_multiplier": 3.5,
    "rr_ratio": 2.5,
    "base_risk_pct": 3.0,
    "min_profit_multiple": 3.0,
    "cooldown_hours": 24,
    "max_hold_hours": 120,
    "quiet_atr_threshold": 1.0,
    "regime_adx_threshold": 24,
}

ALL_STRATEGIES = ["orderflow", "range", "squeeze", "funding_contrarian"]

@dataclass
class StrategyConfig:
    params: Dict[str, Any]
    avg_sharpe: float = 0.0
    avg_pnl: float = 0.0

@dataclass
class SymbolConfig:
    strategies: Dict[str, StrategyConfig]  # strategy_name → config
    regime_params: Dict[str, float]        # quiet_atr_threshold, regime_adx_threshold

class AdoptedUniverse:
    _adopted: Dict[str, SymbolConfig]  # symbol → config

    def is_adopted(self, symbol: str) -> bool
    def get_strategy_params(self, symbol: str, strategy: str) -> Dict[str, Any]
    def get_regime_params(self, symbol: str) -> Dict[str, float]
    def get_enabled_strategies(self, symbol: str) -> List[str]
    def update(self, wf_results: Dict[str, Dict[str, WFResult]]) -> None
    def adopted_symbols(self) -> List[str]
    def count(self) -> int
```

**`update()` logic:**
- Input: `Dict[str, Dict[str, WFResult]]` — `symbol → strategy_name → WFResult`
- For each symbol: iterate strategy results
  - If `result.adopted == True` and `result.recommended_params is not None`: include this strategy
  - If at least one strategy is adopted for a symbol: adopt the symbol
  - Regime params (`quiet_atr_threshold`, `regime_adx_threshold`) come from the best-performing strategy for that symbol (highest avg Sharpe). `volatile_atr_threshold` stays at its default (4.0) — not optimized.
- If zero symbols pass: keep previous universe unchanged (log warning)
- Log diff: `"Universe updated: +BTC-EUR(orderflow,squeeze) +SOL-EUR(range), -DOGE-EUR, 3 symbols / 5 strategy-combos adopted"`
- Replace `_adopted` dict atomically (full swap, not incremental mutation)

**`get_strategy_params()` logic:**
- If symbol+strategy is adopted: return its optimized params
- Otherwise: return `CHAMPION_DEFAULTS`

### 2. BacktestEngine Changes (`bot/backtest/engine.py`)

#### `target_strategy` filter

Add an optional `target_strategy: Optional[str] = None` parameter to `BacktestEngine.__init__()`.

When set, the engine only evaluates signals from that one strategy — all others are skipped. This is a filter applied in `_evaluate_precomputed()` where the engine iterates strategies from the router:

```python
strategies = self._router.get_strategies(regime)
if self._target_strategy:
    strategies = [s for s in strategies if s.name == self._target_strategy]
```

**Confluence is disabled** when `target_strategy` is set, since confluence requires 2+ strategies agreeing. The `check_confluence()` call is skipped.

#### Regime detection uses `strategy_params`

The engine's inline regime detection (currently hardcoded `1.0`, `4.0`, `25`, `20`) must read thresholds from `strategy_params`:

```python
quiet_thresh = self._strategy_params.get("quiet_atr_threshold", 1.0)
regime_adx = self._strategy_params.get("regime_adx_threshold", 24)
# volatile_atr_threshold stays at 4.0 (not optimized)
# ranging_adx_threshold stays at 20.0 (not optimized)
```

This ensures that when Optuna optimizes `quiet_atr_threshold` and `regime_adx_threshold`, the backtest actually uses those values during the optimization — otherwise the optimization is partially blind.

**`regime_adx_threshold` mapping:** This param maps to `trending_adx_threshold` in `detect_regime()`. The separate `ranging_adx_threshold` (default 20.0) and `volatile_atr_threshold` (default 4.0) remain fixed — they are not optimized. This keeps the search space manageable while allowing the most impactful thresholds to be tuned.

### 3. Walk-Forward Per-Strategy Optimization (`bot/learning/walk_forward.py`)

#### Candle prefetch optimization

Currently each walk-forward window fetches candles separately via the API, causing massive redundancy (overlapping 180d windows re-fetch most of the same data). With per-strategy optimization this would be `25 symbols × 4 strategies × 8 fetches = 800 API calls`.

**Fix: fetch once per symbol, slice in memory.**

The caller (`_run_walk_forward` in `main.py`) fetches the full date range for each symbol (total_days = TRAIN_DAYS + TEST_DAYS × MIN_WINDOWS = 300 days) in a single API call, then passes the full candle array to `run_multi_per_strategy()`. The walk-forward slices it per window using array indexing — no additional API calls.

```python
# In main.py _run_walk_forward():
for symbol in symbols:
    all_candles = await fetch_full_range(symbol, total_days=300)
    results[symbol] = await walk_forward.run_symbol_per_strategy(
        all_candles, symbol, strategies
    )
```

This reduces API calls from 800 to 25 (one per symbol). The same candle array is reused across all 4 strategies and all 4 windows for that symbol.

#### New method: `run_multi_per_strategy()`

Replaces `run_multi()` as the primary entry point. For each symbol, fetches candles once, then runs walk-forward independently for each strategy:

```python
async def run_multi_per_strategy(
    self, candle_fetcher_factory, symbols, strategies=None
) -> Dict[str, Dict[str, WFResult]]:
    """Run walk-forward for each strategy-symbol combo.
    Returns: {symbol: {strategy_name: WFResult}}
    """
    strategies = strategies or ALL_STRATEGIES
    results = {}
    for symbol in symbols:
        # Fetch all candles once for this symbol
        all_candles = await self._fetch_full_range(candle_fetcher_factory, symbol)
        results[symbol] = {}
        for strategy in strategies:
            result = await self._execute_from_candles(
                all_candles, symbol, target_strategy=strategy
            )
            results[symbol][strategy] = result
    return results
```

#### `_execute_from_candles()` — new method replacing `_execute()`

Takes the full candle array and slices per window instead of fetching:

```python
async def _execute_from_candles(
    self, all_candles, symbol, target_strategy=None
) -> WFResult:
    windows_spec = self._generate_windows(total_days)
    for ws in windows_spec:
        train_candles = slice_candles(all_candles, ws["train_start_day"], ws["train_end_day"])
        test_candles = slice_candles(all_candles, ws["test_start_day"], ws["test_end_day"])
        best_params, test_result = await loop.run_in_executor(
            None, _run_optuna_window, train_candles, test_candles, max_workers, target_strategy
        )
        ...
```

**`_run_optuna_window()` and `_run_single_backtest()` gain `target_strategy` parameter**, passed through to `BacktestEngine`:

```python
engine = BacktestEngine(
    candles, strategy_params=params,
    slippage_pct=0.001, target_strategy=target_strategy
)
```

**Compute budget:** 25 symbols × 4 strategies = 100 walk-forward optimizations. Each single-strategy backtest is faster (fewer trades). API calls reduced from 800 to 25. Estimated wall-clock: ~6-8h for a weekly run. The weekly scheduler (168h interval) easily accommodates this.

**`_should_adopt()` is unchanged** — same criteria (median_sharpe > 0.3, 70%+ windows profitable, sharpe_std < 1.5, avg_ppf > 1.5) applied independently to each strategy-symbol combo.

**Champion seed consolidation:** Both `enqueue_trial()` calls import `CHAMPION_DEFAULTS` from `bot/strategy/adopted_universe` instead of duplicating values inline.

### 4. Trading Loop Changes (`bot/trading_loop.py`)

**BOTH signal evaluation paths** must be gated by the universe:

#### Path 1: `run_cycle()` (30-second strategy loop)

```python
for symbol in tradeable_symbols:
    if not self._universe.is_adopted(symbol):
        continue  # skip signal evaluation, but stops still monitored

    enabled = self._universe.get_enabled_strategies(symbol)
    regime_params = self._universe.get_regime_params(symbol)

    # Regime detection with per-symbol params (refactored from inline)
    regime = detect_regime(df_4h,
        quiet_atr_threshold=regime_params["quiet_atr_threshold"],
        regime_adx_threshold=regime_params["regime_adx_threshold"])

    # Get strategies for this regime, then filter to only enabled ones
    strategies = router.get_strategies(regime)
    strategies = [s for s in strategies if s.name in enabled]

    if not strategies:
        continue

    # Evaluate each enabled strategy with its own params
    best_signal = None
    best_strat_params = None
    for strategy in strategies:
        sig = strategy.generate_signal(ctx)
        if sig.direction == "NEUTRAL":
            continue
        if best_signal is None or sig.strength > best_signal.strength:
            best_signal = sig
            best_strat_params = self._universe.get_strategy_params(symbol, strategy.name)

    if best_signal and best_strat_params:
        # Use winning strategy's params for stops, sizing, fees:
        # - initial_stops(..., atr_multiplier=best_strat_params["atr_multiplier"],
        #                      rr_ratio=best_strat_params["rr_ratio"])
        # - fixed_fractional_size(..., base_risk_pct=best_strat_params["base_risk_pct"])
        # - check_fee_gate(..., min_profit_multiple=best_strat_params["min_profit_multiple"])
        # - cooldown: best_strat_params["cooldown_hours"] * 3600
```

#### Path 2: `_evaluate_1h_signals()` (1h candle boundary)

Same adoption check, strategy filtering, and per-strategy param injection as `run_cycle()`.

**Confluence with reduced strategy sets:** The current 1h path mandates confluence (2+ strategies agreeing). When only 1 strategy is enabled for a symbol, relax this requirement — treat the single strategy's signal as sufficient (skip the `check_confluence()` gate). When 2+ strategies are enabled, confluence still applies as before.

#### Regime detection refactor

Both paths currently hardcode regime thresholds inline and compute ADX/ATR manually. The refactor:

1. Compute `adx` and `atr` columns on the resampled 4h DataFrame (using existing `compute_adx()` and `compute_atr()` helpers)
2. Add the computed columns to the DataFrame
3. Call `detect_regime(df_4h, ...)` which reads from those columns

```python
from bot.strategy.router import detect_regime
df_4h["atr"] = compute_atr(df_4h, period=14)
df_4h["adx"] = compute_adx(df_4h, period=14)
regime = detect_regime(df_4h,
    quiet_atr_threshold=regime_params["quiet_atr_threshold"],
    trending_adx_threshold=regime_params["regime_adx_threshold"],
)
```

#### Cooldown: per-symbol instead of global

Remove the global `self._cooldown_seconds` instance variable. Read cooldown from the winning strategy's params:

```python
cooldown_s = best_strat_params["cooldown_hours"] * 3600
if time.monotonic() - self._last_trade_closed.get(symbol, 0) < cooldown_s:
    continue
```

**Stop monitoring:** Continues for ALL symbols regardless of adoption. Existing positions must still be managed even if the symbol was dropped from the universe.

**`max_hold_hours`:** Currently only enforced in `BacktestEngine`, not the live loop. Out of scope for this feature — noted as future enhancement.

**Startup behavior:** Universe starts empty. No trades until walk-forward completes and populates it. This prevents trading with unvalidated params.

### 5. Walk-Forward Results Processing (`bot/main.py`)

`_process_walk_forward_results` changes:

**Signature changes** to accept `Dict[str, Dict[str, WFResult]]` (symbol → strategy → result).

**Before:**
- Find single best adopted result
- Apply its params globally via `router.update_params()`
- Update global `trading_loop._cooldown_seconds`

**After:**
- Call `universe.update(results)` which stores per-strategy-per-symbol params for all adopted combos
- Remove `router.update_params()` call (params now come from universe, not router)
- Remove global `_cooldown_seconds` update (now per-strategy)
- Log the adopted universe (which symbols, which strategies, how many combos)
- Discord notification with adopted/dropped symbols and their enabled strategies

**Discord notifier update:** `send_walk_forward_report()` currently expects flat `Dict[str, WFResult]`. Update to handle the nested `Dict[str, Dict[str, WFResult]]` structure, or pass a flattened summary.

**Logging format:**
```
WALK-FORWARD ANALYSIS COMPLETE — 25 symbols × 4 strategies = 100 combos evaluated
  BTC-EUR    | orderflow: ADOPTED (sharpe=1.2, pnl=€+3400) | range: NOT ADOPTED | squeeze: ADOPTED | funding_contrarian: NOT ADOPTED
  ETH-EUR    | orderflow: NOT ADOPTED | range: ADOPTED (sharpe=1.5, pnl=€+2100) | squeeze: NOT ADOPTED | funding_contrarian: ADOPTED
  ...
  Adopted: 3 symbols, 5 strategy-combos out of 100
```

### 6. Walk-Forward Caller (`bot/main.py`)

`_run_walk_forward()` changes to call the new `run_multi_per_strategy()` instead of `run_multi()`:

```python
async def _run_walk_forward():
    symbols = get_tradeable_symbols() or ["BTC-EUR"]
    results = await walk_forward.run_multi_per_strategy(
        _wf_candle_fetcher_factory, symbols
    )
    await _process_walk_forward_results(results, _get_router, trading_loop, discord, settings)
```

### 7. Analytics API (`api/routers/analytics.py`)

The manual walk-forward trigger (`POST /walk-forward/run`) currently calls the dead method `router.update_hybrid_params()` (existing bug). Fix this and update to:
- Run on all tradeable symbols with per-strategy optimization
- Pass results through `universe.update()` instead of the dead method

Add a new endpoint for operational visibility:

```
GET /universe → {
    "adopted_count": 3,
    "strategy_combo_count": 5,
    "symbols": {
        "BTC-EUR": {
            "regime_params": {"quiet_atr_threshold": 1.0, "regime_adx_threshold": 24},
            "strategies": {
                "orderflow": {"params": {"atr_multiplier": 3.0, ...}, "avg_sharpe": 1.2, "avg_pnl": 3400},
                "squeeze":   {"params": {"atr_multiplier": 4.0, ...}, "avg_sharpe": 0.8, "avg_pnl": 890}
            }
        },
        ...
    }
}
```

### 8. Main.py Wiring

- Create `AdoptedUniverse` singleton in `_main()`
- Pass to `TradingLoop` as `self._universe`
- Pass to `_process_walk_forward_results`

### 9. WalkForwardOptimizer Cache Update

The existing `self._results_by_symbol: Dict[str, WFResult]` cache and `result_for_symbol()` method need updating to handle the new per-strategy structure. Change to `Dict[str, Dict[str, WFResult]]` (symbol → strategy → result).

### 10. Router Changes

- `router.update_params()` remains for backward compatibility but is no longer called from walk-forward results processing
- Strategies don't change internally — per-strategy params affect position sizing, stops, regime detection, and fee gate at the trading loop level

## Data Flow

```
Startup:
  _main() → creates AdoptedUniverse (empty)
  → trading loop gets reference
  → walk-forward scheduled
  → no trades until walk-forward completes

Walk-Forward (weekly):
  run_multi_per_strategy(ALL tradeable symbols, ALL strategies)
  For each symbol:
    1. Fetch full 300-day candle range ONCE via API (single call)
    2. For each strategy:
       a. Slice candles per window from prefetched array (no API calls)
       b. BacktestEngine(target_strategy="orderflow") → Optuna optimizes params
          - Engine uses strategy_params for regime thresholds (not hardcoded)
          - Confluence disabled (single strategy isolation)
       c. 4 rolling windows: 180d train, 30d test
       d. _should_adopt() applied independently to this combo
       e. Store WFResult per strategy per symbol
  API calls: 25 total (1 per symbol) instead of 800
  → _process_walk_forward_results()
    → universe.update(results)
      → for each symbol: collect adopted strategies + their params
      → regime params from best-performing strategy
      → store {symbol: SymbolConfig(strategies, regime_params)}
      → log diff (added/removed combos)
    → discord notification (updated for nested structure)

Trading Loop — run_cycle (every 30s):
  for symbol in tradeable_symbols:
    if not universe.is_adopted(symbol): continue
    regime_params = universe.get_regime_params(symbol)
    enabled = universe.get_enabled_strategies(symbol)
    → compute adx/atr on 4h DataFrame
    → detect_regime(regime_params)
    → get_strategies(regime) filtered to enabled list
    → for each enabled strategy with signal:
      strat_params = universe.get_strategy_params(symbol, strategy.name)
    → best signal wins, trade executes with winning strategy's params:
      → initial_stops(strat_params[atr_multiplier], strat_params[rr_ratio])
      → fixed_fractional_size(strat_params[base_risk_pct])
      → check_fee_gate(strat_params[min_profit_multiple])
      → cooldown check: strat_params[cooldown_hours]

Trading Loop — _evaluate_1h_signals (on hour crossing):
  Same adoption check + strategy filtering + per-strategy params as run_cycle
  Confluence relaxed: if only 1 strategy enabled, skip confluence gate
  If 2+ enabled, confluence still applies

Confluence (live only):
  If 2+ enabled strategies agree on direction → confluence fires
  Trade uses winning signal's strategy params
  Not tested during walk-forward (each strategy optimized in isolation)

Stop Monitoring (every 5m candle):
  for all open positions:
    → check stops (unchanged, ALL symbols regardless of adoption)
```

## Edge Cases

1. **Zero adoptions across all symbols:** Keep previous universe, log warning. Don't go to zero trades.
2. **Symbol has zero adopted strategies:** Don't adopt that symbol.
3. **Symbol dropped mid-position:** Stop monitoring continues. Position will be closed by stops. No new entries.
4. **First boot:** Empty universe. No trades until walk-forward finishes (~8-10h).
5. **Walk-forward running:** Universe unchanged until new results arrive.
6. **API manual trigger:** Same flow as scheduled — results go through `universe.update()`.
7. **Strategy only available in certain regimes:** Normal — strategy is enabled but only runs when regime matches (router still gates by regime).
8. **All strategies adopted for a symbol:** All run, with per-strategy params.
9. **Single strategy enabled for symbol + 1h path:** Confluence gate relaxed — single signal is sufficient.
10. **Walk-forward takes longer than expected:** Weekly scheduler (168h) accommodates up to ~10h runs without overlap.

## Testing

- `tests/test_adopted_universe.py`:
  - Empty universe blocks all symbols
  - `update()` with mixed adopted/rejected strategy-symbol combos
  - Per-strategy-per-symbol params returned correctly
  - Symbol with zero adopted strategies not adopted
  - Regime params from best-performing strategy
  - Zero-adoption preserves previous universe
  - `adopted_symbols()` and `get_enabled_strategies()` return correct lists
  - Diff logging (added/removed)
- `tests/test_backtest_engine_target_strategy.py`:
  - `target_strategy="orderflow"` only produces orderflow trades
  - `target_strategy=None` produces trades from all strategies (unchanged behavior)
  - Confluence disabled when target_strategy is set
  - Regime thresholds read from strategy_params, not hardcoded
- `tests/test_walk_forward.py`:
  - Verify `CHAMPION_DEFAULTS` import from adopted_universe
  - `run_multi_per_strategy()` returns correct nested structure
  - `_should_adopt()` applied per strategy-symbol combo
  - `target_strategy` passed through to BacktestEngine
- Trading loop:
  - Both `run_cycle` and `_evaluate_1h_signals` skip non-adopted symbols
  - Only enabled strategies evaluated per symbol
  - Per-strategy params passed to `initial_stops()`, `check_fee_gate()`, `fixed_fractional_size()`, `detect_regime()`
  - Confluence relaxed when single strategy enabled
- API:
  - `GET /universe` returns adopted symbols with per-strategy params
  - `POST /walk-forward/run` uses universe.update() (not dead method)
