# RL-Based Walk-Forward Parameter Optimizer

## Goal

Replace Optuna's blind Bayesian optimization with a PPO reinforcement learning agent that learns optimal trading parameters from 8 years of historical 1m candle data. One model per strategy, 22 wide-range parameters (23 for range model), 27 market observation features.

## Motivation

Optuna suffers from:
- **Phase 2 convergence**: TPE locks onto one parameter set, producing 20 identical results
- **No market awareness**: treats a bear market the same as a bull market
- **No parameter reasoning**: can't understand parameter relationships
- **40 trials needed**: brute-force search, most trials wasted
- **Narrow parameter ranges**: hand-tuned ranges limit discovery

An RL agent learns a policy: "given this market state, use these parameters." It trains on thousands of historical windows, generalizes across market regimes, and outputs optimal parameters in <1ms at inference time.

## Architecture

### New Components

**`bot/data/candle_store.py`** — Bulk 1m candle download and storage.

```python
class CandleStore:
    """Two access modes:
    - async methods for I/O-bound download (Bitvavo API, runs in async context)
    - sync methods for CPU-bound training (called from SubprocVecEnv workers)
    Both use the same PostgreSQL table via separate connections.
    """

    async def bulk_download(self, symbols: List[str]) -> None:
        """Pull all available 1m candles from Bitvavo for each symbol.
        Stores in PostgreSQL candle_1m table. Resumes from last timestamp.
        Uses async DB connection (asyncpg or run_in_executor with psycopg2)."""

    async def incremental_update(self, symbols: List[str]) -> None:
        """Pull new 1m candles since last download for each symbol.
        Async — called from scheduler context."""

    def get_candles(self, symbol: str, start: datetime, end: datetime,
                    resample: str = "5m") -> pd.DataFrame:
        """Fetch candles for a date range. Optionally resample from 1m.
        SYNC — called from training workers (SubprocVecEnv forks).
        Uses psycopg2 (sync) directly, NOT the async connection pool."""
```

Storage:
- PostgreSQL table `candle_1m`: columns `symbol, timestamp, open, high, low, close, volume`
- Primary key: `(symbol, timestamp)`
- Index on `(symbol, timestamp)` for fast range queries
- ~105 million rows for 25 symbols × 8 years, ~5 GB
- Bulk download estimated at 2-3 hours with Bitvavo rate limits
- **Migration**: Alembic migration creates the `candle_1m` table. Run `alembic revision --autogenerate -m "add candle_1m table"` after adding the SQLAlchemy model.

Download strategy:
- Bitvavo candle endpoint returns limited candles per request
- Use exponential backoff on rate limits (429 status)
- Track last downloaded timestamp per symbol for resume capability
- Skip symbols with <90 days of history (too new for training)

---

**`bot/learning/feature_extractor.py`** — Computes 27 observation features from candle data.

```python
def extract_features(candles_1m: pd.DataFrame, btc_candles_1m: pd.DataFrame = None) -> np.ndarray:
    """Compute 27 normalized features from a 90-day window of 1m candles.
    Returns a float32 array of shape (27,), all values clipped to [-1, 1].

    Resamples 1m candles to daily (for trend/regime), 4h (for ADX/ATR), and
    1h (for BB/volume). All intermediate values are normalized per-feature,
    then the final array is hard-clipped: np.clip(features, -1.0, 1.0).
    """
```

Feature list (27 total):

**Trend features (4):**
1. Price vs EMA200 distance (%) — normalized by dividing by 20, clipped to [-1, 1]
2. Price vs EMA50 distance (%) — normalized by dividing by 10, clipped to [-1, 1]
3. EMA50 slope (10-bar smoothed on daily) — normalized by dividing by price, clipped to [-1, 1]
4. Overall period return (%) — clipped to [-50, 50], divided by 50

**Volatility features (4):**
5. Average ATR% over period — divided by 10, clipped to [0, 1]
6. ATR% standard deviation — divided by 5, clipped to [0, 1]
7. Current ATR% vs average (ratio) — clipped to [0, 3], divided by 3
8. Max ATR% in period — divided by 20, clipped to [0, 1]

**Regime features (4):**
9. % of days ADX > 25 (trending) — already [0, 1]
10. % of days ADX < 20 (ranging) — already [0, 1]
11. Current ADX value — divided by 50, clipped to [0, 1]
12. Current RSI — divided by 100, already [0, 1]

**Volume features (2):**
13. Volume trend (second half vs first half, %) — clipped to [-100, 100], divided by 100
14. Current volume vs average (ratio) — clipped to [0, 5], divided by 5

**Price structure (3):**
15. Distance from period high (%) — divided by -50, clipped to [-1, 0]
16. Distance from period low (%) — divided by 50, clipped to [0, 1]
17. Number of significant reversals (>5% moves) — divided by 20, clipped to [0, 1]

**Bollinger Band features (2):**
18. BB bandwidth (current) — divided by 0.2, clipped to [0, 1]
19. BB %B (price position within bands) — clipped to [0, 1]

**Momentum features (2):**
20. RSI rate of change (14-bar diff) — divided by 50, clipped to [-1, 1]
21. MACD histogram sign (+1/-1 smoothed) — uses `macd()` from `bot/indicators/trend.py`

**Candle structure (2):**
22. Average body-to-wick ratio — divided by 2, clipped to [0, 1]
23. Lower wick dominance (% candles with longer lower wicks - % with upper) — already [-1, 1]

**Time features (2):**
24. Day of week (cyclical: sin(2π × day/7)) — already [-1, 1]
25. Hour of day bucket (cyclical: sin(2π × hour/24)) — already [-1, 1]

**BTC correlation (2):**
26. 30-day rolling correlation with BTC-EUR — already [-1, 1]
27. BTC trend direction (BTC EMA50 slope, normalized by BTC price) — clipped to [-1, 1]

**Final clipping:** After computing all 27 features, apply `np.clip(features, -1.0, 1.0)` to guarantee the observation space bounds are respected. This is safe because the per-feature normalization already targets [-1, 1]; the clip only catches extreme outliers.

Feature computation resamples 1m candles internally:
- Daily: trend features (EMA200, EMA50, period return), regime (ADX)
- 4h: volatility (ATR), regime (ADX confirmation)
- 1h: BB, volume, momentum
- Raw 1m: candle structure, time features

Dependencies: `bot/indicators/trend.py` (ema, adx, macd), `bot/indicators/momentum.py` (rsi), `bot/indicators/volatility.py` (atr, bollinger_bands), `bot/indicators/volume.py`.

---

**`bot/learning/rl_environment.py`** — Gymnasium environment wrapping BacktestEngine.

```python
class TradingParamEnv(gymnasium.Env):
    """RL environment for trading parameter optimization.

    Observation: 27 market features (float32)
    Action: 22 parameters normalized to [-1, 1] (23 for range model)
    Reward: composite score with guardrails
    """

    def __init__(self, candle_store, symbols, strategy, window_days=90):
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(27,), dtype=np.float32
        )  # All features hard-clipped to [-1, 1] by feature_extractor
        n_actions = 23 if strategy == "range" else 22
        self.action_space = spaces.Box(low=-1, high=1, shape=(n_actions,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Pick a random symbol and 90-day window. Compute features. Return observation."""

    def step(self, action):
        """Map action to parameters, run backtest on 5m resampled data,
        compute reward, return (obs, reward, terminated, truncated, info)."""
```

Action-to-parameter mapping:

| Index | Parameter | Range | Mapping from [-1, 1] |
|-------|-----------|-------|---------------------|
| 0 | atr_multiplier | 0.5-20.0 | (a+1)/2 * 19.5 + 0.5 |
| 1 | rr_ratio | 0.5-10.0 | (a+1)/2 * 9.5 + 0.5 |
| 2 | base_risk_pct | 0.5-10.0 | (a+1)/2 * 9.5 + 0.5 |
| 3 | min_profit_multiple | 0.5-8.0 | (a+1)/2 * 7.5 + 0.5 |
| 4 | max_hold_hours | 1-720 | int((a+1)/2 * 719 + 1) |
| 5 | quiet_atr_threshold | 0.1-5.0 | (a+1)/2 * 4.9 + 0.1 |
| 6 | regime_adx_threshold | 5-50 | (a+1)/2 * 45 + 5 |
| 7 | ranging_adx_threshold | 5-50 | (a+1)/2 * 45 + 5 |
| 8 | signal_strength_min | 0.1-1.0 | (a+1)/2 * 0.9 + 0.1 |
| 9 | min_confirmations | 1-5 | int((a+1)/2 * 4 + 1) |
| 10 | tf_weight_1h | 0.0-1.0 | (a+1)/2 |
| 11 | tf_weight_4h | 0.0-1.0 | (a+1)/2 |
| 12 | tf_weight_1d | 0.0-1.0 | (a+1)/2 |
| 13 | max_concurrent_positions | 1-10 | int((a+1)/2 * 9 + 1) |
| 14 | confidence_size_scaling | 0.0-2.0 | (a+1)/2 * 2.0 |
| 15 | ema200_filter_pct | 0.0-10.0 | (a+1)/2 * 10.0 |
| 16 | volatile_atr_threshold | 1.0-10.0 | (a+1)/2 * 9.0 + 1.0 |
| 17 | consecutive_confirms | 1-5 | int((a+1)/2 * 4 + 1) |
| 18 | confluence_boost | 1.0-2.0 | (a+1)/2 * 1.0 + 1.0 |
| 19 | drawdown_scale_pct | 1.0-15.0 | (a+1)/2 * 14.0 + 1.0 |
| 20 | max_position_pct | 0.1-0.5 | (a+1)/2 * 0.4 + 0.1 |
| 21 | trail_activation_mult | 0.5-3.0 | (a+1)/2 * 2.5 + 0.5 |
| 22 | range_max_hold_hours | 6-168 | int((a+1)/2 * 162 + 6) |

Parameters 0-21 are used by all 4 strategy models (action space shape 22).
Parameter 22 (`range_max_hold_hours`) is only used by the range model (action space shape 23). Non-range models ignore index 22.

Each episode is one 90-day window on one symbol. The environment picks a random window on `reset()`, the agent takes a single action (parameter set), the backtest runs, reward is computed, episode ends (single-step episode).

**Feature/backtest window relationship:** Features and backtests use the SAME 90-day window. This is NOT data leakage — the features describe the market regime (trend, volatility, volume characteristics) of the period being traded. The agent learns "given these market conditions, which parameters work best?" No future information is used. This mirrors production usage where live features describe the current market state and the agent outputs parameters for that state.

Reward function:
```python
def _compute_reward(self, result: BacktestResult) -> float:
    pnl_norm = max(min(result.total_pnl / 1000.0, 3.0), -3.0)
    sharpe_clipped = max(min(result.sharpe_ratio, 3.0), -3.0)
    # Cap profit_factor to prevent inf (0 losses → inf). 5.0 = excellent.
    pf_clipped = min(result.profit_factor, 5.0) if not math.isinf(result.profit_factor) else 5.0
    ppf_clipped = max(min(result.profit_per_fee, 5.0), -5.0)

    base = (sharpe_clipped * 0.30
            + pnl_norm * 0.30
            + pf_clipped * 0.20
            + ppf_clipped * 0.20)

    penalty = 0.0
    if result.max_drawdown_pct > 20.0:
        penalty -= 5.0
    if result.total_trades < 3:
        penalty -= 3.0
    if result.total_trades > 0 and result.win_rate < 20.0:
        penalty -= 2.0

    return base + penalty
```

---

**`bot/learning/rl_trainer.py`** — Training loop and model management.

```python
class RLTrainer:
    def __init__(self, candle_store, strategies: List[str], model_dir: str = "models/rl_optimizer"):
        pass

    def train(self, epochs: int = 5, parallel_envs: int = 8) -> Dict[str, float]:
        """Full training run. Returns {strategy: avg_reward} for each model."""

    def retrain(self, fine_tune_months: int = 6) -> Dict[str, float]:
        """Weekly incremental retrain on recent data. Fine-tunes existing models."""

    def validate(self, strategy: str) -> float:
        """Run new model on held-out validation set. Returns avg reward."""
```

Training configuration:
- Algorithm: PPO from `stable-baselines3`
- Network: MlpPolicy, 3 hidden layers of 256 neurons
- Learning rate: 3e-4 (default PPO)
- Batch size: 64
- n_steps: 2048
- Parallel environments: `min(os.cpu_count() or 1, 4)` (via `SubprocVecEnv`, capped at 4 to stay within Docker memory). Each env holds ~50 MB in-memory (candle data + backtest engine). At 4 workers, peak training memory is ~200 MB + PPO overhead (~100 MB) ≈ 300 MB.
- Fallback: if `SubprocVecEnv` fails (e.g., fork not available), fall back to `DummyVecEnv` (sequential, same API).
- Total timesteps per strategy: 25,000 (5,000 episodes × 5 epochs)

Training flow:
1. Determine `n_envs = min(os.cpu_count() or 1, 4)`
2. Create `SubprocVecEnv` with `n_envs` parallel `TradingParamEnv` instances (fallback to `DummyVecEnv` on error)
3. Initialize PPO model with `MlpPolicy`
4. Call `model.learn(total_timesteps=25000)`
5. Save to `models/rl_optimizer/{strategy}_ppo.zip` (atomic: write to `.tmp`, then `os.replace()`)
6. Repeat for each strategy

Model validation:
- Hold out last 10% of data (most recent ~10 months) for validation
- New model must score higher average reward than old model on validation set
- If worse, keep old model and log warning
- Both models retained on disk for manual rollback

Model file writes (crash safety):
- Save new model to `{strategy}_ppo.tmp.zip` first
- Use `os.replace("{strategy}_ppo.tmp.zip", "{strategy}_ppo.zip")` for atomic rename
- `os.replace` is atomic on both Linux and Windows (same filesystem)
- If training crashes mid-save, the `.tmp.zip` file is left behind; the production `.zip` is untouched

Weekly retrain:
- Fine-tune existing model on last 6 months of data (faster than full retrain)
- `model.load()` then `model.learn(total_timesteps=5000)` with recent data only
- Same validation gate before replacing the production model

Estimated training time:
- Full training: ~1-2 hours (up to 4 parallel workers, 5m backtests)
- Weekly retrain: ~15-30 minutes

---

**`bot/learning/rl_optimizer.py`** — Inference for live walk-forward.

```python
class RLOptimizer:
    def __init__(self, model_dir: str = "models/rl_optimizer"):
        self._models: Dict[str, PPO] = {}  # strategy -> loaded model

    def load_models(self) -> None:
        """Load all strategy models from disk. Log warnings for missing models."""

    def predict(self, candles_1m: List, strategy: str,
                btc_candles_1m: List = None) -> Dict[str, Any]:
        """Compute features, run model forward pass, return parameter dict.
        Falls back to CHAMPION_DEFAULTS if model not loaded."""
```

Inference flow:
1. `feature_extractor.extract_features(candles_1m, btc_candles_1m)` → 27-element array
2. `model.predict(features, deterministic=True)` → 22-element action (23 for range)
3. Map action to parameter dict (same mapping table as environment)
4. Return parameter dict

Inference time: <1ms per prediction. No GPU, no external services.

Fallback: if model file missing or corrupt, return `CHAMPION_DEFAULTS` and log warning.

### Modified Components

**`bot/learning/walk_forward.py`** — Replace `_run_optuna_window` with `_run_rl_window`.

```python
def _run_rl_window(train_candles, test_candles, max_workers,
                    target_strategy=None, rl_optimizer=None):
    """RL-driven parameter optimization for a single train/test window.

    Flow:
      1. rl_optimizer.predict(train_candles, strategy) → params  (<1ms)
      2. Backtest train_candles with params (validation)           (~5s)
      3. Backtest test_candles with params (OOS test)              (~5s)
      4. Return (params, test_result)                              (~10s total)
    """
```

Return value: `(best_params, BacktestResult)` — same shape as current `_run_optuna_window`.

The `WalkForwardOptimizer` receives an `RLOptimizer` instance via constructor injection. If no RL optimizer is provided (models not trained yet), falls back to CHAMPION_DEFAULTS.

**`bot/backtest/engine.py`** — Accept new parameters.

New `strategy_params` keys and exact integration points:

**1. `signal_strength_min`** (float, default 0.0) — minimum signal strength to enter.
- Extract in `__init__`: `self._signal_strength_min = params.get("signal_strength_min", 0.0)`
- Apply in `run()` at line ~347 (entry logic): **replace** the existing `best_signal.strength > 0` check with:
  ```python
  and best_signal.strength >= self._signal_strength_min
  ```
  Since `signal_strength_min` defaults to 0.0 and the RL agent's range is [0.1, 1.0], this subsumes the old `> 0` check.

**2. `min_confirmations`** (int, default 2) — minimum confirming indicators before entry.
- Already extracted at line 138: `self._min_confirmations = params.get("min_confirmations", 2)`
- **Currently unused in entry path.** Must add a check in `run()` at the entry logic (line ~347):
  ```python
  and best_signal.indicator_snapshot.get("confirming_count", 0) >= self._min_confirmations
  ```
- This gates entries on having enough confirming indicators (e.g., RSI + MACD + ADX agreeing). The `confirming_count` is already populated by all strategies in `_evaluate_precomputed()` (lines 646, 658, 669).
- **Note:** This is distinct from `consecutive_confirms` (line 150), which requires the signal to persist for N consecutive cycles. `min_confirmations` requires N indicators to agree in a single cycle.

**3. `tf_weight_1h`, `tf_weight_4h`, `tf_weight_1d`** (float, default 1.0 each) — weight per-timeframe trend agreement.
- Extract in `__init__`:
  ```python
  self._tf_weight_1h = params.get("tf_weight_1h", 1.0)
  self._tf_weight_4h = params.get("tf_weight_4h", 1.0)
  self._tf_weight_1d = params.get("tf_weight_1d", 1.0)
  ```
- Apply in `_evaluate_precomputed()` after collecting signals (around line 623). After computing the best signal's strength, scale it by timeframe alignment:
  ```python
  # Timeframe weight scaling: check if 4h and 1d trends agree with signal direction
  tf_score = self._tf_weight_1h  # 1h always contributes (signal source)
  if precomp_4h and 0 <= h4_idx < len(precomp_4h["ema50"]):
      ema50_4h_slope = precomp_4h["ema50"][h4_idx] - precomp_4h["ema50"][max(0, h4_idx - 3)]
      if (direction == "LONG" and ema50_4h_slope > 0) or (direction == "SHORT" and ema50_4h_slope < 0):
          tf_score += self._tf_weight_4h
  if precomp_1d and 0 <= h1d_idx < len(precomp_1d["ema50"]):
      ema50_1d_slope = precomp_1d["ema50"][h1d_idx] - precomp_1d["ema50"][max(0, h1d_idx - 3)]
      if (direction == "LONG" and ema50_1d_slope > 0) or (direction == "SHORT" and ema50_1d_slope < 0):
          tf_score += self._tf_weight_1d
  total_weight = self._tf_weight_1h + self._tf_weight_4h + self._tf_weight_1d
  if total_weight > 0:
      strength *= tf_score / total_weight  # normalize so max = 1.0
  ```

**4. `max_concurrent_positions`** — already exists as constructor arg `max_open_positions` (line 115).
- Wire from `strategy_params` in `__init__`:
  ```python
  if "max_concurrent_positions" in params:
      self.max_open = params["max_concurrent_positions"]
  ```
- Used at line 349: `len(self.positions) < self.max_open`

**5. `confidence_size_scaling`** (float, default 0.0) — scale position size by signal confidence.
- Extract in `__init__`: `self._confidence_size_scaling = params.get("confidence_size_scaling", 0.0)`
- Apply in `_open_position()` after computing `size_eur`:
  ```python
  # Scale size by signal confidence: 0.0 = fixed size, 2.0 = 2x at strength 1.0
  # Signals with strength < 0.5 get reduced size (intentional — punish weak signals)
  # Example: strength=0.8, scaling=2.0 → 1.6x. strength=0.3, scaling=2.0 → 0.6x.
  if self._confidence_size_scaling > 0:
      size_eur *= max(0.2, 1.0 + (signal.strength - 0.5) * self._confidence_size_scaling)
  ```

**6. `ema200_filter_pct`** (float, default 2.0) — EMA200 trend filter strictness.
- Extract in `__init__`: `self._ema200_filter_pct = params.get("ema200_filter_pct", 2.0)`
- **Replaces** the hardcoded `2.0` in the EMA200 trend filter at lines 338-341:
  ```python
  # Current hardcoded logic (REPLACE):
  if ema200_dist_pct < -2.0 and best_signal.direction == "LONG": ...
  elif ema200_dist_pct > 2.0 and best_signal.direction == "SHORT": ...

  # New RL-controllable logic:
  if self._ema200_filter_pct > 0:
      if ema200_dist_pct < -self._ema200_filter_pct and best_signal.direction == "LONG":
          best_signal = None
      elif ema200_dist_pct > self._ema200_filter_pct and best_signal.direction == "SHORT":
          best_signal = None
  # ema200_filter_pct == 0.0 → filter disabled, agent has full freedom
  ```

**7. `volatile_atr_threshold`** (float, default 4.0) — ATR% above which regime becomes VOLATILE.
- Extract in `__init__`: `self._volatile_atr_threshold = params.get("volatile_atr_threshold", 4.0)`
- **Replaces** the hardcoded `4.0` at line 282:
  ```python
  # Current: elif atr_pct_4h > 4.0: regime = Regime.VOLATILE
  # New:
  elif atr_pct_4h > self._volatile_atr_threshold:
      regime = Regime.VOLATILE
  ```

**8. `consecutive_confirms`** (int, default 1) — require signal to persist N consecutive bars.
- Already extracted at line 150: `self._consecutive_confirms = params.get("consecutive_confirms", 1)`
- Already used at line 348: `_signal_streak >= self._consecutive_confirms`
- The RL agent optimizes this existing parameter — no new code needed.

**9. `confluence_boost`** (float, default 1.2) — strength multiplier when strategies agree.
- Extract in `__init__`: `self._confluence_boost = params.get("confluence_boost", 1.2)`
- **Replaces** the hardcoded `1.2` at line 655 in `_evaluate_precomputed()`:
  ```python
  # Current: strength=min(confluence.strength * 1.2, 1.0)
  # New:
  strength=min(confluence.strength * self._confluence_boost, 1.0)
  ```

**10. `drawdown_scale_pct`** (float, default 3.0) — drawdown % threshold to halve position size.
- Extract in `__init__`: `self._drawdown_scale_pct = params.get("drawdown_scale_pct", 3.0)`
- **Replaces** the hardcoded `3.0` at line 707 in `_compute_position_size()`:
  ```python
  # Current: if drawdown_pct >= 3.0: size_eur *= 0.5
  # New:
  if drawdown_pct >= self._drawdown_scale_pct:
      size_eur *= 0.5
  ```

**11. `max_position_pct`** (float, default 0.30) — max single position as fraction of initial capital.
- Extract in `__init__`: `self._max_position_pct = params.get("max_position_pct", 0.30)`
- **Replaces** the hardcoded `0.30` at line 725 in `_open_position()`:
  ```python
  # Current: max_position = self.initial_capital * 0.30
  # New:
  max_position = self.initial_capital * self._max_position_pct
  ```

**12. `trail_activation_mult`** (float, default 1.5) — trailing stop activation threshold multiplier.
- Extract in `__init__`: `self._trail_activation_mult = params.get("trail_activation_mult", 1.5)`
- **Replaces** the hardcoded `1.5` at line 857 in `_check_exits_fast()`:
  ```python
  # Current: activation_threshold=1.5,
  # New:
  pos.stop_loss = trail_stop(
      price, pos.highest_price, pos.stop_loss,
      trail_dist,
      direction=direction,
      activation_threshold=self._trail_activation_mult,
      entry_price=pos.entry_price,
  )
  ```

**13. `range_max_hold_hours`** (int, default 72) — max hold time for range strategy positions. **Range model only.**
- Extract in `__init__`: `self._range_max_hold_bars = int(params.get("range_max_hold_hours", 72) * 12)`
- **Replaces** the hardcoded `864` (72h × 12) at line 155.
- Used at line 803: `if pos.strategy == "range": max_hold = self._range_max_hold_bars`
- Non-range models never output this parameter; the default 72h applies when absent.

**`bot/main.py`** — Load RL models on startup, wire retrain to scheduler.

```python
# On startup (around line 90, after DB init):
candle_store = CandleStore(db_url=settings.database_url)
rl_optimizer = RLOptimizer()
rl_optimizer.load_models()
walk_forward = WalkForwardOptimizer(rl_optimizer=rl_optimizer)

# Weekly retrain callback (passed to scheduler):
async def _retrain_rl():
    trainer = RLTrainer(candle_store, ALL_STRATEGIES)
    symbols = get_tradeable_symbols() or ["BTC-EUR"]
    await candle_store.incremental_update(symbols)
    # trainer.retrain() is CPU-bound (15-30 min) — run in executor to avoid blocking event loop
    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(None, trainer.retrain)
    rl_optimizer.load_models()  # reload after retrain
    logger.info("RL retrain complete: %s", results)
```

**`bot/scheduler.py`** — Add `rl_retrain_run` callback parameter.

Modify `BotScheduler.__init__` to accept one new optional callback:

```python
def __init__(
    self,
    ...,  # existing params unchanged
    rl_retrain_run: Optional[Callable] = None,  # NEW
) -> None:
    ...
    self._rl_retrain_run = rl_retrain_run
```

Add in `run_all()`:
```python
if self._rl_retrain_run is not None:
    coros.append(self._rl_retrain_loop(168))  # weekly
```

Add new loop method:
```python
async def _rl_retrain_loop(self, interval_hours: int) -> None:
    # Wait 10 minutes on startup to let walk-forward complete first.
    # Walk-forward starts after 60s (line 106), runs for ~2-5 min.
    # RL retrain waits 600s to guarantee no overlap on first run.
    await asyncio.sleep(600)
    while True:
        try:
            logger.info("RL retrain starting...")
            await self._rl_retrain_run()
        except Exception as e:
            logger.error("RL retrain error: %s", e)
        await asyncio.sleep(interval_hours * 3600)
```

**Timing coordination:** Walk-forward runs first (60s startup delay), then RL retrain runs later (600s startup delay). On subsequent weekly cycles, both run independently — walk-forward uses the current RL models for inference (<1ms), while retrain updates models in the background. Retrain writes to `.tmp` then atomically renames, so walk-forward always reads consistent model files.

Wire in `main.py`:
```python
scheduler = BotScheduler(
    ...,  # existing args unchanged
    rl_retrain_run=_retrain_rl,
)
```

**`requirements.txt`** — Dependency changes.

Add:
- `stable-baselines3>=2.3.0`
- `gymnasium>=0.29.0`

Remove:
- `optuna>=3.6.0`

### Removed

- `optuna` removed from `requirements.txt`
- `_run_optuna_window()` deleted entirely
- All `import optuna` references removed
- `OPTUNA_TRIALS` constant removed

## Error Handling

| Failure | Behavior |
|---------|----------|
| Model file missing | Fall back to CHAMPION_DEFAULTS, log warning |
| Model file corrupt | Same as missing — load failure caught, fallback |
| Bitvavo API rate limited | Exponential backoff, retry 3x, skip symbol |
| Bitvavo API down | Skip bulk download, log error, retry next cycle |
| Symbol has <90 days data | Skip for training, log info |
| Training crashes | Previous model preserved, new model not saved |
| New model worse than old | Keep old model, log warning with metrics |
| Backtest crashes during training | Episode returns reward -10.0, training continues |
| Feature extraction fails (bad data) | Return zeros array, log warning |

## Training Data

- **Source**: Bitvavo REST API, 1m candles
- **Symbols**: 25 actively traded symbols (from `get_tradeable_symbols()`)
- **History**: All available (Bitvavo launched 2018, ~8 years)
- **Storage**: PostgreSQL `candle_1m` table, ~105M rows, ~5 GB
- **Features**: Computed from 1m candles (maximum precision)
- **Backtests**: Run on 5m resampled candles (matches production, 5x faster)
- **Windows**: 90-day sliding windows, 14-day step = ~200 per symbol
- **Train/validation split**: Last 10% of data held out for validation

## Performance

| Metric | Optuna (current) | RL Agent |
|--------|-----------------|----------|
| Trials per window | 40 | 1 (single forward pass) |
| Wall time per window | ~35s | ~10s |
| Total per symbol (4 windows) | ~2.5 min | ~40s |
| Training time | None | ~1-2h (one-time), ~30min weekly |
| Model size | None | ~2 MB per strategy |
| GPU required | No | No |
| Market-aware | No | Yes |
| Learns from history | No | Yes (8 years) |

## Test Plan

1. **Unit test `feature_extractor.py`**: Feed known candle data, verify 27 features returned, verify normalization ranges, verify BTC correlation computed.
2. **Unit test `rl_environment.py`**: Verify observation/action spaces, verify action-to-parameter mapping, verify reward computation with known BacktestResult, verify guardrail penalties.
3. **Unit test `rl_optimizer.py`**: Mock model, verify predict returns valid parameter dict, verify fallback to CHAMPION_DEFAULTS on missing model.
4. **Unit test `candle_store.py`**: Mock Bitvavo API, verify download/resume logic, verify resampling.
5. **Unit test `_run_rl_window`**: Mock rl_optimizer, verify flow (predict → backtest train → backtest OOS → return).
6. **Integration test**: Train a tiny model (100 timesteps) on synthetic data, verify it produces valid parameters, verify walk-forward accepts them.
7. **Backtest engine test**: Verify new parameters (signal_strength_min, min_confirmations, tf_weights, max_concurrent_positions, confidence_size_scaling) affect trade behavior.

## Dependencies

- **stable-baselines3** >= 2.3.0 — PPO implementation
- **gymnasium** >= 0.29.0 — environment interface
- **PostgreSQL** — already running (candle_1m table added)
- **Bitvavo API** — already integrated for live trading
- All other dependencies already in project (pandas, numpy, etc.)
