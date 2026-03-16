# LLM-Based Walk-Forward Parameter Optimizer

## Goal

Replace Optuna's blind Bayesian optimization with a local LLM agent (Ollama + Llama 3 8B) that reasons about market conditions and backtest results to propose better trading parameters in fewer trials.

## Motivation

Optuna suffers from:
- **Phase 2 convergence**: TPE locks onto one parameter set, producing 20 identical results
- **No market awareness**: treats a bear market the same as a bull market
- **No parameter reasoning**: can't understand that high ATR multiplier + low RR ratio = bad combo
- **40 trials needed**: brute-force search, most trials are wasted

An LLM agent can analyze market context, reason about parameter relationships, and converge in 12-18 targeted trials instead of 40 blind ones.

## Architecture

### New Components

**`bot/learning/llm_optimizer.py`** — Ollama client, prompt builder, response parser.

Public interface:
```python
class OllamaUnavailableError(Exception):
    """Raised when Ollama is not reachable or model not found."""

def check_ollama() -> None:
    """Health check — GET /api/tags. Raises OllamaUnavailableError if unreachable."""

def propose_params(
    market_summary: str,
    champion_defaults: Dict,
    target_strategy: str | None,
    prev_windows: List[Dict] | None,
) -> List[Dict[str, Any]]:
    """Round 1: propose 6 parameter sets based on market context."""

def refine_params(
    market_summary: str,
    results: List[Dict],
    champion_defaults: Dict,
    target_strategy: str | None,
) -> List[Dict[str, Any]]:
    """Round 2+: refine params based on backtest results."""
```

Configuration:
- Ollama endpoint: `http://localhost:11434` (configurable via `OLLAMA_URL` env var)
- Model: `llama3` (configurable via `OLLAMA_MODEL` env var)
- Quantization: Q4_K_M recommended (fits in ~4.5 GB VRAM on RTX 2060 Super 8 GB)
- Timeout: 120s per call
- Temperature: 0.7 (enough creativity to explore, not so high it hallucinates)
- `num_predict`: 1024 (sufficient for JSON array of 6 param sets)

Response parsing:
- Extract JSON array from response (handle markdown code fences via regex `\[.*\]` with `re.DOTALL`)
- Validate all 8 keys present, values within allowed ranges
- Clamp out-of-range values silently; round to nearest step (0.5 for atr/rr/risk, 24 for hold hours, 0.1 for thresholds, 2 for ADX thresholds)
- If parsing fails, retry once with "Respond with valid JSON only" appended
- If retry also fails, return `[CHAMPION_DEFAULTS]` as fallback
- Log all prompts and raw responses at DEBUG level for debugging prompt quality

---

**`bot/learning/market_summarizer.py`** — Computes a text summary of training candle data.

Public interface:
```python
def summarize(candles: List, target_strategy: str | None = None) -> str
```

Output contains three sections:

**Section 1: Computed stats**
```
Period: 90 days (Oct 21 - Jan 19)
Trend: -12.3% decline, price below EMA200 since day 34
Volatility: avg ATR% 2.1 (range 0.8-4.7)
Regimes: TRENDING 35%, RANGING 28%, NEUTRAL 22%, QUIET 10%, VOLATILE 5%
Price: high 73400 / low 58200 / current 61800 (16% below high)
Volume: declining trend (-8% over period)
```

**Section 2: Daily candle table** (full training period, ~90 rows)
```
date,open,high,low,close,volume,atr_pct
2025-10-21,67200,68100,66800,67500,12400000,1.9
...
```

Resampled from 5m data. ATR% appended per row.

**Section 3: Recent 4h candles** (last 7 days, ~42 rows)
```
date,open,high,low,close,volume,atr_pct,rsi,adx
2026-01-12 00:00,62100,62800,61500,62400,3200000,2.1,45,22
...
```

Includes RSI and ADX columns for regime context.

### Context Budget

Llama 3 8B has an 8,192-token context window. Budget breakdown:

| Component | Estimated tokens |
|-----------|-----------------|
| System prompt + parameter ranges + strategy description | ~400 |
| Section 1: Computed stats | ~80 |
| Section 2: Daily candles (90 rows CSV) | ~2,700 |
| Section 3: 4h candles (42 rows CSV) | ~1,400 |
| Previous window results (if available) | ~200 |
| Round 2 results table (if refine) | ~300 |
| **Total input** | **~5,100** |
| Reserved for output (6 JSON objects) | ~800 |
| **Safety margin** | **~2,300** |

**Token estimation:** Use `len(prompt) // 3` as a conservative character-to-token estimate. CSV data with numbers tokenizes less efficiently than English prose (~3 chars/token vs ~4), so the conservative divisor prevents underestimation.

**Fallback if over budget:** If the estimated token count exceeds 6,000 tokens, truncate daily candles to the most recent 60 days. This is expected to trigger for some symbols with longer price strings. The truncated version still provides sufficient trend context while fitting comfortably within the 8K window.

### Modified Component

**`bot/learning/walk_forward.py`** — Replace `_run_optuna_window()` with `_run_llm_window()`.

```python
def _run_llm_window(train_candles, test_candles, max_workers,
                     target_strategy=None, prev_windows=None):
    """LLM-driven parameter optimization for a single train/test window.

    Flow:
      Round 1: LLM proposes 6 param sets from market context + defaults
               → backtest all 6 in parallel
      Round 2: LLM sees results, proposes 6 refined param sets
               → backtest all 6 in parallel
      Round 3: (conditional) if best_score(round 2) < best_score(round 1),
               LLM gets one more round with combined results
               → backtest 6 more in parallel
      Pick overall best by scoring function → test on OOS data → return
    """
```

**Scoring function** (same weights as current Optuna scorer, for consistency):

```python
def _score_result(pnl: float, sharpe: float, pf: float, ppf: float) -> float:
    """Score a backtest result. Higher is better."""
    if pnl == 0.0 and sharpe == 0.0:
        return -10.0  # Dead-zone penalty
    pnl_norm = max(min(pnl / 1000.0, 3.0), -3.0)
    return sharpe * 0.30 + pnl_norm * 0.30 + pf * 0.20 + ppf * 0.20
```

**Round 3 trigger**: Round 3 fires when `max(round_2_scores) < max(round_1_scores)`. This means the LLM's refinement went in the wrong direction. Round 3 reuses the Round 2 (refine) prompt template but with a combined results table from both rounds (12 rows total, labeled "ROUND 1" and "ROUND 2" in a header column). If round 2 improved, we stop at 12 total trials.

**Return value**: `_run_llm_window` returns `(best_params, BacktestResult)` on success — same shape as the current `_run_optuna_window`. The OOS `BacktestResult` is used by the async caller to read `.sharpe_ratio`, `.total_pnl`, and `.profit_per_fee`. On Ollama failure, returns `(None, None)` as sentinel.

**`_run_single_backtest` return value** — expanded to provide more data for the LLM:

```python
def _run_single_backtest(candles, params, target_strategy=None):
    """Returns (params, pnl, sharpe, pf, ppf, winning_trades, losing_trades,
               avg_win_pct, avg_loss_pct, avg_hold_hours)."""
    engine = BacktestEngine(candles, strategy_params=params,
                            slippage_pct=0.001, target_strategy=target_strategy)
    result = engine.run()

    # Compute avg hold hours from trade_log
    avg_hold_hours = 0.0
    if result.trade_log:
        from datetime import datetime
        hold_hours = []
        for t in result.trade_log:
            try:
                entry = datetime.fromisoformat(t.entry_time)
                exit_ = datetime.fromisoformat(t.exit_time)
                hold_hours.append((exit_ - entry).total_seconds() / 3600)
            except (ValueError, TypeError):
                pass
        avg_hold_hours = sum(hold_hours) / len(hold_hours) if hold_hours else 0.0

    return (params, result.total_pnl, result.sharpe_ratio,
            result.profit_factor, result.profit_per_fee,
            result.winning_trades, result.losing_trades,
            result.avg_win_pct, result.avg_loss_pct,
            round(avg_hold_hours, 1))
```

**Ollama failure handling:**

Since `walk_forward.py` runs in an executor (sync context), it cannot `await bus.publish()`. Instead:

1. `_run_llm_window()` catches `OllamaUnavailableError` and **returns a sentinel** — a tuple `(None, None)` instead of `(best_params, test_result)`.
2. The async caller (`_execute_from_candles` / `_execute`) checks for the sentinel.
3. The async caller publishes `risk.halt` via the EventBus and sets `self._ollama_down = True`.
4. `WalkForwardOptimizer` gains a `_last_ollama_warning: float = 0` timestamp field.
5. The scheduler's periodic call checks `_ollama_down` and re-publishes the warning every 30 minutes.

```python
# In _execute_from_candles (async context):
best_params, test_result = await loop.run_in_executor(
    None, _run_llm_window, train_candles, test_candles, max_workers, target_strategy, prev_windows
)
if best_params is None:
    # Ollama is down — publish warning and skip
    if hasattr(self, '_event_bus') and self._event_bus:
        await self._event_bus.publish("risk.halt", {
            "reason": "Ollama unavailable — walk-forward skipped",
            "symbol": symbol,
        })
    self._ollama_down = True
    return WFResult()
```

**`prev_windows` format** — list of dicts from prior completed windows in this run:

```python
prev_windows = [
    {
        "window": 1,
        "best_params": {"atr_multiplier": 3.5, "rr_ratio": 2.5, ...},
        "pnl": 120.50,
        "sharpe": 0.82,
        "profit_factor": 1.35,
    },
    ...
]
```

Built up as each window completes. Passed to `propose_params()` so the LLM can see what worked in earlier windows.

### Removed

- `optuna` removed from `requirements.txt`
- `_run_optuna_window()` deleted entirely
- All `import optuna` references removed
- `OPTUNA_TRIALS` constant removed

## Prompt Design

### Round 1 Prompt (propose)

```
You are a quantitative trading parameter optimizer for cryptocurrency markets.

STRATEGY: {strategy_name}
{strategy_description}

MARKET CONDITIONS:
{market_summary_stats}

DAILY CANDLES:
{daily_candle_csv}

RECENT 4H CANDLES (last 7 days):
{4h_candle_csv}

PREVIOUS WINDOW RESULTS (if available):
{prev_windows_summary}

PARAMETER RANGES:
- atr_multiplier: 2.5-5.0 (stop-loss distance, ATR * multiplier)
- rr_ratio: 2.0-4.0 (take-profit distance, risk * ratio)
- base_risk_pct: 2.0-5.0 (position size as % of equity risked)
- min_profit_multiple: 2.0-4.0 (reject trades where profit < N * fees)
- max_hold_hours: 48-240 (maximum trade duration)
- quiet_atr_threshold: 0.8-1.5 (ATR% below this = skip, too quiet)
- regime_adx_threshold: 20-30 (ADX above this = trending market)
- ranging_adx_threshold: 15-25 (ADX below this = ranging market)

CURRENT DEFAULTS: {champion_defaults_json}

Based on the market conditions, suggest 6 parameter sets optimized for this
market environment. Consider:
- In downtrends, wider stops and longer hold times often help
- In ranging markets, tighter stops and shorter holds work better
- High volatility needs wider ATR multipliers
- Low volatility benefits from lower quiet_atr_threshold

Respond ONLY with a JSON array of 6 objects. No explanation.
```

### Round 2 Prompt (refine)

```
You are refining trading parameters based on backtest results.

STRATEGY: {strategy_name}
{strategy_description}

MARKET CONDITIONS:
{market_summary_stats}

PARAMETER RANGES:
- atr_multiplier: 2.5-5.0 (stop-loss distance, ATR * multiplier)
- rr_ratio: 2.0-4.0 (take-profit distance, risk * ratio)
- base_risk_pct: 2.0-5.0 (position size as % of equity risked)
- min_profit_multiple: 2.0-4.0 (reject trades where profit < N * fees)
- max_hold_hours: 48-240 (maximum trade duration)
- quiet_atr_threshold: 0.8-1.5 (ATR% below this = skip, too quiet)
- regime_adx_threshold: 20-30 (ADX above this = trending market)
- ranging_adx_threshold: 15-25 (ADX below this = ranging market)

CURRENT DEFAULTS: {champion_defaults_json}

BACKTEST RESULTS FROM ROUND {round_number - 1}:
{results_table}

Suggest 6 improved parameter sets. Keep what worked, adjust what didn't.
Respond ONLY with a JSON array of 6 objects. No explanation.
```

### Results Table Format (for round 2+ prompts)

```
#  atr  rr  risk  hold  pnl      sharpe  pf    trades  W/L    avg_win%  avg_loss%  avg_hold
1  3.5  2.0  3.0  120h  €120.50  0.82    1.35  14      8/6    1.20      -0.85      18h
2  5.0  4.0  4.0  240h  -€85.20  -0.31   0.78  3       1/2    2.10      -1.90      96h
...
```

Fields sourced from expanded `_run_single_backtest`:
- `avg_win%` / `avg_loss%` from `BacktestResult.avg_win_pct` / `avg_loss_pct`
- `avg_hold` computed from `TradeRecord.entry_time` / `exit_time` in `_run_single_backtest`

### Round 3 Combined Results Table (when round 3 triggers)

```
ROUND 1:
#  atr  rr  risk  hold  pnl      sharpe  pf    trades  W/L    avg_win%  avg_loss%  avg_hold
1  3.5  2.0  3.0  120h  €120.50  0.82    1.35  14      8/6    1.20      -0.85      18h
...

ROUND 2:
#  atr  rr  risk  hold  pnl      sharpe  pf    trades  W/L    avg_win%  avg_loss%  avg_hold
7  3.0  2.5  3.5  96h   €85.30   0.65    1.22  12      7/5    1.05      -0.72      14h
...
```

This gives the LLM visibility into both rounds so it can see what worked and what regressed.

## Strategy Descriptions

Embedded in the prompt based on `target_strategy`:

- **orderflow**: "Detects absorption patterns — high volume candles with small bodies and long wicks. LONG on buying absorption (lower wicks), SHORT on selling absorption (upper wicks). Confirmed by CMF and OBV divergence."
- **funding_contrarian**: "Contrarian strategy — goes LONG when RSI is oversold (<28) with MACD recovering, SHORT when RSI is overbought (>72) with MACD fading. Fades crowd extremes."
- **range**: "Range-bound trading — buys near lower Bollinger Band when RSI <35, sells near upper BB when RSI >65. Only active when ADX <22 (non-trending). Profits from oscillation within bands."
- **squeeze**: "Volatility squeeze breakout — detects BB compression then expansion with volume surge. Goes LONG on upward breakout (EMA50 slope positive), SHORT on downward breakout (EMA50 slope negative)."

## Error Handling

| Failure | Behavior |
|---------|----------|
| Ollama not reachable | `OllamaUnavailableError` → sync function returns sentinel `(None, None)` → async caller publishes `risk.halt`, Discord warning every 30 min |
| Model not pulled | Same as not reachable (Ollama returns 404) |
| LLM returns invalid JSON | Retry once with "Respond with valid JSON only" appended |
| LLM returns wrong param count | Use whatever valid params were parsed (min 1) |
| LLM returns out-of-range values | Clamp to valid range silently |
| All parsing fails (both attempts) | Fall back to `[CHAMPION_DEFAULTS]` (1 candidate) |
| Backtest crashes on a candidate | Skip that candidate, continue with others |
| All 6 backtests crash | Use CHAMPION_DEFAULTS for that round |

## Startup Health Check

On bot startup, `main.py` calls `check_ollama()` during initialization. If Ollama is not reachable:
- Log a warning: "Ollama not available — walk-forward optimization will be skipped until Ollama is running"
- Set `WalkForwardOptimizer._ollama_down = True`
- The bot continues running (Ollama is not required for trading)
- Walk-forward scheduler skips optimization while `_ollama_down` is True
- Each scheduler tick retries `check_ollama()` before skipping

## Docker Networking

The bot runs in Docker, Ollama runs on the host. For the bot container to reach Ollama:

**docker-compose.yml changes:**
```yaml
bot:
  # ... existing config ...
  extra_hosts:
    - "host.docker.internal:host-gateway"
  environment:
    OLLAMA_URL: http://host.docker.internal:11434
```

On Windows with Docker Desktop, `host.docker.internal` resolves automatically, but the `extra_hosts` entry ensures it works on Linux too. The `OLLAMA_URL` env var overrides the default `http://localhost:11434`.

## Performance

| Metric | Optuna (current) | LLM Agent |
|--------|-----------------|-----------|
| Trials per window | 40 | 12-18 |
| Backtests in parallel | 20 (per phase) | 6 (per round) |
| Wall time per window | ~35s | ~20s (backtests) + ~45s (2 LLM calls) = ~65s |
| Total per symbol (4 windows) | ~2.5 min | ~4.5 min |
| GPU usage | None | ~4-5 GB VRAM during LLM calls |
| API cost | Free | Free (local) |

Note: LLM calls are ~20-30s each on RTX 2060 Super with Q4_K_M quantization. The trade-off is slower wall time for smarter, market-aware parameter selection. The total trial count drops from 40 to 12-18, meaning fewer backtests overall.

Threading: `_run_llm_window` runs in the default `ThreadPoolExecutor` via `run_in_executor(None, ...)`. LLM HTTP calls block one thread for 20-30s each. This is fine because walk-forward runs symbols sequentially (one at a time). If parallel symbol optimization is added later, a dedicated thread pool or async HTTP client would be needed.

## Test Plan

1. **Unit test `llm_optimizer.py`**: Mock Ollama HTTP responses. Test JSON parsing, clamping, fallback to CHAMPION_DEFAULTS, retry on bad JSON, OllamaUnavailableError raising.
2. **Unit test `market_summarizer.py`**: Feed known candle data, verify output contains all three sections, verify daily resampling, verify 4h RSI/ADX columns, verify context budget truncation.
3. **Unit test `_run_llm_window`**: Mock `propose_params` / `refine_params`, verify round flow (2 rounds, or 3 if round 2 worse), verify sentinel on OllamaUnavailableError.
4. **Integration test**: Run full walk-forward on a small candle dataset with Ollama running locally. Verify end-to-end: summarize → propose → backtest → refine → backtest → pick best → OOS test.

## Dependencies

- **Ollama** installed and running on host (not in Docker)
- **Llama 3 8B** Q4_K_M quantization pulled via `ollama pull llama3`
- **`requests`** library (already in requirements.txt) for Ollama HTTP calls
- Bot connects to Ollama via `OLLAMA_URL` env var (default `http://localhost:11434`)
- For Docker: bot container needs `extra_hosts` mapping and `OLLAMA_URL` env var (see Docker Networking section)
