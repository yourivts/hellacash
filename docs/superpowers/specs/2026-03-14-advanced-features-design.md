# Advanced Trading Features Design Spec

**Date:** 2026-03-14
**Scope:** 7 new features for HellaCash trading bot
**Approach:** Hybrid (Approach C) — strategy enhancements integrate into the trading pipeline, analytics are a separate package, walk-forward extends the learning system. A `SignalProvider` protocol provides a consistent interface for all signal sources.

---

## 1. SignalProvider Protocol & Module Layout

### New Directories & Files

```
bot/indicators/onchain.py        # on-chain metrics signal provider
bot/indicators/orderbook.py      # order book analysis signal provider
bot/strategy/mtf_voter.py        # multi-timeframe voting system
bot/analytics/                   # new package
bot/analytics/__init__.py
bot/analytics/journal.py         # trade journaling
bot/analytics/attribution.py     # performance attribution
bot/analytics/benchmark.py       # benchmark comparison
bot/charts/                      # new package
bot/charts/__init__.py
bot/charts/snapshot.py           # matplotlib chart generation for journal entries
bot/learning/walk_forward.py     # walk-forward optimizer
```

### SignalProvider Protocol

Added to `bot/strategy/base.py`:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class SignalProvider(Protocol):
    name: str
    def score(self, symbol: str, **kwargs) -> float:
        """Return a directional score in [-1.0, +1.0]."""
        ...
    def is_available(self) -> bool:
        """Whether this provider currently has data."""
        ...
```

The existing composite result, the new onchain module, and the new orderbook module all implement this protocol. The hybrid strategy blends scores from all available providers.

**Sync cache-then-read pattern:** `score()` is intentionally synchronous. On-chain and order book providers maintain internal caches that are populated asynchronously by separate polling/WebSocket tasks (same pattern as `SentimentAggregator.get_score()`). The `score()` method only reads from the cache — it never makes HTTP requests or blocks.

### MarketContext Extension

New fields on the existing `MarketContext` dataclass in `bot/strategy/base.py`:

```python
candles_15m: Optional[pd.DataFrame] = None
candles_4h: Optional[pd.DataFrame] = None
candles_1d: Optional[pd.DataFrame] = None
onchain_score: float = 0.0
orderbook_imbalance: float = 0.0
market_regime: str = "unknown"
```

---

## 2. Multi-Timeframe Strategy Voting

**Module:** `bot/strategy/mtf_voter.py`

### Timeframes & Weights

Runs `composite.compute()` independently on 4 timeframes: 15m, 1h, 4h, 1d.

Base weights: `{"15m": 0.15, "1h": 0.25, "4h": 0.35, "1d": 0.25}`

Regime-adaptive overrides:

| Regime | 15m | 1h | 4h | 1d |
|--------|-----|----|----|-----|
| trending | 0.10 | 0.20 | 0.40 | 0.30 |
| ranging | 0.30 | 0.35 | 0.25 | 0.10 |
| volatile | 0.10 | 0.25 | 0.35 | 0.30 |
| unknown | 0.15 | 0.25 | 0.35 | 0.25 |

### Logic

1. For each timeframe, call `composite.compute(df_tf, weights)` to get a `SignalResult`. The same indicator weights (from the learning system) are shared across all timeframes — only the candle data differs
2. Select timeframe weight map based on current regime
3. Weighted average of per-TF `technical_score` values → `mtf_score` in [-1, +1]
4. Compute `agreement_ratio`: fraction of non-neutral timeframes that agree on the majority direction. Neutral timeframes (|score| < 0.15) are excluded from both numerator and denominator. If fewer than 2 timeframes are non-neutral, `agreement_ratio` = 0. If `agreement_ratio` < 0.5, dampen `mtf_score` by 50%
5. Return `MTFResult` dataclass with `mtf_score`, `agreement_ratio`, and per-TF breakdown

### Integration into HybridStrategy

Replaces the old formula. Both the 1h dampening in `hybrid.py` (lines 57-62) AND the separate 1h EMA confirmation in `trading_loop.py` are removed (subsumed by MTF voting).

New formula:
```
final_score = w_tech * mtf_score + w_sent * sentiment + w_onchain * onchain + w_book * orderbook
```

Default weights: `w_tech=0.55, w_sent=0.20, w_onchain=0.15, w_book=0.10` — configurable in `.env`.

**Entry threshold recalibration:** The current `entry_threshold` of 0.40 was tuned for the old 2-component formula. The new 4-component formula has different score distribution characteristics. The initial threshold is lowered to 0.35 as a starting point; walk-forward optimization will tune it from there.

**Weight redistribution when providers are disabled:** When `ONCHAIN_ENABLED=false` or `ORDERBOOK_ENABLED=false`, the disabled provider's weight is redistributed proportionally to the remaining providers. For example, if both are disabled, `w_tech` and `w_sent` scale to `0.55/0.75=0.733` and `0.20/0.75=0.267` respectively. This follows the same pattern used by the sentiment aggregator for unavailable sources.

### Data Requirements

- 15m: resampled from 5m (3 bars per 15m bar)
- 4h: resampled from 5m (48 bars per 4h bar) — needs ~500 5m candles for 30 usable 4h bars
- 1d: resampled from 5m (288 bars per 1d bar) — needs ~1500 5m candles, fetched on startup via the existing candle loader

**Required changes to `CandleCache`:** The current `MAX_CANDLES = 500` and loader fetch `limit=200` are insufficient for 4h/1d resampling. Changes needed:
- Increase `MAX_CANDLES` to 2000 in `bot/data_loader.py`
- Increase initial candle fetch to `limit=1500` for 5m interval on startup
- Subsequent WebSocket updates keep the cache within MAX_CANDLES via the existing trim logic

---

## 3. On-Chain Metrics

**Module:** `bot/indicators/onchain.py`

Implements `SignalProvider`. Polls free public APIs every 5 minutes (configurable via `ONCHAIN_POLL_INTERVAL_SECS`).

### Data Sources

| Metric | Source | Endpoint | Signal Logic |
|--------|--------|----------|--------------|
| Funding rates | Binance public API (no key) | `GET /fapi/v1/fundingRate?symbol=BTCUSDT&limit=1` | > 0.01% → -0.5 (contrarian bearish), < -0.01% → +0.5 |
| Open interest | Binance public API (no key) | `GET /fapi/v1/openInterest?symbol=BTCUSDT` | Rising OI + rising price → +0.5. Rising OI + falling price → -0.5. Compare current vs cached previous value |
| Whale transactions | Blockchair free API | `GET /bitcoin/transactions?q=output_total(1000000000..)&limit=10&s=time(desc)` | Spike in large txns vs 24h rolling count → ±0.3 directional bias. BTC only. Free tier: ~30 req/day — poll every 30 min, not every 5 min |
| Exchange reserves | Mempool.space free API | `GET /api/address/{address}` for known exchange cold wallets | Approximate exchange balance changes via known Binance/Coinbase/Kraken cold wallet addresses. BTC only. Supplementary signal |

**Note on Binance futures data as proxy:** The bot trades on Bitvavo spot, but futures market data (funding rates, OI) from Binance is used as a sentiment proxy — futures positioning reflects market-wide conviction, not exchange-specific behavior.

**Symbol mapping:** Bitvavo symbols (e.g., `BTC-EUR`) map to Binance futures symbols (e.g., `BTCUSDT`). A static mapping covers the major pairs. Unknown pairs gracefully return no on-chain signal.

### Scoring

Weighted average: funding 35%, open interest 35%, whale activity 20%, exchange reserves 10%. Unavailable metrics redistribute proportionally to available ones.

### Architecture

- Per-symbol cache with circuit breaker (3 failures → 10 min cooldown), matching existing sentiment circuit breaker pattern
- Uses `urllib.request` with 10s timeout (matching existing codebase pattern)
- Binance public endpoints: no API key, generous rate limits (1200 req/min IP-based) — 5 min poll interval is safe
- Blockchair free tier: ~30 req/day — poll whale data every 30 minutes, cache aggressively
- Whale and exchange reserve tracking BTC only (partially ETH via Etherscan free tier `?module=proxy&action=eth_blockNumber`). Altcoins use funding + OI only
- On-chain scores persisted to a new `onchain_scores` table (mirrors `sentiment_scores` pattern) for auditability

### Config

```env
ONCHAIN_POLL_INTERVAL_SECS=300
ONCHAIN_ENABLED=true
```

---

## 4. Order Book Analysis

**Module:** `bot/indicators/orderbook.py`

Implements `SignalProvider`. Two components: signal input and depth analysis.

### 4a: Bid/Ask Imbalance (Signal Input)

Feeds into hybrid strategy via `orderbook_imbalance` in `MarketContext`.

```
bid_volume = sum of qty across top N bid levels
ask_volume = sum of qty across top N ask levels
imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)  # [-1, +1]
```

- Imbalance > +0.3 → buying pressure (bullish)
- Imbalance < -0.3 → selling pressure (bearish)
- Between → neutral

**Wall detection:** Single order level with qty > 3x average level size in top 25 levels.

### 4b: Depth Analysis (Frontend & API)

```python
@dataclass
class DepthAnalysis:
    imbalance: float                    # [-1, +1]
    spread_pct: float                   # bid-ask spread as % of mid price
    bid_walls: List[PriceLevel]         # large buy orders
    ask_walls: List[PriceLevel]         # large sell orders
    support_levels: List[float]         # clustered bid volume
    resistance_levels: List[float]      # clustered ask volume
    depth_buckets: List[DepthBucket]    # for heatmap visualization
```

**Support/resistance detection:** Group levels into 0.5% price buckets. Buckets with volume > 2x median → flagged as support (bids) or resistance (asks). Top 3 of each returned.

**Depth buckets:** 50 evenly spaced buckets spanning ±5% from mid price with aggregated bid/ask volume for frontend heatmap.

### WebSocket Integration

Add `on_book` callback to `BitvavoWebSocket.__init__()`, modify `_subscribe()` to include `{"name": "book", "markets": [...]}`, and add book message handling to the dispatch method.

**Order book state management:** Bitvavo sends a full book snapshot on initial subscribe, followed by incremental delta updates (price levels with updated quantities; quantity=0 means remove level). The `OrderBookProvider` maintains the full book state per symbol:

```python
class _BookState:
    bids: Dict[float, float]   # price → quantity
    asks: Dict[float, float]   # price → quantity
    last_update: float         # timestamp

    def apply_delta(self, side: str, price: float, qty: float):
        book = self.bids if side == "bid" else self.asks
        if qty == 0:
            book.pop(price, None)
        else:
            book[price] = qty
```

Analysis (imbalance, walls, S/R) is recomputed at most once per second per symbol using the current book state. This throttling prevents CPU overhead on busy markets.

### API

```
GET /api/orderbook/{symbol}     → DepthAnalysis JSON
```

Also pushed via `/ws/feed` as `orderbook.update` events.

### Config

```env
ORDERBOOK_DEPTH_LEVELS=25
ORDERBOOK_ENABLED=true
```

---

## 5. Trade Journaling

**Module:** `bot/analytics/journal.py` and `bot/charts/snapshot.py`

### Database Model

New table `trade_journal`:

| Column | Type | Description |
|--------|------|-------------|
| id | Integer PK | |
| trade_id | Integer FK → trades.id | unique |
| symbol | String(20) | |
| direction | String(10) | |
| strategy_name | String(50) | |
| market_regime | String(20) | |
| entry_technical_scores | JSON | per-indicator breakdown at entry |
| entry_mtf_scores | JSON | per-timeframe scores at entry |
| entry_sentiment_score | Float | |
| entry_onchain_score | Float (nullable) | |
| entry_orderbook_imbalance | Float (nullable) | |
| entry_composite_score | Float | |
| exit_technical_scores | JSON (nullable) | per-indicator breakdown at exit |
| exit_composite_score | Float (nullable) | |
| entry_reasoning | Text | auto-generated explanation |
| exit_reasoning | Text (nullable) | auto-generated explanation |
| entry_chart_path | String(255) (nullable) | path to entry PNG |
| exit_chart_path | String(255) (nullable) | path to exit PNG |
| created_at | DateTime | |

### Auto-Generated Reasoning

Builds human-readable strings from the indicator snapshot. Example:

> "LONG BTC-EUR via hybrid in TRENDING regime. MTF agreement: 3/4 timeframes bullish (15m: +0.42, 1h: +0.61, 4h: +0.55, 1d: -0.12). Technical composite: +0.58 (RSI 34 oversold, MACD histogram rising, price above EMA20/50). Sentiment: +0.31 (news bullish). On-chain: +0.22 (funding negative, OI rising). Order book: +0.15 (bid imbalance 0.38, support wall at €61,200). Final score: +0.64 vs threshold 0.40."

Exit reasoning includes exit trigger, hold duration, indicator state at exit, and P&L.

### Chart Snapshots

**Module:** `bot/charts/snapshot.py` — matplotlib with dark theme.

**Entry chart:**
- 100 candles of 5m data (80 before, 20 after entry if available)
- Candlestick chart with entry price marker (horizontal dashed line + arrow)
- Overlay: EMA20/50, Bollinger Bands
- Subplots: RSI, MACD histogram, Volume with surge highlighting
- Saved to `data/charts/{symbol}_{trade_id}_entry.png`

**Exit chart:**
- Full trade duration + 20 candles before entry
- Entry AND exit markers, stop-loss and take-profit lines
- Same overlays as entry chart
- Saved to `data/charts/{symbol}_{trade_id}_exit.png`

### Integration

Hooks into event bus:
- `trade.opened` → create journal entry with entry snapshot + generate entry chart
- `trade.closed` → update journal entry with exit snapshot + generate exit chart

**Required fix:** `TOPIC_TRADE_OPENED` is already published from `bot/exchange/order_manager.py` but with a minimal payload (order_id, symbol, side, fill_price). Rather than adding a second publish (which would fire subscribers twice), extend the existing `order_manager.py` publish payload with the additional fields the journal needs: `signal` (full Signal object with indicator_snapshot), `market_regime`, `mtf_scores`, `onchain_score`, `orderbook_imbalance`, and `candle_data` (last 100 5m candles for chart generation). These extra fields are passed into the order manager from the trading loop at trade-open time.

**Chart generation threading:** Matplotlib chart generation takes 500ms-2s per chart. To avoid blocking the trading loop, chart generation runs in a thread pool executor via `asyncio.get_event_loop().run_in_executor(None, generate_chart, ...)`. The journal entry is created immediately with the snapshot data; the chart path is updated asynchronously once generation completes.

### API

```
GET /api/journal                    → paginated journal entries
GET /api/journal/{trade_id}         → single entry with full snapshot data
GET /api/charts/{filename}          → serves chart PNG files
```

Frontend "view trade" page also renders stored candle data interactively with entry/exit markers.

---

## 6. Performance Attribution

**Module:** `bot/analytics/attribution.py`

Queries existing `trades` table. No new tables, but adds one column to `Trade` model:

```python
market_regime: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
```

Populated at trade open time from the already-detected regime. Existing trades in the database will have `NULL` for this column — the attribution engine treats `NULL` as `"unknown"` when grouping by regime. No backfill of historical trades.

### Dimensions

**By Strategy:** P&L, trade count, win rate, avg P&L, Sharpe, contribution % — grouped by `strategy_name`.

**By Market Regime:** Same metrics grouped by `market_regime`.

**By Trading Session:** Trades bucketed by entry hour (UTC):

| Session | Hours (UTC) |
|---------|-------------|
| Asia | 00:00–07:59 |
| Europe | 08:00–13:59 |
| US | 14:00–20:59 |
| Overlap/Off-hours | 21:00–23:59 |

**By Direction:** LONG vs SHORT comparison.

### Computation

`AttributionEngine.compute()` fetches trades once and groups in memory. Returns `AttributionReport` dataclass containing all four breakdowns plus summary stats (total P&L, overall win rate, best/worst strategy, best/worst session).

### API

```
GET /api/analytics/attribution?start=2026-01-01&end=2026-03-14
```

---

## 7. Benchmark Comparison

**Module:** `bot/analytics/benchmark.py`

Compares bot performance against buy-and-hold for BTC-EUR, ETH-EUR, and each traded asset.

### Benchmark Calculation

```
buy_and_hold_return = (end_price - start_price) / start_price * 100
```

Applied to the same starting capital.

### Report

```python
@dataclass
class BenchmarkReport:
    period_start: datetime
    period_end: datetime
    initial_capital: float
    bot_total_pnl: float
    bot_return_pct: float
    bot_sharpe: float
    bot_max_drawdown_pct: float
    btc_benchmark: BenchmarkEntry        # symbol, prices, return, P&L
    eth_benchmark: BenchmarkEntry
    per_asset_benchmarks: List[BenchmarkEntry]
    alpha_vs_btc: float                  # bot_return - btc_return
    alpha_vs_eth: float                  # bot_return - eth_return
```

### Equity Curve Comparison

Time-series data for frontend charting: bot equity (from `portfolio_snapshots` table) vs benchmark equity (daily close prices, cumulative return from start date).

### Data Source

Prices from `candles` table. Falls back to Bitvavo REST API if candles unavailable for full range.

Default range: first trade to last trade if no dates specified. Initial capital is determined from the first `portfolio_snapshots.total_equity_eur` in the date range (or falls back to the `PAPER_INITIAL_BALANCE` config value).

### API

```
GET /api/analytics/benchmark?start=2026-01-01&end=2026-03-14
```

---

## 8. Walk-Forward Optimization

**Module:** `bot/learning/walk_forward.py`

### Rolling Windows

```
|------- 60 days train -------|----- 14 days test -----|
                    |------- 60 days train -------|----- 14 days test -----|
                                        |------- 60 days train -------|----- 14 days test -----|
```

Train: 60 days. Test: 14 days. Step: 14 days (no gaps). Minimum 3 windows required.

### BacktestEngine Parameterization

The current `BacktestEngine` hardcodes `StrategyRouter()` (line 102). To support walk-forward, add an optional `strategy_params` argument:

```python
class BacktestEngine:
    def __init__(self, candles, initial_capital=10_000.0, ...,
                 strategy_params: Optional[Dict[str, Any]] = None):
        self._router = StrategyRouter()
        if strategy_params:
            self._router.update_hybrid_params(
                sentiment_weight=strategy_params.get("sentiment_weight", 0.25),
                entry_threshold=strategy_params.get("entry_threshold", 0.40),
                indicator_weights=strategy_params.get("indicator_weights"),
            )
```

This allows each walk-forward window to backtest with different parameter sets without modifying the core engine.

### Per-Window Process

1. Fetch candles for train period from `candles` table
2. For each candidate parameter set (generated by Bayesian search from `param_optimizer.py`), run `BacktestEngine` on train period with `strategy_params=candidate`
3. Select best params from training backtest (by net P&L, since AUC requires labeled data which backtesting doesn't produce)
4. Run `BacktestEngine` on test period with those best params (out-of-sample score)
5. Record `WFWindow` result

### Adoption Criteria

Parameters adopted only if ALL hold:
- At least 3 test windows completed
- Average OOS Sharpe ratio > 0.5 (consistent risk-adjusted returns)
- Sharpe stability (std dev across windows) < 0.5 (not wildly inconsistent)
- Average OOS P&L > 0 (actually profitable out of sample)
- New params outperform current params' average OOS P&L

### Scheduling

Runs bi-weekly via scheduler (time-based, not trade-count). Falls back to current `ParamOptimizer` if < 88 days of candle history.

```python
if has_enough_history(88):
    await walk_forward_optimizer.run()
else:
    await param_optimizer.optimize_hybrid()
```

### API

```
GET /api/analytics/walk-forward          → latest WFResult (from DB/cache)
POST /api/analytics/walk-forward/run     → trigger manual run (returns 202 Accepted immediately)
```

The POST endpoint launches the walk-forward run as a background `asyncio.Task` and returns a 202 with `{"status": "running"}`. Progress and results are stored in memory and retrievable via the GET endpoint. Only one walk-forward run can be active at a time — subsequent POST requests return 409 Conflict if a run is in progress.

---

## Summary of All New API Endpoints

| Endpoint | Method | Feature |
|----------|--------|---------|
| `/api/orderbook/{symbol}` | GET | Order book depth analysis |
| `/api/journal` | GET | Paginated trade journal |
| `/api/journal/{trade_id}` | GET | Single journal entry |
| `/api/charts/{filename}` | GET | Chart PNG files |
| `/api/analytics/attribution` | GET | Performance attribution |
| `/api/analytics/benchmark` | GET | Benchmark comparison |
| `/api/analytics/walk-forward` | GET | Walk-forward results |
| `/api/analytics/walk-forward/run` | POST | Trigger walk-forward |

## Summary of Database Changes

- New table: `trade_journal`
- New table: `onchain_scores` (mirrors `sentiment_scores` structure)
- New column on `trades`: `market_regime` (String, nullable)
- Alembic migration required

## Chart Retention

Chart PNGs in `data/charts/` accumulate over time (~200KB per chart, 2 per trade). A cleanup task runs weekly via the scheduler, deleting charts older than 90 days. Journal entries retain the snapshot data (scores, reasoning) permanently — only the PNG files are pruned.

## Summary of Config Additions

```env
# Multi-timeframe voting
MTF_TECH_WEIGHT=0.55
MTF_SENT_WEIGHT=0.20
MTF_ONCHAIN_WEIGHT=0.15
MTF_BOOK_WEIGHT=0.10

# On-chain metrics
ONCHAIN_ENABLED=true
ONCHAIN_POLL_INTERVAL_SECS=300

# Order book
ORDERBOOK_ENABLED=true
ORDERBOOK_DEPTH_LEVELS=25
```

## Existing Systems NOT Modified

- Sentiment scrapers (Reddit, News, Twitter, Fear & Greed) — unchanged
- Risk engine hard limits — unchanged
- Discord notifications — unchanged (journal charts could optionally be attached later)
- Paper trading system — unchanged
