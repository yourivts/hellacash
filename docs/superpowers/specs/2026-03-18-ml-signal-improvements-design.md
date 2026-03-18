# ML Signal Generator Improvements — Design Spec

## Goal

Improve ML signal generator accuracy by: (1) adding 25 new external data features from free APIs, (2) replacing LSTM with a Transformer encoder, (3) adding class-weighted training, (4) threshold tuning via search, and (5) Optuna hyperparameter optimization. Add failsafe mechanisms for API reliability.

## Architecture Overview

Five improvements to the existing ML signal generator, all implemented in one cycle:

1. **External Data Pipeline** — fetch and cache historical data from 12+ free APIs
2. **Extended Features** — expand N_TABULAR from 65 → 90
3. **Transformer Encoder** — replace LSTM with 4-layer Transformer
4. **Training Improvements** — class weighting, threshold search, Optuna HPO
5. **Failsafe System** — graceful degradation when APIs fail

---

## 1. External Data Pipeline

### New module: `bot/data/external_features.py`

Two modes of operation:

**Training mode** — bulk-fetch full history, cache to parquet in `data/external_cache/`:

| Source | API | Data | History | Granularity | Key Required |
|--------|-----|------|---------|-------------|--------------|
| Binance Futures | `fapi.binance.com/fapi/v1/fundingRate` | Funding rates | 2019+ | 8-hourly | No |
| Alternative.me | `api.alternative.me/fng/` | Fear & Greed Index | Feb 2018+ | Daily | No |
| Google Trends | pytrends library | Search interest | 2015+ | Weekly | No |
| yfinance | yfinance library | DXY, S&P 500, Gold, VIX, Treasury yields | Decades | Daily | No |
| BGeometrics | `charts.bgeometrics.com/api/` | NVT, MVRV, SOPR, Puell, exchange flows, hashrate | Full | Daily | No |
| CoinMetrics | `community-api.coinmetrics.io/v4` | Active addresses, tx count (BTC+ETH) | Years | Daily | No |
| DefiLlama | `stablecoins.llama.fi/` | USDT/USDC supply | Inception | Daily | No |
| DefiLlama | `api.llama.fi/charts` | Total DeFi TVL | 2020+ | Daily | No |
| Coinalyze | `api.coinalyze.net/v1/` | OI, liquidations aggregated | Daily unlimited | Daily | Free signup |
| Binance CSV | `data.binance.vision` | Liquidation snapshots | 2020+ | Daily | No |
| Binance klines | `fapi.binance.com/fapi/v1/klines` | Taker buy/sell volume | Inception | 5m | No |
| Deribit | `deribit.com/api/v2/public/` | DVOL implied volatility | Limited | Hourly | No |

**Live mode** — periodic refresh:
- Daily data: refresh every hour, use cached value between refreshes
- Funding rates: refresh every trading cycle
- All data cached in memory with timestamps

### Caching strategy
- Training: parquet files in `data/external_cache/{source}_{asset}.parquet`
- Live: in-memory dict with last-fetched timestamp per source
- Cache is reused across training runs (skip fetch if parquet exists and covers the date range)

### Data alignment rules
- All external data is aligned to the 5m candle DatetimeIndex via `reindex(idx_5m, method='ffill')`
- Daily data (Fear & Greed, macro, on-chain): value applies to all 5m bars on that UTC date
- Weekly data (Google Trends): forward-filled across the week
- 8-hourly data (funding rates): forward-filled to 5m
- Macro market data (DXY, S&P, yields): uses previous trading day's close to avoid look-ahead bias (markets close at ~21:00 UTC)
- Periods before a data source's inception: filled with 0.0 (XGBoost natively handles this as missing data)
- Deribit DVOL: only available since March 2021; earlier periods filled with 0.0

### Source reliability notes
- **Google Trends (pytrends)**: Unreliable — frequently rate-limited (429 errors). Treated as optional in both training and live modes. Training retries up to 3 times with exponential backoff; on failure, features 67-68 are zero-filled. Live mode uses last cached weekly value (staleness threshold = 7 days).
- **Coinalyze**: Requires free API key. Key stored in environment variable `COINALYZE_API_KEY`. If key is not configured, OI/liquidation features are zero-filled and a warning is logged. Training continues without this source.
- **BGeometrics**: Lesser-known provider. If unavailable, fall back to Blockchain.com Charts API for hashrate/difficulty. NVT/MVRV/SOPR features zero-filled if both sources fail.
- **BTC dominance (index 88)**: Computed as `BTC market cap / total crypto market cap` using CoinGecko `/global` endpoint. If unavailable, derived from yfinance BTC-USD market cap vs total crypto ETF proxies. Zero-filled on failure.

---

## 2. Extended Feature Engineering

### N_TABULAR: 65 → 90

**Existing slots redefined (indices 55-61):**

The live bot currently passes funding_rate, funding_score, ob_imbalance, spread_pct, bid_ask_wall_ratio, onchain_composite, exchange_reserve_trend into these slots. The semantics change as follows:

| Index | Old Meaning | New Meaning | Migration |
|-------|-------------|-------------|-----------|
| 55 | funding_rate (kept) | Funding rate current (Binance) | Same semantic, now populated in training too |
| 56 | funding_score | Funding rate 7d average | Live bot updated to compute 7d avg |
| 57 | ob_imbalance | OI change 24h | Live bot fetches from Binance/Coinalyze |
| 58 | spread_pct | Long liquidations 24h | Live bot fetches from Binance |
| 59 | bid_ask_wall_ratio | Short liquidations 24h | Live bot fetches from Binance |
| 60 | onchain_composite | BTC exchange netflow | Live bot fetches from BGeometrics |
| 61 | exchange_reserve_trend | Active addresses change 7d | Live bot fetches from CoinMetrics |

Indices 62-64 (BTC cross-asset returns, correlation) — unchanged.

The `extract_tabular_features()` function signature changes: the individual float params (funding_rate, funding_score, etc.) are replaced with a single `external_data: dict` parameter. The live bot's trading loop is updated to build this dict from the external data provider.

**New features (indices 65-89):**

| Index | Feature | Source | Scale |
|-------|---------|--------|-------|
| 65 | Fear & Greed Index | Alternative.me | 0-1 |
| 66 | Fear & Greed 7d momentum | Alternative.me | -1 to 1 |
| 67 | Google Trends "bitcoin" | pytrends | 0-1 |
| 68 | Google Trends "crypto" | pytrends | 0-1 |
| 69 | DXY daily return | yfinance | -1 to 1 |
| 70 | S&P 500 daily return | yfinance | -1 to 1 |
| 71 | Gold daily return | yfinance | -1 to 1 |
| 72 | VIX level | yfinance | 0-1 (scaled by 80) |
| 73 | 10Y Treasury yield | yfinance/FRED | 0-1 (scaled by 10%) |
| 74 | 10Y-2Y yield spread | yfinance/FRED | -1 to 1 |
| 75 | BTC NVT ratio | BGeometrics | 0-1 (log-scaled) |
| 76 | BTC MVRV ratio | BGeometrics | 0-1 (scaled by 5) |
| 77 | BTC SOPR | BGeometrics | -1 to 1 (centered at 1.0) |
| 78 | BTC Puell Multiple | BGeometrics | 0-1 (scaled by 4) |
| 79 | BTC hashrate change 30d | BGeometrics (fallback: Blockchain.com) | -1 to 1 |
| 80 | ETH active addresses change 7d | CoinMetrics | -1 to 1 |
| 81 | USDT supply change 7d | DefiLlama | -1 to 1 |
| 82 | DeFi total TVL change 7d | DefiLlama | -1 to 1 |
| 83 | OI change 7d | Coinalyze | -1 to 1 |
| 84 | Liquidation ratio (long/total 24h) | Binance CSV | 0-1 |
| 85 | Taker buy ratio | Binance klines | 0-1 |
| 86 | BTC DVOL implied volatility | Deribit | 0-1 (scaled by 200%) |
| 87 | Funding rate 24h average | Binance | -1 to 1 |
| 88 | BTC dominance change 7d | CoinGecko/yfinance | -1 to 1 |
| 89 | Stablecoin mcap / BTC mcap ratio | DefiLlama + CoinGecko | 0-1 |

**Total XGBoost input: 90 tabular + 16 Transformer embeddings = 106 features.**

### Changes to existing code
- `ml_features.py`: Update `N_TABULAR = 90`, extend `extract_tabular_features()` signature to accept `external_data: dict` parameter, update `_precompute_indicators()` and `_batch_extract_tabular()` to accept and merge external data arrays at the correct indices
- `ml_signal_generator.py`: Pass external data to feature extraction
- `trading_loop.py`: Fetch external data and pass to ML signal generator
- All tests updated for new feature count

---

## 3. Transformer Encoder

### New module: `bot/learning/transformer_embedder.py`

Replaces `lstm_embedder.py`. Same public interface.

**Architecture:**
```
Input: (batch, 96, 7)
  → Linear(7 → 64)                          # projection
  → + SinusoidalPositionalEncoding(96, 64)   # position info
  → TransformerEncoder(
      num_layers=4,
      d_model=64,
      nhead=4,
      dim_feedforward=128,
      dropout=0.1,
      batch_first=True,
    )
  → MeanPooling(dim=1) → (batch, 64)        # aggregate (h_pooled)
  ┌─→ Linear(64 → 16) + ReLU → embedding    # embed branch (from h_pooled)
  └─→ Linear(64 → 60)                       # predict branch (5 OHLCV channels × 12 bars, discarded after training)
```

Both the embedding projection and the prediction head branch from the same 64-dim mean-pooled output (`h_pooled`). The prediction head is only used during self-supervised training and is discarded afterward.

**Design choices:**
- d_model=64, 4 heads (16-dim per head) — fits 8GB VRAM
- Mean pooling over CLS token — more stable for time-series
- Sinusoidal positional encoding — no learned parameters, generalizes to unseen positions
- Same self-supervised objective: predict next 12 bars normalized OHLCV (MSE loss)
- Cosine annealing LR scheduler (better for Transformers than constant LR)

**Public interface (same as LSTMEmbedder):**
- `embed(x) → (batch, 16)` — get embeddings
- `predict_next(x) → (batch, 60)` — predict next bars (training only)
- `embed_numpy(seq) → (16,)` — single sequence convenience
- `save(path)` / `load(path)` — model persistence

**Backward compatibility:**
- `lstm_embedder.py` remains in codebase but is not imported by default
- `ml_signal_generator.py` loads from `transformer.pt` (falls back to `lstm.pt` if not found)
- Training script produces `transformer.pt`

---

## 4. Training Improvements

**Training pipeline order:** Optuna Transformer HPO → train best Transformer → extract embeddings → threshold search (needs embeddings for F1) → Optuna XGBoost HPO (with class weights) → train final XGBoost models.

### 4a. Class-weighted training

XGBoost `scale_pos_weight` parameter set per model:
```python
scale_pos_weight = n_negative / n_positive
```

This tells XGBoost to penalize false negatives proportionally to class imbalance. For a model with 16% positive rate, false negatives get ~5x penalty.

### 4b. Threshold tuning

After Transformer training (embeddings needed for F1 evaluation), search for optimal thresholds per horizon:
- For each horizon (30m, 1h, 4h, 12h, 24h, 72h), try thresholds from 50% to 150% of the current value in 10 steps
- Compute labels with each threshold
- Pick threshold that maximizes validation **F1 score** of a quick XGBoost run (100 estimators, no HPO)
- Use winning thresholds for the full training

### 4c. Optuna hyperparameter optimization

**Transformer HPO (20 trials):**
- `d_model`: [32, 64, 128]
- `n_heads`: [2, 4, 8] (constrained: n_heads must divide d_model)
- `n_layers`: [2, 3, 4, 6]
- `dropout`: [0.05, 0.1, 0.15, 0.2, 0.3]
- `lr`: [5e-4, 1e-3, 2e-3]
- Objective: validation MSE loss
- VRAM management: batch size is reduced proportionally for larger d_model configs (d_model=128 uses half the default batch size). Trials that OOM are pruned automatically via Optuna's `TrialPruned` exception.

**XGBoost HPO (30 shared trials on 4h_up, then 10 fine-tune trials per model):**
- `max_depth`: int [4, 10]
- `learning_rate`: float [0.01, 0.2] (log scale)
- `n_estimators`: int [200, 1000]
- `min_child_weight`: int [5, 50]
- `subsample`: float [0.6, 1.0]
- `colsample_bytree`: float [0.5, 1.0]
- `reg_alpha`: float [0.01, 1.0] (log scale)
- `reg_lambda`: float [0.5, 5.0]
- Objective: validation F1 score

**Optimization strategy:** Run shared HPO on a single representative model (4h_up) to find a good baseline, then optionally fine-tune per model with 10 trials each. This reduces total trials from 600 to ~160.

### 4d. Updated metrics

Training script reports per model:
- F1 score (primary metric)
- Precision and recall
- Accuracy (for comparison)
- AUC-ROC
- Positive rate and scale_pos_weight used

---

## 5. Failsafe System

### Layer 1: Per-feature graceful degradation

Each external data source in live mode:
- On fetch success: update cache + timestamp
- On fetch failure: use last cached value, log warning
- If no cached value exists: fill with 0.0 (matches training behavior for missing periods)

### Layer 2: Staleness detection

```python
STALENESS_THRESHOLDS = {
    "funding_rate": timedelta(hours=2),
    "fear_greed": timedelta(hours=48),
    "google_trends": timedelta(days=7),
    "macro": timedelta(hours=48),
    "onchain": timedelta(hours=48),
    "defi": timedelta(hours=48),
    "oi_liquidations": timedelta(hours=48),
    "dvol": timedelta(hours=48),
}
```

- Track staleness per source
- If >30% of external feature sources exceed their staleness threshold: **disable ML signal generator**
- Log alert: "ML signals disabled: {n}/{total} data sources stale"
- Auto-re-enable when freshness recovers below 30%
- Bot continues operating without ML signals (no trades from ML path)

---

## 6. Backward Compatibility & Rollback

- **Model loading**: `ml_signal_generator.py` tries `transformer.pt` first, falls back to `lstm.pt`. This allows rolling back to the LSTM by simply deleting `transformer.pt`.
- **Feature count**: `feature_config.json` stores `n_tabular` (90) and `n_embed` (16). The signal generator reads these at load time. Old models with `n_tabular=65` still load correctly — the generator checks the config and uses the matching feature extraction path.
- **External data unavailable**: If the external data provider fails entirely at startup, the bot logs a warning and runs with zero-filled external features (indices 55-89). This matches training behavior for pre-inception periods.
- **Config key**: `feature_config.json` writes `n_embed` (generic) instead of `n_lstm_embed`. When reading the config, accept both `n_embed` and `n_lstm_embed` (legacy) to avoid breaking existing saved models.

---

## Dependencies

Add to `requirements.txt`:
- `optuna>=3.0` — hyperparameter optimization
- `pytrends>=4.9` — Google Trends data
- `yfinance>=0.2` — macro market data
- `pyarrow>=14.0` — parquet caching (likely already installed with pandas)

---

## Files Changed

| File | Change |
|------|--------|
| `bot/data/external_features.py` | **NEW** — external data fetching, caching, staleness tracking |
| `bot/learning/transformer_embedder.py` | **NEW** — Transformer encoder (replaces LSTM) |
| `bot/learning/ml_features.py` | MODIFY — N_TABULAR 65→90, new feature slots, updated batch extraction |
| `bot/learning/ml_signal_generator.py` | MODIFY — load Transformer, pass external data, failsafe integration |
| `bot/trading_loop.py` | MODIFY — fetch external data, pass to ML signal generator |
| `bot/main.py` | MODIFY — initialize external data provider |
| `scripts/train_ml_signals.py` | MODIFY — external data fetch, Transformer training, threshold search, Optuna HPO, class weights, F1 metrics |
| `tests/test_external_features.py` | **NEW** — tests for data fetching, caching, staleness |
| `tests/test_transformer_embedder.py` | **NEW** — tests for Transformer model |
| `tests/test_ml_features.py` | MODIFY — update for N_TABULAR=90 |
| `tests/test_train_ml_signals.py` | MODIFY — update for new training pipeline |
| `requirements.txt` | MODIFY — add optuna, pytrends, yfinance |
