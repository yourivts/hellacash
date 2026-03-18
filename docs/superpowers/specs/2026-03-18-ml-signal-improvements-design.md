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

---

## 2. Extended Feature Engineering

### N_TABULAR: 65 → 90

**Existing slots now populated during training (indices 55-61):**
- 55: Funding rate current (Binance, scaled by 0.01)
- 56: Funding rate 7d average
- 57: OI change 24h (scaled)
- 58: Long liquidations 24h (scaled)
- 59: Short liquidations 24h (scaled)
- 60: BTC exchange netflow (BGeometrics, scaled)
- 61: Active addresses change 7d (CoinMetrics, scaled)

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
| 79 | BTC hashrate change 30d | Blockchain.com | -1 to 1 |
| 80 | ETH active addresses change 7d | CoinMetrics | -1 to 1 |
| 81 | USDT supply change 7d | DefiLlama | -1 to 1 |
| 82 | DeFi total TVL change 7d | DefiLlama | -1 to 1 |
| 83 | OI change 7d | Coinalyze | -1 to 1 |
| 84 | Liquidation ratio (long/total 24h) | Binance CSV | 0-1 |
| 85 | Taker buy ratio | Binance klines | 0-1 |
| 86 | BTC DVOL implied volatility | Deribit | 0-1 (scaled by 200%) |
| 87 | Funding rate 24h average | Binance | -1 to 1 |
| 88 | BTC dominance change 7d | CoinGecko/yfinance | -1 to 1 |
| 89 | Stablecoin mcap / BTC mcap ratio | DefiLlama | 0-1 |

**Total XGBoost input: 90 tabular + 16 Transformer embeddings = 106 features.**

### Changes to existing code
- `ml_features.py`: Update `N_TABULAR = 90`, extend `extract_tabular_features()` signature to accept external data dict, update `_precompute_indicators()` and `_batch_extract_tabular()` for training
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
  → MeanPooling(dim=1) → (batch, 64)        # aggregate
  → Linear(64 → 16) + ReLU → embedding      # project to embed dim
  → Linear(64 → 60)                          # prediction head (discarded after training)
```

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

### 4a. Class-weighted training

XGBoost `scale_pos_weight` parameter set per model:
```python
scale_pos_weight = n_negative / n_positive
```

This tells XGBoost to penalize false negatives proportionally to class imbalance. For a model with 16% positive rate, false negatives get ~5x penalty.

### 4b. Threshold tuning

Before XGBoost training, search for optimal thresholds per horizon:
- For each horizon (30m, 1h, 4h, 12h, 24h, 72h), try thresholds from 50% to 150% of the current value in 10 steps
- Compute labels with each threshold
- Pick threshold that maximizes validation **F1 score** of a quick XGBoost run (100 estimators, no HPO)
- Use winning thresholds for the full training

### 4c. Optuna hyperparameter optimization

**Transformer HPO (20 trials):**
- `d_model`: [32, 64, 128]
- `n_heads`: [2, 4, 8]
- `n_layers`: [2, 3, 4, 6]
- `dropout`: [0.05, 0.1, 0.15, 0.2, 0.3]
- `lr`: [5e-4, 1e-3, 2e-3]
- Objective: validation MSE loss

**XGBoost HPO (50 trials per model, or 30 shared + per-model fine-tune):**
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
