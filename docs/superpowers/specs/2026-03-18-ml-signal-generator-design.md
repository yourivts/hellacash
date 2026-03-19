# ML Signal Generator — Design Spec

## Goal

Replace the 8 hand-coded trading strategies with a machine-learning signal generation system. An ensemble of XGBoost (tabular features) and LSTM (temporal patterns) predicts directional probabilities at 6 time horizons. The existing RL Signal Evaluator gates each prediction and adjusts execution parameters. Online PPO learning continues adapting to live market conditions.

## Motivation

The current strategies (orderflow, squeeze, range, etc.) use simple indicator threshold rules. In 5-year backtests, every isolated strategy loses money across BTC, ETH, and XRP. The RL parameter tuner cannot create edge where the underlying signals have none. An ML model trained on actual outcomes — rather than hand-coded heuristics — should capture genuine predictive patterns in the feature space.

## Architecture

```
Market Data → Feature Engineering → ┬─ Tabular (65 features)
                                     │
                                     └─ Price Sequence (96×7)
                                              │
                                         LSTM → embedding (16-dim)
                                              │
                       Tabular (65) + Embedding (16) = 81 features
                                              │
                                     XGBoost × 12 models
                                     (up + down × 6 horizons)
                                              │
                              12 probabilities (up/down × 6)
                                              │
                                    RL Signal Evaluator (PPO)
                                   confidence + atr_mult + rr_ratio
                                              │
                                    ┌─────────┴─────────┐
                                take_trade          skip_signal
                                    │
                            Trade opens → closes → P&L
                                    │
                            Online PPO update
```

### Prediction Horizons

Twelve XGBoost classifiers — one "up" and one "down" model per horizon:

| Horizon | Bars (5m) | Use Case |
|---------|-----------|----------|
| 30m     | 6         | Entry timing |
| 1h      | 12        | Short-term momentum |
| 4h      | 48        | Swing trade signal |
| 12h     | 144       | Half-day trend |
| 24h     | 288       | Daily trend |
| 72h     | 864       | Multi-day conviction |

Labels are binarized using horizon-scaled thresholds to account for natural price variance at longer horizons:

| Horizon | Threshold | Rationale |
|---------|-----------|-----------|
| 30m     | 0.15%     | Tight — must exceed spread + fees |
| 1h      | 0.3%      | ~2× fees |
| 4h      | 0.8%      | Meaningful swing |
| 12h     | 1.5%      | Half-day move |
| 24h     | 2.5%      | Daily trend |
| 72h     | 4.0%      | Multi-day conviction |

For each horizon, two independent labels are generated:
- **label_up**: `1` if future return exceeds +threshold
- **label_down**: `1` if future return falls below -threshold

Both can be `0` simultaneously (sideways market). This requires 12 XGBoost models total (2 per horizon).

## Feature Engineering

### Tabular Features (~65 features)

| Group | Features | Count |
|-------|----------|-------|
| Price action | RSI (1h, 4h), MACD hist (1h, 4h), MACD sign, Stochastic K/D, CCI | 9 |
| Volatility | ATR% (1h, 4h), BB bandwidth, BB %b, Keltner position, ATR ratio (current/median) | 6 |
| Trend | EMA20/50/200 distances (%), EMA50 slope (4h, 1d), ADX (4h), Supertrend sign | 7 |
| Volume | Volume surge ratio, OBV trend, CMF, VWAP distance %, volume profile score | 5 |
| Divergence | RSI divergence, volume divergence | 2 |
| Regime | Regime ID (one-hot: 5), hours in current regime | 6 |
| Multi-TF returns | Price return over last 1h, 4h, 12h, 24h, 72h, 7d | 6 |
| Candle structure | Body ratio, upper/lower wick ratio, consecutive green/red count | 4 |
| Time | Hour sin/cos, day-of-week sin/cos, month sin/cos | 6 |
| Coin markers | Market cap tier, volatility class, coin age, is_BTC | 4 |
| Funding/OB | Funding rate, funding score, OB imbalance, spread %, bid/ask wall ratio | 5 |
| On-chain | On-chain composite score, exchange reserve trend | 2 |
| Cross-asset | BTC return (1h, 4h), BTC-correlation (30d) | 3 |

Funding, order book, and on-chain features are zero-filled during backtests on historical data. XGBoost handles missing values natively. As historical data for these sources is collected, models can be retrained with richer features.

### LSTM Price Sequence

Shape `(96, 7)` — last 96 five-minute candles (8 hours). Window length chosen to cover a full trading session while keeping sequence length manageable for LSTM training. At 5m resolution, 96 bars captures intraday momentum, volume patterns, and short-term trend structure without overwhelming the model with noise from older bars:

| Channel | Description |
|---------|-------------|
| 0 | Normalized close (% change from first bar) |
| 1 | Normalized volume (ratio to 20-bar average) |
| 2 | High-low range (normalized by close) |
| 3 | Body direction (+1 green, -1 red) |
| 4 | RSI (5m, scaled 0-1) |
| 5 | Close vs EMA20 distance (%) |
| 6 | Volume surge ratio |

## Model Architecture

### LSTM Component

```
Input: (batch, 96, 7)
  → LSTM(input=7, hidden=32, layers=2, dropout=0.2)
  → Last hidden state: (batch, 32)
  → Linear(32 → 16) + ReLU
  → Output: (batch, 16) embedding
```

~10K parameters. Trained via self-supervised next-bar prediction: given a 96-bar sequence, predict the next 12 bars' normalized OHLCV (5 channels × 12 bars = 60 outputs). Loss function: MSE on normalized values. This teaches the LSTM to encode price dynamics without needing classification labels. After training, the embedding layer is frozen and used as a feature extractor for XGBoost.

### XGBoost Models (×12)

Twelve independent binary classifiers: one "up" model and one "down" model per horizon. Each receives 81 features (65 tabular + 16 LSTM embedding). Using independent models (rather than P(down) = 1 - P(up)) allows the system to recognize sideways markets where neither direction has edge — both probabilities can be low simultaneously.

```python
{
    "objective": "binary:logistic",
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
}
```

## Training Pipeline

### Step 1: Data Collection
- Load 5 years of 5m candles per pair from Bitvavo API
- Compute all 65 tabular features at every SIGNAL_EVERY (30m) interval
- Build LSTM sequences (96-bar sliding windows)
- Compute labels: future returns at 6 horizons, binarized with horizon-scaled thresholds (see Prediction Horizons table)
- Generate independent up/down labels per horizon (12 label columns total)

### Step 2: Train LSTM (self-supervised)
- Objective: predict next 12 bars' normalized OHLCV from 96-bar input sequence
- Output head: Linear(32 → 60) for 12 bars × 5 channels
- Loss: MSE on normalized OHLCV values
- 50 epochs, batch_size=256, Adam lr=1e-3
- Validation: last 6 months held out
- After training, discard the prediction head and freeze the embedding layer
- Save as TorchScript for inference

### Step 3: Generate Embeddings
- Run trained LSTM on all data points (including test-set data)
- Append 16-dim embeddings to tabular features
- Note: the LSTM sees all time periods for embedding extraction, but this is not data leakage because the LSTM is trained via self-supervised prediction (no classification labels). The XGBoost walk-forward split remains the guard against label leakage

### Step 4: Train XGBoost (×12)
- Walk-forward split: train years 1-3, validate year 4, test year 5
- For each horizon × direction (6 × 2 = 12 models): fit XGBoost on (81 features, label)
- Log feature importances per model

### Step 5: Save Models
- LSTM: `models/ml_signals/lstm.pt`
- XGBoost: `models/ml_signals/xgb_{horizon}_{direction}.json` (×12)
- Feature config: `models/ml_signals/feature_config.json`

### Validation Strategy

Walk-forward (not random split):

```
Train:    [── Year 1-3 ──]
Validate: [── Year 4 ──]
Test:     [── Year 5 ──]   ← never seen during training
```

Success criteria:
- Each horizon model: accuracy > 52% on test set
- No single feature dominates >30% importance
- Predicted probabilities match observed frequencies (calibration)

## RL Signal Evaluator Updates

The observation space changes to receive ML predictions instead of raw market features:

| Group | Features | Count |
|-------|----------|-------|
| ML predictions | prob_up/down × 6 horizons | 12 |
| Cross-horizon | max_prob, min_prob, horizon_agreement, trend_alignment | 4 |
| Portfolio state | equity ratio, drawdown, open positions, win rate, loss streak, etc. | 10 |
| Coin markers | cap tier, vol class, age, is_BTC | 4 |
| **Total** | | **30** |

XGBoost already digests the raw market features into probabilities. The RL evaluator sees ML outputs + portfolio context only.

The RL evaluator continues to output:
- confidence (0-1): gate for taking the trade (> 0.3 threshold)
- atr_multiplier (0.5-4.0): stop-loss distance
- rr_ratio (1.0-5.0): reward-to-risk ratio

Online PPO learning continues: every closed trade feeds P&L as reward.

## Engine Integration

### New signal path in BacktestEngine

```python
engine = BacktestEngine(
    candles,
    ml_signal_generator=generator,
    signal_evaluator=rl_evaluator,
    online_learning=True,
)
```

Every `SIGNAL_EVERY = 6` bars (30 minutes on 5m candles — canonical value across all engine implementations):
1. `ml_signal_generator.predict(...)` → 12 probabilities (6 up + 6 down)
2. Determine direction: compute `net_up = mean(prob_up_30m..72h)` and `net_down = mean(prob_down_30m..72h)`. If `max(net_up, net_down) < 0.4`: skip (no signal). Else direction = LONG if `net_up > net_down`, SHORT otherwise. The 12 raw probabilities are passed to the RL evaluator regardless — it learns which horizons matter
3. Build RL observation from ML signal + portfolio state
4. `signal_evaluator.evaluate(obs)` → confidence + params
5. If `take_trade`: open position with RL-adjusted stops

### What stays

- Regime detection (provides 6 features to ML model; QUIET regime gate remains as a hard risk filter — if the market is in QUIET regime, no signals are generated regardless of ML output, preventing overtrading in low-volatility conditions)
- Risk management (fee gate, position sizing, drawdown scaling)
- RL Signal Evaluator (updated observation space)
- Online PPO learning
- Stop/TP management (RL adjusts ATR multiplier + R:R)
- Pre-training pipeline for RL evaluator

### Fallback behavior

If no trained ML models exist in `models/ml_signals/`, the bot refuses to start and logs an error. There is no fallback to the old strategy-based signal path — the ML models must be trained first via `scripts/train_ml_signals.py`. This is intentional: the old strategies have negative expectancy, so running them as a fallback provides no value.

### What is removed from signal path

- 8 hand-coded strategies (kept as code reference)
- StrategyRouter
- MTF Voter (ML handles multi-horizon natively)
- Confluence checker (ML replaces multi-strategy agreement)

## File Structure

### New files

```
bot/learning/ml_signal_generator.py    — MLSignalGenerator class (predict interface)
bot/learning/ml_features.py            — Feature engineering (tabular + LSTM sequence)
bot/learning/lstm_embedder.py          — LSTM model, forward pass, save/load
scripts/train_ml_signals.py            — Training pipeline (data → LSTM → XGBoost → save)
models/ml_signals/                     — Saved models directory
```

### Modified files

```
bot/backtest/engine.py                 — Add ml_signal_generator, new signal path
bot/learning/rl_signal_evaluator.py    — Update OBS_DIM to 30, new observation layout, remove STRATEGY_IDS and old build_signal_obs()
bot/learning/gpu_backtest_kernel.py    — Update SIGNAL_EVERY from 12 to 6, update signal generation to use ML predictions
bot/learning/gpu_vec_env.py            — Update _OBS_DIM from 32 to 30, new observation layout
bot/learning/rl_environment.py         — Update obs_dim from 32 to 30, new observation building
bot/main.py                            — Wire MLSignalGenerator into live loop
bot/trading_loop.py                    — Use MLSignalGenerator instead of StrategyRouter
scripts/rl_vs_champion.py             — Add ML signal generator test mode
scripts/pretrain_signal_evaluator.py   — Update to use ML signal observations instead of strategy-based observations
```

### Unchanged

```
bot/strategy/*                         — Kept (code reference only, not used in signal path)
bot/indicators/*                       — Kept (used by ml_features.py)
bot/risk/*                             — Unchanged
bot/data/*                             — Unchanged
bot/learning/walk_forward.py           — Unchanged
```

## Dependencies

New packages required:
- `xgboost` — gradient-boosted trees (new dependency, add to requirements.txt)
- `torch` — LSTM training and inference. Already installed as a transitive dependency of stable-baselines3. No additional install needed, but add explicit `torch>=2.0` to requirements.txt to make the dependency visible

## Deployment Flow

1. Run `scripts/train_ml_signals.py` — trains LSTM + 12 XGBoost models (up/down × 6 horizons), saves to `models/ml_signals/`
2. Run `scripts/pretrain_signal_evaluator.py` — pre-trains RL evaluator on historical ML signals
3. Bot startup: `MLSignalGenerator` loads models, `RLSignalEvaluator` loads pre-trained weights
4. Live: ML generates signals → RL gates and adjusts → trades execute → online learning
5. Weekly retraining (scheduled task):
   - Re-fetch latest candle data (append to existing dataset)
   - Retrain LSTM on full dataset (including new data)
   - Retrain 12 XGBoost models with updated embeddings
   - Validate retrained models: each must achieve >52% accuracy on held-out test window (last 30 days). If any model fails, keep the previous version for that model and log a warning
   - Re-pretrain RL evaluator on new ML signals
   - Hot-swap validated models at next SIGNAL_EVERY boundary (no downtime)
