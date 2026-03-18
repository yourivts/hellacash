"""Train ML signal generator models: LSTM + 12 XGBoost classifiers.

Usage:
    python scripts/train_ml_signals.py

Pipeline:
    1. Fetch 5 years of 5m candles per pair
    2. Compute labels (horizon-scaled thresholds)
    3. Train LSTM (self-supervised next-bar prediction)
    4. Generate embeddings
    5. Train 12 XGBoost models (up/down × 6 horizons)
    6. Save all models to models/ml_signals/
"""
import sys, os, json, time
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from bot.exchange.bitvavo_client import BitvavoClient
from bot.learning.lstm_embedder import LSTMEmbedder, train_lstm
from bot.learning.ml_features import (
    extract_all_features, N_TABULAR, LSTM_WINDOW, LSTM_CHANNELS,
)

PAIRS = ["BTC-EUR", "ETH-EUR", "XRP-EUR"]
YEARS = 5
MODEL_DIR = "models/ml_signals"

# Horizon definitions: (name, bars_ahead, threshold_pct)
HORIZONS = [
    ("30m",  6,    0.15),
    ("1h",   12,   0.3),
    ("4h",   48,   0.8),
    ("12h",  144,  1.5),
    ("24h",  288,  2.5),
    ("72h",  864,  4.0),
]

# LSTM training params
LSTM_EPOCHS = 50
LSTM_BATCH = 256
LSTM_LR = 1e-3
LSTM_PREDICT_BARS = 12

# XGBoost params
XGB_PARAMS = {
    "objective": "binary:logistic",
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "eval_metric": "logloss",
    "tree_method": "hist",
    "device": "cuda",
    "early_stopping_rounds": 50,
}


def compute_labels(df_5m: pd.DataFrame) -> np.ndarray:
    """Compute binary labels for all horizons (vectorized).

    Returns:
        (N, 12) array: columns are [up_30m, down_30m, up_1h, down_1h, ...]
        NaN where future data is unavailable.
    """
    close = df_5m["close"].values.astype(np.float64)
    n = len(close)
    labels = np.full((n, 12), np.nan, dtype=np.float32)

    for h_idx, (name, bars, threshold) in enumerate(HORIZONS):
        threshold_frac = threshold / 100.0
        valid = n - bars
        if valid <= 0:
            continue
        future_close = close[bars:bars + valid]
        current_close = close[:valid]
        future_return = (future_close - current_close) / (current_close + 1e-12)
        labels[:valid, h_idx * 2] = (future_return > threshold_frac).astype(np.float32)
        labels[:valid, h_idx * 2 + 1] = (future_return < -threshold_frac).astype(np.float32)

    return labels


def build_lstm_targets(df_5m: pd.DataFrame) -> np.ndarray:
    """Build LSTM prediction targets: next 12 bars normalized OHLCV (vectorized).

    Returns:
        (N-12, 60) array where each row is 12 bars × 5 channels (OHLCV).
    """
    ohlcv = df_5m[["open", "high", "low", "close", "volume"]].values.astype(np.float64)
    n = len(ohlcv)
    n_targets = n - LSTM_PREDICT_BARS
    targets = np.zeros((n_targets, LSTM_PREDICT_BARS * 5), dtype=np.float32)

    ref_close = ohlcv[:n_targets, 3]
    ref_vol = np.where(ohlcv[:n_targets, 4] > 0, ohlcv[:n_targets, 4], 1.0)

    for step in range(LSTM_PREDICT_BARS):
        future = ohlcv[step + 1:step + 1 + n_targets]
        for ch in range(4):
            targets[:, step * 5 + ch] = (future[:, ch] - ref_close) / (ref_close + 1e-12)
        targets[:, step * 5 + 4] = future[:, 4] / (ref_vol + 1e-12)

    return targets


def main():
    print("=" * 80)
    print("  ML SIGNAL GENERATOR TRAINING PIPELINE")
    print("=" * 80)

    model_dir = Path(MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Fetch candles
    all_dfs = {}
    client = BitvavoClient(api_key="", api_secret="", paper_trading=True)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=YEARS * 365)

    for pair in PAIRS:
        print(f"\n  Fetching {pair}...", end="", flush=True)
        candles = client.get_candles_range(pair, interval="5m", start=start, end=end)
        if not candles or len(candles) < 10000:
            print(f" skipped (insufficient data: {len(candles) if candles else 0})")
            continue
        df = pd.DataFrame([{
            "open": c.open, "high": c.high, "low": c.low,
            "close": c.close, "volume": c.volume,
        } for c in candles], index=[c.timestamp for c in candles])
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        print(f" {len(df):,} candles")
        all_dfs[pair] = df
        time.sleep(1)

    if not all_dfs:
        print("  ERROR: No data fetched. Exiting.")
        sys.exit(1)

    # Step 2: Build LSTM targets and train LSTM
    print("\n  Building LSTM training data...")
    all_seqs = []
    all_lstm_targets = []
    LSTM_STRIDE = 6  # sample every 6 bars (30 min) — consecutive bars have 95/96 overlap
    WINDOW_PAD = 300  # extra bars for indicator warmup (only need ~120, pad generously)
    for pair, df in all_dfs.items():
        from bot.learning.ml_features import build_lstm_sequence, LSTM_MIN_BARS
        targets = build_lstm_targets(df)
        n_pair = len(df) - LSTM_PREDICT_BARS
        count_before = len(all_seqs)
        for i in range(LSTM_MIN_BARS, n_pair, LSTM_STRIDE):
            # Pass a small window instead of growing slice — avoids O(N²)
            window_start = max(0, i + 1 - WINDOW_PAD)
            seq = build_lstm_sequence(df.iloc[window_start:i + 1])
            if not np.all(seq == 0):
                all_seqs.append(seq)
                all_lstm_targets.append(targets[i])
        print(f"    {pair}: {len(all_seqs) - count_before:,} sequences ({len(all_seqs):,} total)")

    seqs_arr = np.stack(all_seqs)
    targets_arr = np.stack(all_lstm_targets)
    print(f"  Total LSTM samples: {len(seqs_arr):,}")

    val_cutoff = int(len(seqs_arr) * 0.88)
    train_seqs, val_seqs = seqs_arr[:val_cutoff], seqs_arr[val_cutoff:]
    train_targets, val_targets = targets_arr[:val_cutoff], targets_arr[val_cutoff:]

    print(f"  LSTM train: {len(train_seqs):,}, val: {len(val_seqs):,}")
    lstm = LSTMEmbedder()
    print("  Training LSTM...")
    t0 = time.time()
    train_lstm(
        lstm, train_seqs, train_targets,
        val_seqs=val_seqs, val_targets=val_targets,
        epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH, lr=LSTM_LR,
    )
    print(f"  LSTM trained in {time.time() - t0:.0f}s")
    lstm.save(str(model_dir / "lstm.pt"))

    # Step 3: Extract features + embeddings for XGBoost
    print("\n  Extracting tabular features + LSTM embeddings...")
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm.to(device)
    lstm.eval()
    all_tabular = []
    all_labels = []
    all_embeddings = []
    all_timestamps_flat = []

    btc_df = all_dfs.get("BTC-EUR")

    for pair, df in all_dfs.items():
        print(f"    {pair}: extracting features...", end="", flush=True)
        t0 = time.time()
        tabular, sequences, timestamps = extract_all_features(
            df, symbol=pair, btc_df_5m=btc_df if pair != "BTC-EUR" else None,
        )
        if len(tabular) == 0:
            print(" skipped (no features)")
            continue

        labels = compute_labels(df)
        ts_indices = [df.index.get_loc(ts) for ts in timestamps]
        label_rows = labels[ts_indices]

        # Batch embedding extraction in chunks (GPU memory limited)
        import torch
        device = next(lstm.parameters()).device
        EMB_BATCH = 4096
        emb_parts = []
        with torch.no_grad():
            for eb_start in range(0, len(sequences), EMB_BATCH):
                chunk = torch.from_numpy(sequences[eb_start:eb_start + EMB_BATCH]).float().to(device)
                emb_parts.append(lstm.embed(chunk).cpu().numpy())
        embeddings = np.vstack(emb_parts)

        all_tabular.append(tabular)
        all_labels.append(label_rows)
        all_embeddings.append(embeddings)
        all_timestamps_flat.extend(timestamps)
        print(f" {len(tabular):,} samples ({time.time() - t0:.0f}s)")

    X_tab = np.vstack(all_tabular)
    y_all = np.vstack(all_labels)
    X_emb = np.vstack(all_embeddings)
    X = np.hstack([X_tab, X_emb])
    print(f"  Total XGBoost samples: {len(X):,}, features: {X.shape[1]}")

    # Step 4: Walk-forward split and train XGBoost
    all_ts = pd.DatetimeIndex(all_timestamps_flat)
    total_span = (all_ts.max() - all_ts.min()).days
    train_cutoff = all_ts.min() + timedelta(days=int(total_span * 0.6))
    val_cutoff_dt = all_ts.min() + timedelta(days=int(total_span * 0.8))

    train_mask = all_ts < train_cutoff
    val_mask = (all_ts >= train_cutoff) & (all_ts < val_cutoff_dt)
    test_mask = all_ts >= val_cutoff_dt

    print(f"  Walk-forward split: train={train_mask.sum():,}, "
          f"val={val_mask.sum():,}, test={test_mask.sum():,}")

    importances = {}
    for h_idx, (h_name, bars, threshold) in enumerate(HORIZONS):
        for d_idx, d_name in enumerate(["up", "down"]):
            col = h_idx * 2 + d_idx
            y = y_all[:, col]

            train_valid = train_mask & ~np.isnan(y)
            val_valid = val_mask & ~np.isnan(y)
            test_valid = test_mask & ~np.isnan(y)

            X_train, y_train = X[train_valid], y[train_valid]
            X_val, y_val = X[val_valid], y[val_valid]
            X_test, y_test = X[test_valid], y[test_valid]

            model = xgb.XGBClassifier(**XGB_PARAMS)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            test_pred = model.predict(X_test)
            acc = float(np.mean(test_pred == y_test))
            pos_rate = float(y_test.mean())

            fi = model.feature_importances_
            max_fi = float(fi.max())
            top_feat = int(fi.argmax())
            importances[f"{h_name}_{d_name}"] = fi.tolist()

            status = "OK"
            if acc < 0.52:
                status = "WARN: acc < 52%"
            if max_fi > 0.30:
                status = f"WARN: feature {top_feat} dominates ({max_fi:.1%})"

            print(f"    xgb_{h_name}_{d_name}: test_acc={acc:.3f}, "
                  f"pos_rate={pos_rate:.3f}, top_feat={top_feat}({max_fi:.1%}) [{status}]")

            model.save_model(str(model_dir / f"xgb_{h_name}_{d_name}.json"))

    (model_dir / "feature_importances.json").write_text(json.dumps(importances, indent=2))

    # Step 5: Save config
    config = {
        "n_tabular": N_TABULAR,
        "n_lstm_embed": 16,
        "horizons": [h[0] for h in HORIZONS],
        "thresholds": {h[0]: h[2] for h in HORIZONS},
        "pairs_trained": list(all_dfs.keys()),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    (model_dir / "feature_config.json").write_text(json.dumps(config, indent=2))

    print(f"\n  All models saved to {MODEL_DIR}/")
    print("  Done!")


if __name__ == "__main__":
    main()
