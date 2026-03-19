"""Train ML signal generator models: Transformer + 12 XGBoost classifiers.

Pipeline:
    1. Fetch 5 years of 5m candles per pair
    2. Fetch external data from 12+ APIs
    3. Compute labels (horizon-scaled thresholds)
    4. Optuna HPO: Transformer architecture search (20 trials)
    5. Train Transformer (self-supervised next-bar prediction)
    6. Generate embeddings
    7. Threshold search per horizon (maximize F1)
    8. Optuna HPO: XGBoost hyperparameters (30 shared + 10 per-model)
    9. Train 12 XGBoost models (up/down x 6 horizons) with class weights
    10. Report F1, precision, recall, AUC-ROC per model
    11. Save all models to models/ml_signals/
"""
import sys, os, json, time
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from bot.exchange.bitvavo_client import BitvavoClient
from bot.data.external_features import ExternalDataProvider
from bot.learning.transformer_embedder import TransformerEmbedder, train_transformer
from bot.learning.ml_features import (
    extract_all_features, build_lstm_sequence, N_TABULAR, LSTM_WINDOW, LSTM_CHANNELS,
    LSTM_MIN_BARS,
)

PAIRS = [
    "BTC-EUR", "ETH-EUR", "XRP-EUR", "SOL-EUR", "ADA-EUR",
    "DOGE-EUR", "LINK-EUR", "AVAX-EUR", "DOT-EUR", "POL-EUR",
    "SHIB-EUR", "UNI-EUR", "LTC-EUR", "ATOM-EUR", "NEAR-EUR",
    "FIL-EUR", "ARB-EUR", "OP-EUR", "APT-EUR", "SUI-EUR",
    "PEPE-EUR", "INJ-EUR", "FET-EUR", "RENDER-EUR", "TIA-EUR",
]
YEARS = 5
MODEL_DIR = "models/ml_signals"

HORIZONS = [
    ("30m",  6,    0.15),
    ("1h",   12,   0.3),
    ("4h",   48,   0.8),
    ("12h",  144,  1.5),
    ("24h",  288,  2.5),
    ("72h",  864,  4.0),
]

# Transformer training params
TRANSFORMER_EPOCHS = 50
TRANSFORMER_BATCH = 256
TRANSFORMER_LR = 1e-3
TRANSFORMER_PREDICT_BARS = 12
LSTM_PREDICT_BARS = TRANSFORMER_PREDICT_BARS  # alias — build_lstm_targets() uses this name

# XGBoost base params (HPO will override most of these)
XGB_BASE_PARAMS = {
    "objective": "binary:logistic",
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
            targets[:, step * 5 + ch] = np.clip(
                (future[:, ch] - ref_close) / (ref_close + 1e-12), -0.20, 0.20
            )
        targets[:, step * 5 + 4] = np.clip(
            future[:, 4] / (ref_vol + 1e-12), 0.0, 5.0
        )

    return targets


def optuna_transformer_hpo(train_seqs, train_targets, val_seqs, val_targets, n_trials=20):
    """Search for best Transformer architecture using Optuna with MedianPruner."""
    import optuna

    def objective(trial):
        d_model = trial.suggest_categorical("d_model", [32, 64, 128])
        nhead = trial.suggest_categorical("nhead", [2, 4, 8])
        if d_model % nhead != 0:
            raise optuna.TrialPruned()
        n_layers = trial.suggest_categorical("n_layers", [2, 3, 4, 6])
        dropout = trial.suggest_categorical("dropout", [0.05, 0.1, 0.15, 0.2, 0.3])
        lr = trial.suggest_categorical("lr", [5e-4, 1e-3, 2e-3])

        batch_size = TRANSFORMER_BATCH
        if d_model >= 128:
            batch_size = batch_size // 2

        model = TransformerEmbedder(d_model=d_model, nhead=nhead, num_layers=n_layers, dropout=dropout)

        try:
            import torch
            losses = train_transformer(
                model, train_seqs, train_targets,
                val_seqs=val_seqs, val_targets=val_targets,
                epochs=15, batch_size=batch_size, lr=lr,
                trial=trial,
            )
            # Batched validation to avoid OOM
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
            vX_cpu = torch.from_numpy(val_seqs).float()
            vY_cpu = torch.from_numpy(val_targets).float()
            val_loss_sum = 0.0
            val_n = 0
            with torch.no_grad():
                for vs in range(0, len(vX_cpu), batch_size):
                    vx = vX_cpu[vs:vs + batch_size].to(device)
                    vy = vY_cpu[vs:vs + batch_size].to(device)
                    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                        vp = model.predict_next(vx)
                        val_loss_sum += float(torch.nn.functional.mse_loss(vp, vy).item()) * len(vx)
                    val_n += len(vx)
            return val_loss_sum / max(val_n, 1)
        except (RuntimeError, Exception) as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                import torch
                torch.cuda.empty_cache()
                raise optuna.TrialPruned()
            raise

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )
    study.optimize(objective, n_trials=n_trials)

    print(f"  Best Transformer params: {study.best_params}")
    print(f"  Best val MSE: {study.best_value:.6f}")
    return study.best_params


def search_thresholds(X, all_dfs, processed_pairs, train_mask, val_mask, horizons):
    """Search for optimal thresholds per horizon that maximize validation F1."""
    from bot.learning.ml_features import MIN_BARS_5M, SIGNAL_EVERY

    optimal = {}
    for h_idx, (h_name, bars, base_threshold) in enumerate(horizons):
        best_f1 = -1
        best_thresh = base_threshold
        for factor in np.linspace(0.5, 1.5, 10):
            thresh_pct = base_threshold * factor
            thresh_frac = thresh_pct / 100.0

            all_y_up = []
            for pair in processed_pairs:
                df = all_dfs[pair]
                close = df["close"].values.astype(np.float64)
                n = len(close)
                y_up_full = np.full(n, np.nan, dtype=np.float32)
                valid = n - bars
                if valid > 0:
                    future_close = close[bars:bars + valid]
                    current_close = close[:valid]
                    future_return = (future_close - current_close) / (current_close + 1e-12)
                    y_up_full[:valid] = (future_return > thresh_frac).astype(np.float32)
                sample_indices = np.arange(MIN_BARS_5M, n, SIGNAL_EVERY)
                all_y_up.append(y_up_full[sample_indices - 1])

            y_up = np.concatenate(all_y_up)

            train_valid = train_mask & ~np.isnan(y_up)
            val_valid = val_mask & ~np.isnan(y_up)

            if train_valid.sum() < 100 or val_valid.sum() < 100:
                continue

            X_t, y_t = X[train_valid], y_up[train_valid]
            X_v, y_v = X[val_valid], y_up[val_valid]

            quick_model = xgb.XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                tree_method="hist", device="cuda", eval_metric="logloss",
                early_stopping_rounds=20,
            )
            quick_model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
            preds = quick_model.predict(X_v)
            f1 = f1_score(y_v, preds, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh_pct

        optimal[h_name] = best_thresh
        print(f"    {h_name}: best_threshold={best_thresh:.3f}% (F1={best_f1:.3f})")

    return optimal


def optuna_xgboost_hpo(X_train, y_train, X_val, y_val, n_trials=30,
                       scale_pos_weight=1.0, seed_params=None):
    """Search for best XGBoost hyperparameters using Optuna."""
    import optuna

    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "device": "cuda",
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
            "scale_pos_weight": scale_pos_weight,
            "early_stopping_rounds": 50,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        return f1_score(y_val, preds, zero_division=0)

    study = optuna.create_study(direction="maximize")

    if seed_params:
        study.enqueue_trial({
            k: seed_params[k] for k in (
                "max_depth", "learning_rate", "n_estimators",
                "min_child_weight", "subsample", "colsample_bytree",
                "reg_alpha", "reg_lambda",
            ) if k in seed_params
        })

    study.optimize(objective, n_trials=n_trials)

    print(f"  Best XGBoost params: max_depth={study.best_params['max_depth']}, "
          f"lr={study.best_params['learning_rate']:.4f}, "
          f"n_est={study.best_params['n_estimators']}")
    print(f"  Best val F1: {study.best_value:.4f}")
    return study.best_params


def main():
    print("=" * 80)
    print("  ML SIGNAL GENERATOR TRAINING PIPELINE v2")
    print("  Transformer + External Data + Optuna HPO + Class Weights")
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

    # Step 2: Fetch external data
    print("\n  Fetching external data from 12+ APIs...")
    ext_provider = ExternalDataProvider(cache_dir="data/external_cache")
    btc_df = all_dfs.get("BTC-EUR")
    if btc_df is not None:
        idx_5m = btc_df.index
        if idx_5m.tz is None:
            idx_5m = idx_5m.tz_localize("UTC")
        ext_features_full = ext_provider.build_training_features(idx_5m)
        print(f"  External features: {ext_features_full.shape}")
    else:
        ext_features_full = None

    # Step 3: Build Transformer training data
    print("\n  Building Transformer training data...")
    STRIDE = 6
    WINDOW_PAD = 300

    # Pass 1: count total sequences to pre-allocate arrays (avoids MemoryError)
    total_count = 0
    pair_counts = {}
    for pair, df in all_dfs.items():
        n_pair = len(df) - TRANSFORMER_PREDICT_BARS
        count = 0
        for i in range(LSTM_MIN_BARS, n_pair, STRIDE):
            count += 1
        pair_counts[pair] = count
        total_count += count
    print(f"  Pre-allocating arrays for ~{total_count:,} sequences...")

    # Pre-allocate contiguous arrays
    seqs_arr = np.zeros((total_count, 96, 7), dtype=np.float32)
    targets_arr = np.zeros((total_count, TRANSFORMER_PREDICT_BARS * 5), dtype=np.float32)
    write_idx = 0
    valid_count = 0

    # Pass 2: fill arrays in-place
    for pair, df in all_dfs.items():
        targets = build_lstm_targets(df)
        n_pair = len(df) - TRANSFORMER_PREDICT_BARS
        count_before = valid_count
        for i in range(LSTM_MIN_BARS, n_pair, STRIDE):
            window_start = max(0, i + 1 - WINDOW_PAD)
            seq = build_lstm_sequence(df.iloc[window_start:i + 1])
            if not np.all(seq == 0):
                seqs_arr[valid_count] = seq
                targets_arr[valid_count] = targets[i]
                valid_count += 1
        print(f"    {pair}: {valid_count - count_before:,} sequences ({valid_count:,} total)")

    # Trim to actual valid count
    seqs_arr = seqs_arr[:valid_count]
    targets_arr = targets_arr[:valid_count]
    print(f"  Total Transformer samples: {len(seqs_arr):,}")

    val_cutoff = int(len(seqs_arr) * 0.88)
    train_seqs, val_seqs = seqs_arr[:val_cutoff], seqs_arr[val_cutoff:]
    train_targets, val_targets = targets_arr[:val_cutoff], targets_arr[val_cutoff:]
    print(f"  Transformer train: {len(train_seqs):,}, val: {len(val_seqs):,}")

    # Step 4: Optuna Transformer HPO
    print("\n  Running Optuna Transformer HPO (20 trials)...")
    t0 = time.time()
    best_transformer_params = optuna_transformer_hpo(
        train_seqs, train_targets, val_seqs, val_targets, n_trials=20,
    )
    print(f"  HPO completed in {time.time() - t0:.0f}s")

    # Step 5: Train best Transformer
    import torch, gc
    torch.cuda.empty_cache()
    gc.collect()

    print("\n  Training Transformer with best params...")
    transformer = TransformerEmbedder(
        d_model=best_transformer_params["d_model"],
        nhead=best_transformer_params["nhead"],
        num_layers=best_transformer_params["n_layers"],
        dropout=best_transformer_params["dropout"],
    )
    t0 = time.time()
    final_batch = TRANSFORMER_BATCH
    if best_transformer_params["d_model"] >= 128:
        final_batch = TRANSFORMER_BATCH // 2

    train_transformer(
        transformer, train_seqs, train_targets,
        val_seqs=val_seqs, val_targets=val_targets,
        epochs=TRANSFORMER_EPOCHS, batch_size=final_batch,
        lr=best_transformer_params["lr"],
    )
    print(f"  Transformer trained in {time.time() - t0:.0f}s")
    transformer.save(str(model_dir / "transformer.pt"))

    # Step 6: Extract features + embeddings for XGBoost
    print("\n  Extracting tabular features + Transformer embeddings...")
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer.to(device)
    transformer.eval()

    all_tabular = []
    all_labels = []
    all_embeddings = []
    all_timestamps_flat = []
    processed_pairs = []

    for pair, df in all_dfs.items():
        print(f"    {pair}: extracting features...", end="", flush=True)
        t0 = time.time()

        pair_ext = None
        if ext_features_full is not None and pair == "BTC-EUR":
            pair_ext = ext_features_full
        elif ext_features_full is not None:
            pair_idx = df.index
            if pair_idx.tz is None:
                pair_idx = pair_idx.tz_localize("UTC")
            pair_ext = ext_provider.build_training_features(pair_idx)

        tabular, sequences, timestamps = extract_all_features(
            df, symbol=pair,
            btc_df_5m=btc_df if pair != "BTC-EUR" else None,
            external_features=pair_ext,
        )
        if len(tabular) == 0:
            print(" skipped (no features)")
            continue

        processed_pairs.append(pair)

        labels = compute_labels(df)
        ts_indices = [df.index.get_loc(ts) for ts in timestamps]
        label_rows = labels[ts_indices]

        EMB_BATCH = 4096
        emb_parts = []
        with torch.no_grad():
            for eb_start in range(0, len(sequences), EMB_BATCH):
                chunk = torch.from_numpy(sequences[eb_start:eb_start + EMB_BATCH]).float().to(device)
                emb_parts.append(transformer.embed(chunk).cpu().numpy())
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

    # Step 7: Walk-forward split
    all_ts = pd.DatetimeIndex(all_timestamps_flat)
    total_span = (all_ts.max() - all_ts.min()).days
    train_cutoff = all_ts.min() + timedelta(days=int(total_span * 0.6))
    val_cutoff_dt = all_ts.min() + timedelta(days=int(total_span * 0.8))

    train_mask = all_ts < train_cutoff
    val_mask = (all_ts >= train_cutoff) & (all_ts < val_cutoff_dt)
    test_mask = all_ts >= val_cutoff_dt

    print(f"  Walk-forward split: train={train_mask.sum():,}, "
          f"val={val_mask.sum():,}, test={test_mask.sum():,}")

    # Step 8: Threshold search
    print("\n  Searching optimal thresholds per horizon...")
    optimal_thresholds = search_thresholds(X, all_dfs, processed_pairs, train_mask, val_mask, HORIZONS)

    # Step 8b: Recompute labels with optimal thresholds
    print("  Recomputing labels with optimal thresholds...")
    OPT_HORIZONS = [
        (name, bars, optimal_thresholds.get(name, thresh))
        for name, bars, thresh in HORIZONS
    ]
    from bot.learning.ml_features import MIN_BARS_5M, SIGNAL_EVERY
    all_labels = []
    for pair in processed_pairs:
        df = all_dfs[pair]
        n = len(df)
        close = df["close"].values.astype(np.float64)
        labels = np.full((n, 12), np.nan, dtype=np.float32)
        for h_idx, (name, bars, threshold) in enumerate(OPT_HORIZONS):
            threshold_frac = threshold / 100.0
            valid = n - bars
            if valid <= 0:
                continue
            future_close = close[bars:bars + valid]
            current_close = close[:valid]
            future_return = (future_close - current_close) / (current_close + 1e-12)
            labels[:valid, h_idx * 2] = (future_return > threshold_frac).astype(np.float32)
            labels[:valid, h_idx * 2 + 1] = (future_return < -threshold_frac).astype(np.float32)

        sample_indices = np.arange(MIN_BARS_5M, n, SIGNAL_EVERY)
        all_labels.append(labels[sample_indices - 1])

    y_all = np.vstack(all_labels)
    print(f"  Labels recomputed with optimal thresholds for {len(y_all):,} samples")

    # Step 9: Optuna XGBoost HPO (shared baseline on 4h_up)
    print("\n  Running Optuna XGBoost HPO (30 shared trials on 4h_up)...")
    h_idx_4h = 2
    col_4h_up = h_idx_4h * 2
    y_4h = y_all[:, col_4h_up]
    tv = train_mask & ~np.isnan(y_4h)
    vv = val_mask & ~np.isnan(y_4h)
    n_pos = y_4h[tv].sum()
    n_neg = tv.sum() - n_pos
    spw = n_neg / max(n_pos, 1)

    shared_xgb_params = optuna_xgboost_hpo(
        X[tv], y_4h[tv], X[vv], y_4h[vv],
        n_trials=30, scale_pos_weight=spw,
    )

    # Step 10: Train 12 XGBoost models
    print("\n  Training 12 XGBoost models (shared params + 10 fine-tune trials each)...")
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

            n_pos = float(y_train.sum())
            n_neg = float(len(y_train) - n_pos)
            scale_pos_weight = n_neg / max(n_pos, 1)

            fine_tuned = optuna_xgboost_hpo(
                X_train, y_train, X_val, y_val,
                n_trials=10, scale_pos_weight=scale_pos_weight,
                seed_params=shared_xgb_params,
            )

            model_params = {
                **XGB_BASE_PARAMS,
                **{k: fine_tuned[k] for k in fine_tuned
                   if k in ("max_depth", "learning_rate", "n_estimators",
                            "min_child_weight", "subsample", "colsample_bytree",
                            "reg_alpha", "reg_lambda")},
                "scale_pos_weight": scale_pos_weight,
            }

            model = xgb.XGBClassifier(**model_params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            test_pred = model.predict(X_test)
            test_prob = model.predict_proba(X_test)[:, 1]

            acc = float(np.mean(test_pred == y_test))
            pos_rate = float(y_test.mean())
            f1 = f1_score(y_test, test_pred, zero_division=0)
            prec = precision_score(y_test, test_pred, zero_division=0)
            rec = recall_score(y_test, test_pred, zero_division=0)
            try:
                auc = roc_auc_score(y_test, test_prob)
            except ValueError:
                auc = 0.0

            fi = model.feature_importances_
            top_feat = int(fi.argmax())
            max_fi = float(fi.max())
            importances[f"{h_name}_{d_name}"] = fi.tolist()

            print(f"    xgb_{h_name}_{d_name}: F1={f1:.3f}, prec={prec:.3f}, "
                  f"rec={rec:.3f}, acc={acc:.3f}, AUC={auc:.3f}, "
                  f"pos_rate={pos_rate:.3f}, spw={scale_pos_weight:.1f}, "
                  f"top_feat={top_feat}({max_fi:.1%})")

            model.save_model(str(model_dir / f"xgb_{h_name}_{d_name}.json"))

    (model_dir / "feature_importances.json").write_text(json.dumps(importances, indent=2))

    # Step 11: Save config
    config = {
        "n_tabular": N_TABULAR,
        "n_embed": 16,
        "horizons": [h[0] for h in HORIZONS],
        "thresholds": {h[0]: optimal_thresholds.get(h[0], h[2]) for h in HORIZONS},
        "pairs_trained": list(all_dfs.keys()),
        "transformer_params": best_transformer_params,
        "xgb_shared_params": shared_xgb_params,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    (model_dir / "feature_config.json").write_text(json.dumps(config, indent=2))

    print(f"\n  All models saved to {MODEL_DIR}/")
    print("  Done!")


if __name__ == "__main__":
    main()
