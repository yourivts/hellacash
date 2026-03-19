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
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from bot.exchange.bitvavo_client import BitvavoClient
from bot.config import get_settings
from bot.data.candle_store import CandleStore
from bot.data.external_features import ExternalDataProvider
from bot.learning.transformer_embedder import TransformerEmbedder, train_transformer
from bot.learning.ml_features import (
    extract_all_features, build_lstm_sequence, build_lstm_sequences_batch,
    N_TABULAR, LSTM_WINDOW, LSTM_CHANNELS, LSTM_MIN_BARS,
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
TRANSFORMER_BATCH = 2048
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
        d_model = trial.suggest_categorical("d_model", [64, 128])
        nhead = trial.suggest_categorical("nhead", [2, 4, 8])
        if d_model % nhead != 0:
            raise optuna.TrialPruned()
        n_layers = trial.suggest_categorical("n_layers", [2, 3, 4])
        dropout = trial.suggest_categorical("dropout", [0.1, 0.15, 0.2])
        lr = trial.suggest_categorical("lr", [1e-3, 2e-3])

        # Scale batch size by model size to avoid OOM on 8GB GPU
        batch_size = TRANSFORMER_BATCH
        if d_model >= 128 and n_layers >= 4:
            batch_size = TRANSFORMER_BATCH // 4
        elif d_model >= 128:
            batch_size = TRANSFORMER_BATCH // 2

        model = TransformerEmbedder(d_model=d_model, nhead=nhead, num_layers=n_layers, dropout=dropout)

        try:
            import torch
            losses = train_transformer(
                model, train_seqs, train_targets,
                val_seqs=val_seqs, val_targets=val_targets,
                epochs=8, batch_size=batch_size, lr=lr,
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
        except BaseException as e:
            import torch
            torch.cuda.empty_cache()
            err_msg = str(e).lower()
            if "out of memory" in err_msg or "cuda" in err_msg or "accelerator" in err_msg:
                raise optuna.TrialPruned()
            raise

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=2, reduction_factor=3),
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-transformer", action="store_true",
                        help="Skip Transformer HPO+training, load saved model")
    args = parser.parse_args()

    print("=" * 80)
    print("  ML SIGNAL GENERATOR TRAINING PIPELINE v2")
    print("  Transformer + External Data + Optuna HPO + Class Weights")
    print("=" * 80)

    model_dir = Path(MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load candles from PostgreSQL (1m -> resample to 5m)
    #   First top-up DB with the last few days from Bitvavo API
    settings = get_settings()
    candle_store = CandleStore(db_url=settings.database_url)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=YEARS * 365)

    import asyncio
    from bot.exchange.bitvavo_client import _api_get, _API_BASE

    async def _topup():
        """Ensure all pairs have data in DB, then fetch recent candles."""
        print("\n  Syncing candle data to DB...")
        loop = asyncio.get_running_loop()
        now_ms = int(time.time() * 1000)
        for pair in PAIRS:
            last_ts = candle_store._get_last_timestamp(pair)
            if last_ts is None:
                # Full download for pairs with no DB data
                print(f"    {pair}: no data in DB, downloading full history...", flush=True)
                await candle_store._download_symbol(pair)
                span = candle_store._get_data_span_days(pair)
                print(f"    {pair}: downloaded {span or 0} days")
                last_ts = candle_store._get_last_timestamp(pair)
                if last_ts is None:
                    continue
            # Top-up: fetch from last stored to now
            cursor_ms = int(last_ts.timestamp() * 1000) + 60_000
            stored = 0
            while cursor_ms < now_ms:
                page_end = min(cursor_ms + 1440 * 60_000, now_ms)
                url = (f"{_API_BASE}/{pair}/candles?interval=1m"
                       f"&start={cursor_ms}&end={page_end}&limit=1440")
                try:
                    candles = await loop.run_in_executor(None, _api_get, url)
                except Exception:
                    break
                if not candles:
                    break
                candle_store._store_candles(pair, candles)
                stored += len(candles)
                newest = max(c[0] for c in candles)
                cursor_ms = newest + 60_000
                await asyncio.sleep(0.2)
            if stored > 0:
                print(f"    {pair}: +{stored:,} new candles")
        candle_store.clear_cache()

    asyncio.run(_topup())
    print("  Top-up complete.")

    all_dfs = {}
    for pair in PAIRS:
        print(f"\n  Loading {pair} from DB...", end="", flush=True)
        df = candle_store.get_candles(pair, start, end, resample="5m")
        if df.empty or len(df) < 10000:
            print(f" skipped (insufficient data: {len(df)})")
            continue
        print(f" {len(df):,} candles")
        all_dfs[pair] = df

    if not all_dfs:
        print("  ERROR: No data loaded from DB. Exiting.")
        sys.exit(1)

    # Step 2: Fetch external data
    print("\n  Fetching external data from 12+ APIs...")
    ext_provider = ExternalDataProvider(cache_dir="data/external_cache", db_url=settings.database_url)
    btc_df = all_dfs.get("BTC-EUR")
    if btc_df is not None:
        idx_5m = btc_df.index
        if idx_5m.tz is None:
            idx_5m = idx_5m.tz_localize("UTC")
        ext_features_full = ext_provider.build_training_features(idx_5m)
        print(f"  External features: {ext_features_full.shape}")
    else:
        ext_features_full = None

    import torch, gc

    if args.skip_transformer:
        # Load existing Transformer model
        print("\n  Loading pre-trained Transformer (--skip-transformer)...")
        # Load config to get architecture params
        config_path = model_dir / "feature_config.json"
        if config_path.exists():
            saved_config = json.loads(config_path.read_text())
            best_transformer_params = saved_config.get("transformer_params", {})
        else:
            best_transformer_params = {"d_model": 128, "nhead": 2, "n_layers": 3, "dropout": 0.15, "lr": 0.001}
        transformer = TransformerEmbedder(
            d_model=best_transformer_params.get("d_model", 128),
            nhead=best_transformer_params.get("nhead", 2),
            num_layers=best_transformer_params.get("n_layers", 3),
            dropout=best_transformer_params.get("dropout", 0.15),
        )
        transformer.load(str(model_dir / "transformer.pt"))
        print(f"  Loaded Transformer: {best_transformer_params}")
    else:
        # Step 3: Build Transformer training data
        print("\n  Building Transformer training data...")
        STRIDE = 6
        WINDOW_PAD = 300

        # Pass 1: count total sequences to pre-allocate arrays (avoids MemoryError)
        total_count = 0
        pair_indices = {}
        for pair, df in all_dfs.items():
            n_pair = len(df) - TRANSFORMER_PREDICT_BARS
            indices = np.arange(LSTM_MIN_BARS, n_pair, STRIDE)
            pair_indices[pair] = indices
            total_count += len(indices)
        print(f"  Pre-allocating arrays for ~{total_count:,} sequences...")

        # Pre-allocate contiguous arrays
        seqs_arr = np.zeros((total_count, 96, 7), dtype=np.float32)
        targets_arr = np.zeros((total_count, TRANSFORMER_PREDICT_BARS * 5), dtype=np.float32)
        valid_count = 0

        # Pass 2: batch-build sequences per pair (indicators computed once per pair)
        for pair, df in all_dfs.items():
            targets = build_lstm_targets(df)
            indices = pair_indices[pair]

            # Batch build all sequences for this pair at once
            batch_seqs = build_lstm_sequences_batch(df, indices)

            # Filter out all-zero sequences
            nonzero_mask = np.any(batch_seqs != 0, axis=(1, 2))
            n_valid = int(nonzero_mask.sum())
            count_before = valid_count

            seqs_arr[valid_count:valid_count + n_valid] = batch_seqs[nonzero_mask]
            targets_arr[valid_count:valid_count + n_valid] = targets[indices[nonzero_mask]]
            valid_count += n_valid
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
        print("\n  Running Optuna Transformer HPO (12 trials)...")
        t0 = time.time()
        best_transformer_params = optuna_transformer_hpo(
            train_seqs, train_targets, val_seqs, val_targets, n_trials=12,
        )
        print(f"  HPO completed in {time.time() - t0:.0f}s")

        # Step 5: Train best Transformer
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
        d = best_transformer_params["d_model"]
        nl = best_transformer_params["n_layers"]
        if d >= 128 and nl >= 4:
            final_batch = TRANSFORMER_BATCH // 4
        elif d >= 128:
            final_batch = TRANSFORMER_BATCH // 2

        # OOM retry: if batch_size OOMs, halve and retry
        for attempt in range(3):
            try:
                train_transformer(
                    transformer, train_seqs, train_targets,
                    val_seqs=val_seqs, val_targets=val_targets,
                    epochs=TRANSFORMER_EPOCHS, batch_size=final_batch,
                    lr=best_transformer_params["lr"],
                )
                break
            except BaseException as e:
                torch.cuda.empty_cache()
                if "out of memory" in str(e).lower() or "cuda" in str(e).lower() or "accelerator" in str(e).lower():
                    final_batch = final_batch // 2
                    print(f"  OOM — retrying with batch_size={final_batch} (attempt {attempt + 2}/3)")
                    transformer = TransformerEmbedder(
                        d_model=best_transformer_params["d_model"],
                        nhead=best_transformer_params["nhead"],
                        num_layers=best_transformer_params["n_layers"],
                        dropout=best_transformer_params["dropout"],
                    )
                else:
                    raise
        print(f"  Transformer trained in {time.time() - t0:.0f}s")
        transformer.save(str(model_dir / "transformer.pt"))

        # Free Transformer training data (~4GB) before feature extraction
        del seqs_arr, targets_arr, train_seqs, val_seqs, train_targets, val_targets
        del pair_indices
        gc.collect()

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

    # Pre-build external features aligned to each pair's index (reindex once, not re-fetch)
    ext_cache = {}
    if ext_features_full is not None:
        btc_idx = btc_df.index if btc_df.index.tz is not None else btc_df.index.tz_localize("UTC")
        ext_df = pd.DataFrame(ext_features_full, index=btc_idx)
        for pair, df in all_dfs.items():
            pair_idx = df.index if df.index.tz is not None else df.index.tz_localize("UTC")
            ext_cache[pair] = ext_df.reindex(pair_idx, method="ffill").values.astype(np.float32)

    for pair, df in all_dfs.items():
        print(f"    {pair}: extracting features...", end="", flush=True)
        t0 = time.time()

        pair_ext = ext_cache.get(pair)

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

    # Step 10: Train 12 XGBoost models (up/down pairs in parallel per horizon)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    print_lock = threading.Lock()

    print("\n  Training 12 XGBoost models (shared params + 10 fine-tune trials each)...")
    importances = {}

    def _train_one_xgb(h_idx, h_name, bars, d_idx, d_name):
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
        spw = n_neg / max(n_pos, 1)

        fine_tuned = optuna_xgboost_hpo(
            X_train, y_train, X_val, y_val,
            n_trials=10, scale_pos_weight=spw,
            seed_params=shared_xgb_params,
        )

        model_params = {
            **XGB_BASE_PARAMS,
            **{k: fine_tuned[k] for k in fine_tuned
               if k in ("max_depth", "learning_rate", "n_estimators",
                        "min_child_weight", "subsample", "colsample_bytree",
                        "reg_alpha", "reg_lambda")},
            "scale_pos_weight": spw,
        }

        mdl = xgb.XGBClassifier(**model_params)
        mdl.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        test_pred = mdl.predict(X_test)
        test_prob = mdl.predict_proba(X_test)[:, 1]

        acc = float(np.mean(test_pred == y_test))
        pos_rate = float(y_test.mean())
        f1 = f1_score(y_test, test_pred, zero_division=0)
        prec = precision_score(y_test, test_pred, zero_division=0)
        rec = recall_score(y_test, test_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, test_prob)
        except ValueError:
            auc = 0.0

        fi = mdl.feature_importances_
        top_feat = int(fi.argmax())
        max_fi = float(fi.max())

        mdl.save_model(str(model_dir / f"xgb_{h_name}_{d_name}.json"))

        with print_lock:
            print(f"    xgb_{h_name}_{d_name}: F1={f1:.3f}, prec={prec:.3f}, "
                  f"rec={rec:.3f}, acc={acc:.3f}, AUC={auc:.3f}, "
                  f"pos_rate={pos_rate:.3f}, spw={spw:.1f}, "
                  f"top_feat={top_feat}({max_fi:.1%})")

        return f"{h_name}_{d_name}", fi.tolist()

    # Run up/down pairs in parallel (2 at a time to share GPU)
    futures = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for h_idx, (h_name, bars, threshold) in enumerate(HORIZONS):
            for d_idx, d_name in enumerate(["up", "down"]):
                futures.append(pool.submit(
                    _train_one_xgb, h_idx, h_name, bars, d_idx, d_name
                ))
        for fut in as_completed(futures):
            key, fi_list = fut.result()
            importances[key] = fi_list

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
