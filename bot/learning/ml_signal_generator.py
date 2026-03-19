"""ML Signal Generator: load trained models and predict directional probabilities.

Loads:
    - Transformer embedder (transformer.pt, fallback to lstm.pt)
    - 12 XGBoost classifiers (xgb_{horizon}_{direction}.json)
    - Feature config (feature_config.json)

Exposes:
    predict(df_5m, symbol) -> dict with probabilities, direction, net_up, net_down
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import xgboost as xgb

from bot.learning.ml_features import extract_tabular_features, build_lstm_sequence

logger = logging.getLogger(__name__)

HORIZONS = ["30m", "1h", "4h", "12h", "24h", "72h"]
DIRECTIONS = ["up", "down"]
MIN_SIGNAL_PROB = 0.4  # minimum average probability to generate a direction signal


class MLSignalGenerator:
    """Load trained embedder + XGBoost models and generate trading signals."""

    def __init__(self, model_dir: str = "models/ml_signals") -> None:
        self._model_dir = Path(model_dir)
        self._disabled = False  # Set by failsafe system

        if not self._model_dir.exists():
            raise FileNotFoundError(
                f"ML model directory not found: {model_dir}. "
                f"Run scripts/train_ml_signals.py first."
            )

        # Load embedder: prefer Transformer, fall back to LSTM
        transformer_path = self._model_dir / "transformer.pt"
        lstm_path = self._model_dir / "lstm.pt"

        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if transformer_path.exists():
            from bot.learning.transformer_embedder import TransformerEmbedder
            # Read architecture params from feature_config to match saved weights
            config_path = self._model_dir / "feature_config.json"
            tp = {}
            if config_path.exists():
                with open(config_path) as f:
                    tp = json.load(f).get("transformer_params", {})
            self._embedder = TransformerEmbedder(
                d_model=tp.get("d_model", 64),
                nhead=tp.get("nhead", 4),
                num_layers=tp.get("n_layers", 4),
                dropout=tp.get("dropout", 0.1),
            )
            self._embedder.load(str(transformer_path))
            self._embedder.to(device)
            self._embedder.eval()
            logger.info("Loaded Transformer embedder on %s from %s", device, transformer_path)
        elif lstm_path.exists():
            from bot.learning.lstm_embedder import LSTMEmbedder
            self._embedder = LSTMEmbedder()
            self._embedder.load(str(lstm_path))
            self._embedder.to(device)
            self._embedder.eval()
            logger.info("Loaded LSTM embedder (fallback) on %s from %s", device, lstm_path)
        else:
            raise FileNotFoundError(
                f"No embedder model found. Expected transformer.pt or lstm.pt in {model_dir}"
            )

        # Load XGBoost models (GPU inference if available)
        self._xgb_models: Dict[str, xgb.XGBClassifier] = {}
        for h in HORIZONS:
            for d in DIRECTIONS:
                key = f"{h}_{d}"
                path = self._model_dir / f"xgb_{key}.json"
                if not path.exists():
                    raise FileNotFoundError(f"XGBoost model not found: {path}")
                model = xgb.XGBClassifier()
                model.load_model(str(path))
                # Enable GPU prediction if model was trained on GPU
                try:
                    model.set_params(device="cuda")
                except Exception:
                    pass  # Fall back to CPU prediction if GPU not available
                self._xgb_models[key] = model

        # Load feature config (accept both n_embed and n_lstm_embed for backward compat)
        config_path = self._model_dir / "feature_config.json"
        if config_path.exists():
            with open(config_path) as f:
                self._config = json.load(f)
        else:
            self._config = {"n_tabular": 90, "n_embed": 16, "horizons": HORIZONS}

        # Accept both key names
        if "n_embed" not in self._config and "n_lstm_embed" in self._config:
            self._config["n_embed"] = self._config["n_lstm_embed"]

        logger.info(
            "MLSignalGenerator loaded: embedder + %d XGBoost models from %s",
            len(self._xgb_models), model_dir,
        )

    def predict(
        self,
        df_5m: "pd.DataFrame",
        symbol: str = "BTC-EUR",
        btc_df_5m: Optional["pd.DataFrame"] = None,
        external_data: dict | None = None,
        **live_features,
    ) -> Dict[str, Any]:
        """Generate predictions for the current market state.

        Args:
            df_5m: Recent 5m candles (at least MIN_BARS_5M bars)
            symbol: Trading pair symbol
            btc_df_5m: BTC candles for cross-asset features
            external_data: Dict of external feature values (from ExternalDataProvider)
            **live_features: Additional live features (regime_id, regime_hours)

        Returns:
            dict with keys:
                probabilities: list of 12 floats [up_30m, down_30m, up_1h, down_1h, ...]
                direction: "LONG", "SHORT", or None (no signal)
                net_up: float (mean of up probabilities across horizons)
                net_down: float (mean of down probabilities across horizons)
        """
        default = {
            "probabilities": [0.5] * 12,
            "direction": None,
            "net_up": 0.5,
            "net_down": 0.5,
        }

        # Failsafe: if ML signals are disabled, return no signal
        if self._disabled:
            return default

        # Build external_data dict from explicit parameter
        ext = dict(external_data or {})

        regime_id = int(live_features.get("regime_id", 0))
        regime_hours = live_features.get("regime_hours", 0.0)

        # Extract features
        tabular = extract_tabular_features(
            df_5m, symbol, btc_df_5m,
            external_data=ext,
            regime_id=regime_id,
            regime_hours=regime_hours,
        )
        seq = build_lstm_sequence(df_5m)

        # Check for insufficient data
        if np.all(tabular == 0) or np.all(seq == 0):
            return default

        # Get embedding (works for both Transformer and LSTM — same interface)
        embedding = self._embedder.embed_numpy(seq)  # (16,)

        # Concatenate: 90 tabular + 16 embedding = 106 features
        combined = np.concatenate([tabular, embedding]).reshape(1, -1)

        # Run all 12 XGBoost models
        probabilities = []
        for h in HORIZONS:
            for d in DIRECTIONS:
                key = f"{h}_{d}"
                model = self._xgb_models[key]
                prob = float(model.predict_proba(combined)[0, 1])
                probabilities.append(prob)

        # Compute net direction
        up_probs = [probabilities[i] for i in range(0, 12, 2)]
        down_probs = [probabilities[i] for i in range(1, 12, 2)]
        net_up = float(np.mean(up_probs))
        net_down = float(np.mean(down_probs))

        # Direction decision
        direction = None
        if max(net_up, net_down) >= MIN_SIGNAL_PROB:
            direction = "LONG" if net_up > net_down else "SHORT"

        return {
            "probabilities": probabilities,
            "direction": direction,
            "net_up": net_up,
            "net_down": net_down,
        }
