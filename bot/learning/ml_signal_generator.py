"""ML Signal Generator: load trained models and predict directional probabilities.

Loads:
    - LSTM embedder (lstm.pt)
    - 12 XGBoost classifiers (xgb_{horizon}_{direction}.json)
    - Feature config (feature_config.json)

Exposes:
    predict(df_5m, symbol) → dict with probabilities, direction, net_up, net_down
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import xgboost as xgb

from bot.learning.lstm_embedder import LSTMEmbedder
from bot.learning.ml_features import extract_tabular_features, build_lstm_sequence

logger = logging.getLogger(__name__)

HORIZONS = ["30m", "1h", "4h", "12h", "24h", "72h"]
DIRECTIONS = ["up", "down"]
MIN_SIGNAL_PROB = 0.4  # minimum average probability to generate a direction signal


class MLSignalGenerator:
    """Load trained LSTM + XGBoost models and generate trading signals."""

    def __init__(self, model_dir: str = "models/ml_signals") -> None:
        self._model_dir = Path(model_dir)
        if not self._model_dir.exists():
            raise FileNotFoundError(
                f"ML model directory not found: {model_dir}. "
                f"Run scripts/train_ml_signals.py first."
            )

        # Load LSTM
        lstm_path = self._model_dir / "lstm.pt"
        if not lstm_path.exists():
            raise FileNotFoundError(f"LSTM model not found: {lstm_path}")
        self._lstm = LSTMEmbedder()
        self._lstm.load(str(lstm_path))
        self._lstm.eval()

        # Load XGBoost models
        self._xgb_models: Dict[str, xgb.XGBClassifier] = {}
        for h in HORIZONS:
            for d in DIRECTIONS:
                key = f"{h}_{d}"
                path = self._model_dir / f"xgb_{key}.json"
                if not path.exists():
                    raise FileNotFoundError(f"XGBoost model not found: {path}")
                model = xgb.XGBClassifier()
                model.load_model(str(path))
                self._xgb_models[key] = model

        # Load feature config
        config_path = self._model_dir / "feature_config.json"
        if config_path.exists():
            with open(config_path) as f:
                self._config = json.load(f)
        else:
            self._config = {"n_tabular": 65, "n_lstm_embed": 16, "horizons": HORIZONS}

        logger.info(
            "MLSignalGenerator loaded: LSTM + %d XGBoost models from %s",
            len(self._xgb_models), model_dir,
        )

    def predict(
        self,
        df_5m: "pd.DataFrame",
        symbol: str = "BTC-EUR",
        btc_df_5m: Optional["pd.DataFrame"] = None,
        **live_features,
    ) -> Dict[str, Any]:
        """Generate predictions for the current market state.

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

        # Extract features — supply zero defaults for live market data
        tabular = extract_tabular_features(
            df_5m,
            symbol,
            btc_df_5m,
            live_features.get("funding_rate", 0.0),
            live_features.get("funding_score", 0.0),
            live_features.get("ob_imbalance", 0.0),
            live_features.get("spread_pct", 0.0),
            live_features.get("bid_ask_wall_ratio", 0.0),
            live_features.get("onchain_composite", 0.0),
            live_features.get("exchange_reserve_trend", 0.0),
            int(live_features.get("regime_id", 0)),
            live_features.get("regime_hours", 0.0),
        )
        seq = build_lstm_sequence(df_5m)

        # Check for insufficient data (all zeros)
        if np.all(tabular == 0) or np.all(seq == 0):
            return default

        # Get LSTM embedding
        embedding = self._lstm.embed_numpy(seq)  # (16,)

        # Concatenate: 65 tabular + 16 embedding = 81 features
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
        # probabilities layout: [up_30m, down_30m, up_1h, down_1h, ...]
        up_probs = [probabilities[i] for i in range(0, 12, 2)]  # indices 0,2,4,6,8,10
        down_probs = [probabilities[i] for i in range(1, 12, 2)]  # indices 1,3,5,7,9,11
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
