"""RandomForest regime classifier trained on 1h candle features."""
from __future__ import annotations

import logging
import os
import pickle
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_PATH = "./models/regime_classifier.pkl"


class RegimeClassifier:
    """
    Classifies market regime (trending / ranging / volatile) from 1h candle features.
    Trained weekly using labeled data derived from ADX/ATR thresholds.
    """

    def __init__(self, model_dir: str = "./models") -> None:
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self._model = None
        self._load()

    def _load(self) -> None:
        path = os.path.join(self.model_dir, "regime_classifier.pkl")
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    self._model = pickle.load(f)
                logger.info("Regime classifier loaded from %s", path)
            except Exception as e:
                logger.warning("Failed to load regime classifier: %s", e)

    def _save(self) -> None:
        path = os.path.join(self.model_dir, "regime_classifier.pkl")
        with open(path, "wb") as f:
            pickle.dump(self._model, f)

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        from bot.indicators.trend import adx
        from bot.indicators.volatility import atr

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        a = adx(high, low, close).iloc[-1]
        atr_val = atr(high, low, close).iloc[-1]
        price = close.iloc[-1]
        atr_pct = atr_val / price * 100 if price > 0 else 0

        price_range = (high.iloc[-20:].max() - low.iloc[-20:].min()) / price * 100
        vol_change = volume.iloc[-5:].mean() / volume.iloc[-20:-5].mean() if volume.iloc[-20:-5].mean() > 0 else 1.0

        return np.array([[a, atr_pct, price_range, vol_change]])

    def predict(self, df_1h: pd.DataFrame) -> str:
        """Predict regime from 1h candles. Falls back to rule-based if no model."""
        from bot.strategy.router import detect_regime
        if self._model is None or len(df_1h) < 30:
            return detect_regime(df_1h)

        try:
            X = self._extract_features(df_1h)
            label = self._model.predict(X)[0]
            regime_map = {0: "ranging", 1: "trending", 2: "volatile"}
            return regime_map.get(label, "unknown")
        except Exception:
            from bot.strategy.router import detect_regime
            return detect_regime(df_1h)

    def train(self, candle_records: List[pd.DataFrame]) -> None:
        """Train on a list of 1h candle DataFrames with auto-labeling."""
        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError:
            logger.warning("scikit-learn not available; regime classifier not trained")
            return

        from bot.indicators.trend import adx
        from bot.indicators.volatility import atr

        X_list, y_list = [], []
        for df in candle_records:
            if len(df) < 30:
                continue
            try:
                feats = self._extract_features(df)
                # Auto-label based on ADX/ATR thresholds
                a_val = feats[0][0]
                atr_pct = feats[0][1]
                if atr_pct > 3.0:
                    label = 2  # volatile
                elif a_val > 25:
                    label = 1  # trending
                else:
                    label = 0  # ranging
                X_list.append(feats[0])
                y_list.append(label)
            except Exception:
                continue

        if len(X_list) < 10:
            logger.warning("Not enough data to train regime classifier (%d samples)", len(X_list))
            return

        X = np.array(X_list)
        y = np.array(y_list)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)
        self._model = clf
        self._save()
        logger.info("Regime classifier trained on %d samples", len(X_list))
