"""RLOptimizer — load trained PPO models and predict parameters."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import numpy as np

from bot.learning.rl_environment import action_to_params
from bot.strategy.adopted_universe import CHAMPION_DEFAULTS

logger = logging.getLogger(__name__)


class RLOptimizer:
    """Load trained PPO models and predict trading parameters from market features."""

    def __init__(self, model_dir: str = "models/rl_optimizer") -> None:
        self._model_dir = model_dir
        self._models: Dict[str, Any] = {}  # strategy -> PPO model

    def load_models(self) -> None:
        """Load all strategy models from disk."""
        from stable_baselines3 import PPO

        if not os.path.isdir(self._model_dir):
            logger.warning("RLOptimizer: model dir %s not found", self._model_dir)
            return

        loaded = 0
        for fname in os.listdir(self._model_dir):
            if fname.endswith("_ppo.zip") and not fname.endswith(".tmp.zip"):
                strategy = fname.replace("_ppo.zip", "")
                path = os.path.join(self._model_dir, fname)
                try:
                    self._models[strategy] = PPO.load(path)
                    loaded += 1
                    logger.info("Loaded RL model for %s", strategy)
                except Exception as e:
                    logger.warning("Failed to load model %s: %s", path, e)

        if loaded == 0:
            logger.warning("RLOptimizer: no models loaded, will use CHAMPION_DEFAULTS")

    def predict(self, features: np.ndarray, strategy: str) -> Dict[str, Any]:
        """Predict parameters from 27-element feature vector.

        Returns parameter dict. Falls back to CHAMPION_DEFAULTS if model not loaded.
        """
        model = self._models.get(strategy)
        if model is None:
            logger.debug("No model for %s, using CHAMPION_DEFAULTS", strategy)
            return dict(CHAMPION_DEFAULTS)

        try:
            action, _ = model.predict(features, deterministic=True)
            return action_to_params(action, strategy)
        except Exception as e:
            logger.warning("Prediction failed for %s: %s", strategy, e)
            return dict(CHAMPION_DEFAULTS)

    def has_model(self, strategy: str) -> bool:
        """Check if a model is loaded for the given strategy."""
        return strategy in self._models
