"""Tests for RLOptimizer — inference from trained models."""
import numpy as np
from unittest.mock import MagicMock, patch
from bot.learning.rl_optimizer import RLOptimizer


def test_predict_fallback_no_model():
    """Returns CHAMPION_DEFAULTS when no model loaded."""
    from bot.strategy.adopted_universe import CHAMPION_DEFAULTS
    opt = RLOptimizer(model_dir="/nonexistent")
    result = opt.predict(np.zeros(27, dtype=np.float32), "orderflow")
    assert result == CHAMPION_DEFAULTS


def test_predict_with_mock_model():
    """Returns valid param dict when model is loaded."""
    opt = RLOptimizer.__new__(RLOptimizer)
    opt._model_dir = "/tmp"
    opt._models = {}

    # Mock a model that returns all-zeros action
    mock_model = MagicMock()
    mock_model.predict.return_value = (np.zeros(21, dtype=np.float32), None)
    opt._models["squeeze"] = mock_model

    result = opt.predict(np.zeros(27, dtype=np.float32), "squeeze")
    assert isinstance(result, dict)
    assert "atr_multiplier" in result
    # Midpoint of [1.0, 15.0] at action=0 is 8.0
    assert abs(result["atr_multiplier"] - 8.0) < 0.01


def test_predict_range_model_has_extra_param():
    """Range model returns range_max_hold_hours."""
    opt = RLOptimizer.__new__(RLOptimizer)
    opt._model_dir = "/tmp"
    opt._models = {}

    mock_model = MagicMock()
    mock_model.predict.return_value = (np.zeros(22, dtype=np.float32), None)
    opt._models["range"] = mock_model

    result = opt.predict(np.zeros(27, dtype=np.float32), "range")
    assert "range_max_hold_hours" in result
