"""Tests for RLTrainer — validation gate and atomic save."""
import os
from unittest.mock import MagicMock, patch
from bot.learning.rl_trainer import RLTrainer


def test_trainer_init_creates_model_dir(tmp_path):
    """RLTrainer creates model directory on init."""
    model_dir = str(tmp_path / "models")
    store = MagicMock()
    trainer = RLTrainer(store, ["squeeze"], model_dir=model_dir)
    assert os.path.isdir(model_dir)


def test_trainer_skips_when_no_symbols():
    """train() returns empty dict when no candle data available."""
    store = MagicMock()
    store.available_symbols.return_value = []
    trainer = RLTrainer(store, ["squeeze"], model_dir="/tmp/rl_test")
    result = trainer.train()
    assert result == {}


def test_validate_returns_float():
    """_validate returns average reward as float."""
    store = MagicMock()
    store.get_date_range.return_value = None
    trainer = RLTrainer(store, ["squeeze"], model_dir="/tmp/rl_test")
    mock_model = MagicMock()
    mock_model.predict.return_value = (
        __import__("numpy").zeros(21, dtype=__import__("numpy").float32), None
    )
    avg = trainer._validate(mock_model, "squeeze", ["BTC-EUR"], n_episodes=2)
    assert isinstance(avg, float)
