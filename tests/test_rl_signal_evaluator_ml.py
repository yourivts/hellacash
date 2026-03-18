"""Tests for ML-updated RL signal evaluator observation space."""
import numpy as np
import pytest


class TestBuildMLSignalObs:
    def test_output_shape_is_30(self):
        from bot.learning.rl_signal_evaluator import build_ml_signal_obs
        obs = build_ml_signal_obs(
            probabilities=[0.6, 0.3, 0.55, 0.35, 0.7, 0.2, 0.65, 0.25, 0.5, 0.4, 0.45, 0.55],
        )
        assert obs.shape == (30,), f"Expected (30,), got {obs.shape}"

    def test_output_dtype(self):
        from bot.learning.rl_signal_evaluator import build_ml_signal_obs
        obs = build_ml_signal_obs(probabilities=[0.5] * 12)
        assert obs.dtype == np.float32

    def test_probabilities_in_obs(self):
        from bot.learning.rl_signal_evaluator import build_ml_signal_obs
        probs = [0.8, 0.1, 0.7, 0.2, 0.6, 0.3, 0.5, 0.4, 0.9, 0.05, 0.75, 0.15]
        obs = build_ml_signal_obs(probabilities=probs)
        for i in range(12):
            assert abs(obs[i] - probs[i]) < 0.01

    def test_cross_horizon_features(self):
        from bot.learning.rl_signal_evaluator import build_ml_signal_obs
        probs = [0.9, 0.1, 0.7, 0.2, 0.6, 0.3, 0.5, 0.4, 0.8, 0.05, 0.75, 0.15]
        obs = build_ml_signal_obs(probabilities=probs)
        assert obs[12] == pytest.approx(0.9, abs=0.01)   # max prob
        assert obs[13] == pytest.approx(0.05, abs=0.01)  # min prob

    def test_coin_markers_in_obs(self):
        from bot.learning.rl_signal_evaluator import build_ml_signal_obs
        obs_btc = build_ml_signal_obs(probabilities=[0.5] * 12, symbol="BTC-EUR")
        obs_eth = build_ml_signal_obs(probabilities=[0.5] * 12, symbol="ETH-EUR")
        assert not np.array_equal(obs_btc[-4:], obs_eth[-4:])

    def test_obs_dim_constant_is_30(self):
        from bot.learning.rl_signal_evaluator import OBS_DIM
        assert OBS_DIM == 30


class TestEvaluatorWithNewObs:
    def test_evaluate_with_30_dim_obs(self):
        from bot.learning.rl_signal_evaluator import RLSignalEvaluator, build_ml_signal_obs
        evaluator = RLSignalEvaluator()
        obs = build_ml_signal_obs(probabilities=[0.6, 0.3] * 6)
        result = evaluator.evaluate(obs)
        assert 0 <= result.confidence <= 1
        assert 0.5 <= result.atr_multiplier <= 4.0
        assert 1.0 <= result.rr_ratio <= 5.0

    def test_save_load_with_new_dims(self, tmp_path):
        from bot.learning.rl_signal_evaluator import RLSignalEvaluator, build_ml_signal_obs
        evaluator = RLSignalEvaluator()
        obs = build_ml_signal_obs(probabilities=[0.7, 0.2] * 6)
        result1 = evaluator.evaluate(obs, deterministic=True)

        path = str(tmp_path / "test_model.npz")
        evaluator.save(path)

        evaluator2 = RLSignalEvaluator()
        evaluator2.load(path)
        result2 = evaluator2.evaluate(obs, deterministic=True)

        assert abs(result1.confidence - result2.confidence) < 1e-5
