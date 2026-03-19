"""Integration test: train tiny model -> predict -> verify valid params.

This tests the full train-save-load-predict pipeline end-to-end.
"""
import os
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from bot.learning.rl_environment import TradingParamEnv, action_to_params
from bot.learning.rl_optimizer import RLOptimizer
from bot.strategy.adopted_universe import CHAMPION_DEFAULTS


def _make_mock_candle_store():
    """Create a mock CandleStore that returns synthetic candle data."""
    store = MagicMock()
    # Provide enough data for a 90-day window
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 6, 1, tzinfo=timezone.utc)
    store.get_date_range.return_value = (start, end)
    store.available_symbols.return_value = ["BTC-EUR"]

    n = 1440 * 90  # 90 days of 1m candles
    timestamps = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    np.random.seed(42)
    prices = 50000.0 * np.exp(np.cumsum(np.random.normal(0, 0.0001, n)))

    df_1m = pd.DataFrame({
        "open": prices, "high": prices * 1.001, "low": prices * 0.999,
        "close": prices, "volume": np.random.exponential(100, n),
    }, index=timestamps)

    df_5m = df_1m.resample("5min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()

    def get_candles(symbol, s, e, resample="5m"):
        if resample == "1m":
            return df_1m.loc[s:e]
        return df_5m.loc[s:e]

    store.get_candles.side_effect = get_candles
    return store


@pytest.mark.slow
def test_train_predict_pipeline(tmp_path):
    """Train a tiny PPO model, save, load via RLOptimizer, predict valid params."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    store = _make_mock_candle_store()
    model_dir = str(tmp_path / "models")
    os.makedirs(model_dir)

    # Train a tiny model (multi-step: n_segments=3 = 3 steps per episode)
    env = DummyVecEnv([lambda: TradingParamEnv(store, ["BTC-EUR"], "squeeze",
                                                n_segments=3)])
    model = PPO("MlpPolicy", env, n_steps=33, batch_size=33, verbose=0)
    model.learn(total_timesteps=66)

    # Save model
    model_path = os.path.join(model_dir, "squeeze_ppo.zip")
    model.save(model_path)
    env.close()

    # Load via RLOptimizer and predict
    optimizer = RLOptimizer(model_dir=model_dir)
    optimizer.load_models()
    assert optimizer.has_model("squeeze")

    features = np.random.randn(27).astype(np.float32)
    features = np.clip(features, -1.0, 1.0)
    params = optimizer.predict(features, "squeeze")

    # Verify all expected params are present and within bounds (squeeze ranges)
    assert isinstance(params, dict)
    assert params["atr_multiplier"] >= 2.0 and params["atr_multiplier"] <= 6.0
    assert params["rr_ratio"] >= 2.0 and params["rr_ratio"] <= 5.0
    assert params["base_risk_pct"] >= 1.0 and params["base_risk_pct"] <= 4.0
    assert "range_max_hold_hours" not in params  # squeeze, not range
    assert isinstance(params["max_hold_hours"], int)
    assert isinstance(params["consecutive_confirms"], int)


def test_action_to_params_all_strategies_all_bounds():
    """All action values produce valid parameter ranges for all strategies."""
    from bot.learning.rl_environment import get_param_table
    for strategy in ["orderflow", "range", "squeeze", "funding_contrarian"]:
        table = get_param_table(strategy)
        n = len(table)
        for val in [-1.0, 0.0, 1.0]:
            action = np.full(n, val, dtype=np.float32)
            params = action_to_params(action, strategy)
            # Verify all params are within their strategy-specific bounds
            for name, lo, hi, is_int in table:
                assert params[name] >= lo - 0.01, f"{strategy}/{name}: {params[name]} < {lo}"
                assert params[name] <= hi + 0.01, f"{strategy}/{name}: {params[name]} > {hi}"
            if strategy == "range":
                assert "range_max_hold_hours" in params


def test_optimizer_fallback_without_models():
    """RLOptimizer returns CHAMPION_DEFAULTS when no models exist."""
    opt = RLOptimizer(model_dir="/nonexistent_dir")
    opt.load_models()
    result = opt.predict(np.zeros(27, dtype=np.float32), "squeeze")
    assert result == CHAMPION_DEFAULTS
