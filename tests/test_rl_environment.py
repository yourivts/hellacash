"""Tests for TradingParamEnv -- RL environment wrapping BacktestEngine."""
import numpy as np
import math
from unittest.mock import MagicMock, patch
from bot.learning.rl_environment import TradingParamEnv, action_to_params
from bot.backtest.engine import BacktestResult


def test_action_to_params_shape_non_range():
    """action_to_params maps 21-element action to param dict."""
    action = np.zeros(21, dtype=np.float32)
    params = action_to_params(action, strategy="squeeze")
    assert isinstance(params, dict)
    assert "atr_multiplier" in params
    assert "range_max_hold_hours" not in params
    assert len(params) == 21  # 21 params for non-range


def test_action_to_params_shape_range():
    """action_to_params maps 22-element action to param dict with range_max_hold_hours."""
    action = np.zeros(22, dtype=np.float32)
    params = action_to_params(action, strategy="range")
    assert "range_max_hold_hours" in params
    assert len(params) == 22


def test_action_to_params_bounds():
    """Actions at -1 and +1 produce correct parameter bounds."""
    # All -1: should produce minimum values
    action_min = np.full(21, -1.0, dtype=np.float32)
    p_min = action_to_params(action_min, strategy="squeeze")
    assert abs(p_min["atr_multiplier"] - 1.0) < 0.01
    assert abs(p_min["rr_ratio"] - 1.0) < 0.01

    # All +1: should produce maximum values
    action_max = np.full(21, 1.0, dtype=np.float32)
    p_max = action_to_params(action_max, strategy="squeeze")
    assert abs(p_max["atr_multiplier"] - 15.0) < 0.01
    assert abs(p_max["rr_ratio"] - 8.0) < 0.01


def test_action_to_params_integer_params():
    """Integer parameters are rounded correctly."""
    action = np.full(21, 0.0, dtype=np.float32)  # midpoint
    params = action_to_params(action, strategy="orderflow")
    assert isinstance(params["max_hold_hours"], int)
    assert isinstance(params["max_concurrent_positions"], int)
    assert isinstance(params["consecutive_confirms"], int)


def test_compute_reward_basic():
    """Reward computation produces reasonable values."""
    env = TradingParamEnv.__new__(TradingParamEnv)
    result = BacktestResult(
        total_pnl=500.0,
        sharpe_ratio=1.5,
        profit_factor=2.0,
        profit_per_fee=3.0,
        max_drawdown_pct=10.0,
        total_trades=20,
        win_rate=55.0,
    )
    reward = env._compute_reward(result)
    assert reward > 0  # good result should be positive


def test_compute_reward_penalties():
    """Reward applies penalties for bad results."""
    env = TradingParamEnv.__new__(TradingParamEnv)

    # High drawdown penalty
    result_dd = BacktestResult(
        total_pnl=100.0, sharpe_ratio=0.5, profit_factor=1.5,
        profit_per_fee=1.0, max_drawdown_pct=25.0, total_trades=10, win_rate=50.0,
    )
    reward_dd = env._compute_reward(result_dd)

    # Too few trades penalty
    result_few = BacktestResult(
        total_pnl=100.0, sharpe_ratio=0.5, profit_factor=1.5,
        profit_per_fee=1.0, max_drawdown_pct=5.0, total_trades=2, win_rate=50.0,
    )
    reward_few = env._compute_reward(result_few)

    # Low win rate penalty
    result_wr = BacktestResult(
        total_pnl=100.0, sharpe_ratio=0.5, profit_factor=1.5,
        profit_per_fee=1.0, max_drawdown_pct=5.0, total_trades=10, win_rate=15.0,
    )
    reward_wr = env._compute_reward(result_wr)

    # No penalties baseline
    result_ok = BacktestResult(
        total_pnl=100.0, sharpe_ratio=0.5, profit_factor=1.5,
        profit_per_fee=1.0, max_drawdown_pct=5.0, total_trades=10, win_rate=50.0,
    )
    reward_ok = env._compute_reward(result_ok)

    assert reward_dd < reward_ok   # drawdown penalty
    assert reward_few < reward_ok  # too-few-trades penalty
    assert reward_wr < reward_ok   # low win rate penalty


def test_reset_returns_valid_observation():
    """reset() returns observation with correct shape and range."""
    mock_store = MagicMock()
    mock_store.get_date_range.return_value = None
    env = TradingParamEnv(mock_store, ["BTC-EUR"], "squeeze")
    obs, info = env.reset()
    assert obs.shape == (27,)
    assert obs.dtype == np.float32
    assert np.all(obs >= -1.0) and np.all(obs <= 1.0)
    assert isinstance(info, dict)


def test_step_returns_correct_tuple():
    """step() returns (obs, reward, terminated, truncated, info)."""
    mock_store = MagicMock()
    mock_store.get_date_range.return_value = None
    env = TradingParamEnv(mock_store, ["BTC-EUR"], "squeeze")
    env.reset()
    action = env.action_space.sample()
    result = env.step(action)
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert obs.shape == (27,)
    assert isinstance(reward, float)
    assert terminated is True   # single-step episode
    assert truncated is False
    assert isinstance(info, dict)


def test_reward_for_default_backtest_result():
    """Reward for default BacktestResult (0 trades) is negative (penalty)."""
    env = TradingParamEnv.__new__(TradingParamEnv)
    reward = env._compute_reward(BacktestResult())
    assert reward < 0  # penalty for < 3 trades
