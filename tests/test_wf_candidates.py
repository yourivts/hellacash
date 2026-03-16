"""Test that walk-forward candidates produce differentiated backtest results."""
import numpy as np
import pandas as pd
from bot.backtest.engine import BacktestEngine


def test_atr_multiplier_is_used():
    """atr_multiplier param should flow through to the engine."""
    params = {"atr_multiplier": 5.0}
    engine = BacktestEngine([], strategy_params=params)
    assert engine._atr_multiplier == 5.0


def test_base_risk_pct_is_used():
    """base_risk_pct param should affect position sizing."""
    params = {"base_risk_pct": 1.5}
    engine = BacktestEngine([], strategy_params=params)
    assert engine._base_risk_pct == 1.5


def test_min_confirmations_is_used():
    """min_confirmations param should be stored."""
    params = {"min_confirmations": 4}
    engine = BacktestEngine([], strategy_params=params)
    assert engine._min_confirmations == 4


def test_indicator_weights_is_used():
    """indicator_weights param should be stored."""
    w = {"rsi": 0.30, "macd": 0.30}
    params = {"indicator_weights": w}
    engine = BacktestEngine([], strategy_params=params)
    assert engine._indicator_weights == w


def test_cooldown_bars_always_zero():
    """Cooldown removed — 1 position per coin, reopen immediately."""
    engine = BacktestEngine([], strategy_params={})
    assert engine._cooldown_bars == 0


def test_min_adx_is_used():
    """min_adx param should be stored."""
    params = {"min_adx": 25}
    engine = BacktestEngine([], strategy_params=params)
    assert engine._min_adx == 25


def test_min_atr_pct_is_used():
    """min_atr_pct param should be stored."""
    params = {"min_atr_pct": 0.5}
    engine = BacktestEngine([], strategy_params=params)
    assert engine._min_atr_pct == 0.5


def test_max_hold_hours_is_used():
    """max_hold_hours param should be converted to bars."""
    params = {"max_hold_hours": 72}
    engine = BacktestEngine([], strategy_params=params)
    assert engine._max_hold_bars == 72 * 12  # 12 x 5m bars per hour


def test_defaults_when_no_params():
    """Without params, defaults should match live config."""
    engine = BacktestEngine([])
    assert engine._atr_multiplier == 2.0
    assert engine._base_risk_pct == 3.0
    assert engine._min_confirmations == 2
    assert engine._indicator_weights is None
    assert engine._cooldown_bars == 0  # no cooldown
    assert engine._min_adx == 0  # disabled by default
    assert engine._min_atr_pct == 0.0  # disabled by default
    assert engine._max_hold_bars == 48 * 12  # default 48h hold
