"""Tests that new RL-tunable parameters affect BacktestEngine behavior."""
from bot.backtest.engine import BacktestEngine


def _make_candles(n=2000, base_price=100.0):
    """Generate simple candle data as list of dicts."""
    import numpy as np
    from datetime import datetime, timedelta, timezone
    np.random.seed(42)
    candles = []
    price = base_price
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        ret = np.random.normal(0, 0.005)
        price *= (1 + ret)
        ts = start + timedelta(minutes=5 * i)
        candles.append({
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "open": price * 0.999,
            "high": price * 1.003,
            "low": price * 0.997,
            "close": price,
            "volume": 1000.0,
            "symbol": "TEST-EUR",
        })
    return candles


def test_signal_strength_min_filters_weak_signals():
    candles = _make_candles()
    r_low = BacktestEngine(candles, strategy_params={"signal_strength_min": 0.1}).run()
    r_high = BacktestEngine(candles, strategy_params={"signal_strength_min": 0.9}).run()
    assert r_high.total_trades <= r_low.total_trades


def test_max_position_pct_caps_size():
    candles = _make_candles()
    r_big = BacktestEngine(candles, strategy_params={"max_position_pct": 0.5}).run()
    r_small = BacktestEngine(candles, strategy_params={"max_position_pct": 0.1}).run()
    if r_big.total_trades > 0 and r_small.total_trades > 0:
        avg_big = sum(t.size_eur for t in r_big.trade_log) / len(r_big.trade_log)
        avg_small = sum(t.size_eur for t in r_small.trade_log) / len(r_small.trade_log)
        assert avg_small <= avg_big


def test_ema200_filter_disabled_at_zero():
    candles = _make_candles()
    r_strict = BacktestEngine(candles, strategy_params={"ema200_filter_pct": 2.0}).run()
    r_disabled = BacktestEngine(candles, strategy_params={"ema200_filter_pct": 0.0}).run()
    assert r_disabled.total_trades >= r_strict.total_trades


def test_volatile_atr_threshold_affects_regime():
    candles = _make_candles()
    r_low = BacktestEngine(candles, strategy_params={"volatile_atr_threshold": 1.0}).run()
    r_high = BacktestEngine(candles, strategy_params={"volatile_atr_threshold": 10.0}).run()
    assert r_high.total_trades >= r_low.total_trades


def test_drawdown_scale_pct_default():
    engine = BacktestEngine([], strategy_params={"drawdown_scale_pct": 5.0})
    assert engine._drawdown_scale_pct == 5.0


def test_trail_activation_mult_default():
    engine = BacktestEngine([], strategy_params={"trail_activation_mult": 2.5})
    assert engine._trail_activation_mult == 2.5


def test_confluence_boost_default():
    engine = BacktestEngine([], strategy_params={"confluence_boost": 1.5})
    assert engine._confluence_boost == 1.5


def test_confidence_size_scaling_default():
    engine = BacktestEngine([], strategy_params={"confidence_size_scaling": 1.0})
    assert engine._confidence_size_scaling == 1.0


def test_range_max_hold_hours():
    engine = BacktestEngine([], strategy_params={"range_max_hold_hours": 24})
    assert engine._range_max_hold_bars == 24 * 12


def test_tf_weights_default():
    engine = BacktestEngine([], strategy_params={
        "tf_weight_1h": 0.5, "tf_weight_4h": 0.8, "tf_weight_1d": 0.3,
    })
    assert engine._tf_weight_1h == 0.5
    assert engine._tf_weight_4h == 0.8
    assert engine._tf_weight_1d == 0.3
