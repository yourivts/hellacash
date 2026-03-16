"""Tests for the 27-feature market feature extractor."""
import numpy as np
import pandas as pd
from bot.learning.feature_extractor import extract_features


def _make_candles_1m(days: int = 90, base_price: float = 100.0) -> pd.DataFrame:
    """Generate synthetic 1m candle data for testing."""
    n = days * 24 * 60
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    np.random.seed(42)
    returns = np.random.normal(0.00001, 0.001, n)
    prices = base_price * np.exp(np.cumsum(returns))
    high = prices * (1 + np.abs(np.random.normal(0, 0.002, n)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.002, n)))
    volume = np.random.exponential(1000, n)
    return pd.DataFrame({
        "open": prices, "high": high, "low": low,
        "close": prices * (1 + np.random.normal(0, 0.0005, n)),
        "volume": volume,
    }, index=timestamps)


def test_extract_features_returns_27_floats():
    df = _make_candles_1m()
    features = extract_features(df)
    assert features.shape == (27,)
    assert features.dtype == np.float32


def test_features_clipped_to_bounds():
    df = _make_candles_1m()
    features = extract_features(df)
    assert np.all(features >= -1.0), f"Min: {features.min()}"
    assert np.all(features <= 1.0), f"Max: {features.max()}"


def test_features_with_btc_candles():
    df = _make_candles_1m()
    btc = _make_candles_1m(base_price=50000.0)
    features = extract_features(df, btc_candles_1m=btc)
    assert features.shape == (27,)
    assert features[25] != 0.0 or features[26] != 0.0


def test_features_without_btc_candles():
    df = _make_candles_1m()
    features = extract_features(df, btc_candles_1m=None)
    assert features[25] == 0.0
    assert features[26] == 0.0


def test_features_short_data_returns_zeros():
    n = 500
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    prices = np.linspace(100, 105, n)
    df = pd.DataFrame({
        "open": prices, "high": prices + 1, "low": prices - 1,
        "close": prices + 0.5, "volume": np.ones(n) * 100,
    }, index=timestamps)
    features = extract_features(df)
    assert features.shape == (27,)
    assert np.all(features == 0.0)


def test_features_no_nan_or_inf():
    n = 1440 * 3
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    price = 100.0
    df = pd.DataFrame({
        "open": np.full(n, price), "high": np.full(n, price),
        "low": np.full(n, price), "close": np.full(n, price),
        "volume": np.zeros(n),
    }, index=timestamps)
    features = extract_features(df)
    assert features.shape == (27,)
    assert not np.any(np.isnan(features)), f"NaN found: {features}"
    assert not np.any(np.isinf(features)), f"Inf found: {features}"
    assert np.all(features >= -1.0) and np.all(features <= 1.0)
