"""Tests for bot/learning/ml_features.py — feature engineering for ML signal generator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_5m_candles(bars: int = 2000) -> pd.DataFrame:
    np.random.seed(42)
    timestamps = pd.date_range("2024-01-01", periods=bars, freq="5min")
    base = 50000.0
    returns = np.random.randn(bars) * 0.001
    close = base * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.randn(bars)) * 0.002)
    low = close * (1 - np.abs(np.random.randn(bars)) * 0.002)
    open_ = close * (1 + np.random.randn(bars) * 0.001)
    volume = np.random.exponential(100, bars)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=timestamps,
    )


# ---------------------------------------------------------------------------
# TestExtractTabularFeatures
# ---------------------------------------------------------------------------

class TestExtractTabularFeatures:
    """Tests for extract_tabular_features()."""

    def test_output_shape_is_65(self):
        from bot.learning.ml_features import extract_tabular_features, N_TABULAR

        assert N_TABULAR == 65
        df = _make_5m_candles(2000)
        result = extract_tabular_features(df, "ETHUSDT", None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)
        assert result.shape == (65,), f"Expected (65,), got {result.shape}"

    def test_output_dtype_float32(self):
        from bot.learning.ml_features import extract_tabular_features

        df = _make_5m_candles(2000)
        result = extract_tabular_features(df, "ETHUSDT", None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)
        assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"

    def test_all_finite(self):
        from bot.learning.ml_features import extract_tabular_features

        df = _make_5m_candles(2000)
        result = extract_tabular_features(df, "ETHUSDT", None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)
        assert np.all(np.isfinite(result)), f"Non-finite values found: {result[~np.isfinite(result)]}"

    def test_different_symbols_change_coin_features(self):
        """Coin marker features (indices 51-54) should differ between BTC and a small-cap."""
        from bot.learning.ml_features import extract_tabular_features

        df = _make_5m_candles(2000)
        btc_features = extract_tabular_features(df, "BTCUSDT", None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)
        eth_features = extract_tabular_features(df, "ETHUSDT", None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)
        # At minimum is_btc flag (index 54) should differ
        assert btc_features[54] != eth_features[54], "is_btc flag should differ between BTC and ETH"

    def test_insufficient_data_returns_zeros(self):
        from bot.learning.ml_features import extract_tabular_features, MIN_BARS_5M

        # Use fewer bars than the minimum
        df = _make_5m_candles(MIN_BARS_5M - 1)
        result = extract_tabular_features(df, "ETHUSDT", None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)
        assert np.all(result == 0.0), "Should return all zeros for insufficient data"

    def test_live_features_zero_filled_funding_ob(self):
        """Funding/OB features (indices 55-59) should be zero when passed as 0."""
        from bot.learning.ml_features import extract_tabular_features

        df = _make_5m_candles(2000)
        result = extract_tabular_features(df, "ETHUSDT", None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)
        assert np.all(result[55:60] == 0.0), f"Expected funding/OB features to be 0, got {result[55:60]}"

    def test_live_features_zero_filled_onchain(self):
        """On-chain features (indices 60-61) should be zero when passed as 0."""
        from bot.learning.ml_features import extract_tabular_features

        df = _make_5m_candles(2000)
        result = extract_tabular_features(df, "ETHUSDT", None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)
        assert np.all(result[60:62] == 0.0), f"Expected on-chain features to be 0, got {result[60:62]}"

    def test_regime_one_hot_encoding(self):
        """Regime features (29-34) — one-hot + hours should have exactly one 1 in first 5 slots."""
        from bot.learning.ml_features import extract_tabular_features

        df = _make_5m_candles(2000)
        # regime_id=2 should set index 31 to 1.0 (29+2)
        result = extract_tabular_features(df, "ETHUSDT", None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2, 5.0)
        regime_one_hot = result[29:34]
        assert regime_one_hot.sum() == 1.0, f"One-hot should sum to 1, got {regime_one_hot}"
        assert result[29 + 2] == 1.0, f"Expected index 31 to be 1, got {result[29 + 2]}"

    def test_with_btc_df_populates_cross_asset(self):
        """Cross-asset features (62-64) should be non-zero when BTC df provided."""
        from bot.learning.ml_features import extract_tabular_features

        df = _make_5m_candles(2000)
        btc_df = _make_5m_candles(2000)
        result = extract_tabular_features(df, "ETHUSDT", btc_df, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)
        # BTC 1h return (index 62) should not be zero given random price data
        assert np.any(result[62:65] != 0.0), "Cross-asset features should be populated when BTC df provided"


# ---------------------------------------------------------------------------
# TestBuildLSTMSequence
# ---------------------------------------------------------------------------

class TestBuildLSTMSequence:
    """Tests for build_lstm_sequence()."""

    def test_output_shape(self):
        from bot.learning.ml_features import build_lstm_sequence, LSTM_WINDOW, LSTM_CHANNELS

        df = _make_5m_candles(2000)
        result = build_lstm_sequence(df)
        assert result.shape == (LSTM_WINDOW, LSTM_CHANNELS), (
            f"Expected ({LSTM_WINDOW}, {LSTM_CHANNELS}), got {result.shape}"
        )
        assert result.shape == (96, 7)

    def test_output_dtype_float32(self):
        from bot.learning.ml_features import build_lstm_sequence

        df = _make_5m_candles(2000)
        result = build_lstm_sequence(df)
        assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"

    def test_values_finite(self):
        from bot.learning.ml_features import build_lstm_sequence

        df = _make_5m_candles(2000)
        result = build_lstm_sequence(df)
        assert np.all(np.isfinite(result)), f"Non-finite values in LSTM sequence"

    def test_insufficient_data_returns_zeros(self):
        from bot.learning.ml_features import build_lstm_sequence, LSTM_MIN_BARS

        # Use fewer bars than the minimum
        df = _make_5m_candles(LSTM_MIN_BARS - 1)
        result = build_lstm_sequence(df)
        assert np.all(result == 0.0), "Should return all zeros for insufficient data"

    def test_close_channel_normalized_first_bar_near_zero(self):
        """Channel 0 (normalized close) should start near 0 (% from first bar)."""
        from bot.learning.ml_features import build_lstm_sequence

        df = _make_5m_candles(2000)
        result = build_lstm_sequence(df)
        # First bar of the window: pct change from itself = 0
        assert abs(result[0, 0]) < 1e-5, f"First bar normalized close should be ~0, got {result[0, 0]}"

    def test_body_direction_values(self):
        """Channel 3 (body direction) should only contain +1 or -1."""
        from bot.learning.ml_features import build_lstm_sequence

        df = _make_5m_candles(2000)
        result = build_lstm_sequence(df)
        body_dir = result[:, 3]
        valid = np.isin(body_dir, [-1.0, 1.0])
        assert np.all(valid), f"Body direction contains values other than +/-1: {np.unique(body_dir)}"

    def test_volume_channel_non_negative(self):
        """Channel 1 (normalized volume) should be >= 0."""
        from bot.learning.ml_features import build_lstm_sequence

        df = _make_5m_candles(2000)
        result = build_lstm_sequence(df)
        assert np.all(result[:, 1] >= 0.0), "Normalized volume should be non-negative"


# ---------------------------------------------------------------------------
# TestExtractAllFeatures
# ---------------------------------------------------------------------------

class TestExtractAllFeatures:
    """Tests for extract_all_features()."""

    def test_returns_aligned_arrays(self):
        """Tabular array, sequence array, and timestamps list must have same length."""
        from bot.learning.ml_features import extract_all_features

        df = _make_5m_candles(2000)
        tabular, sequences, timestamps = extract_all_features(df, "ETHUSDT", None)
        assert len(tabular) == len(sequences) == len(timestamps), (
            f"Misaligned outputs: tabular={len(tabular)}, "
            f"sequences={len(sequences)}, timestamps={len(timestamps)}"
        )

    def test_tabular_shape_n_65(self):
        from bot.learning.ml_features import extract_all_features

        df = _make_5m_candles(2000)
        tabular, sequences, timestamps = extract_all_features(df, "ETHUSDT", None)
        assert len(tabular) > 0, "Should produce at least one sample"
        assert tabular.shape[1] == 65, f"Expected 65 tabular features, got {tabular.shape[1]}"

    def test_sequence_shape_n_96_7(self):
        from bot.learning.ml_features import extract_all_features

        df = _make_5m_candles(2000)
        tabular, sequences, timestamps = extract_all_features(df, "ETHUSDT", None)
        assert len(sequences) > 0
        assert sequences.shape[1:] == (96, 7), f"Expected (N, 96, 7), got {sequences.shape}"

    def test_signal_every_spacing(self):
        """Consecutive timestamps should be SIGNAL_EVERY * 5min apart."""
        from bot.learning.ml_features import extract_all_features, SIGNAL_EVERY

        df = _make_5m_candles(2000)
        tabular, sequences, timestamps = extract_all_features(df, "ETHUSDT", None)
        if len(timestamps) >= 2:
            dt = pd.Timestamp(timestamps[1]) - pd.Timestamp(timestamps[0])
            expected_minutes = SIGNAL_EVERY * 5
            assert dt.seconds // 60 == expected_minutes, (
                f"Expected {expected_minutes}min gap, got {dt.seconds // 60}min"
            )

    def test_insufficient_data_returns_empty(self):
        """With fewer bars than MIN_BARS_5M, should return empty arrays."""
        from bot.learning.ml_features import extract_all_features, MIN_BARS_5M

        df = _make_5m_candles(100)
        tabular, sequences, timestamps = extract_all_features(df, "ETHUSDT", None)
        assert len(tabular) == 0, "Should return empty tabular array for insufficient data"
        assert len(sequences) == 0, "Should return empty sequences array for insufficient data"
        assert len(timestamps) == 0, "Should return empty timestamps list for insufficient data"

    def test_output_dtypes(self):
        from bot.learning.ml_features import extract_all_features

        df = _make_5m_candles(2000)
        tabular, sequences, timestamps = extract_all_features(df, "ETHUSDT", None)
        if len(tabular) > 0:
            assert tabular.dtype == np.float32, f"Tabular dtype should be float32, got {tabular.dtype}"
            assert sequences.dtype == np.float32, f"Sequences dtype should be float32, got {sequences.dtype}"

    def test_all_values_finite(self):
        from bot.learning.ml_features import extract_all_features

        df = _make_5m_candles(2000)
        tabular, sequences, timestamps = extract_all_features(df, "ETHUSDT", None)
        if len(tabular) > 0:
            assert np.all(np.isfinite(tabular)), "Non-finite values in tabular array"
            assert np.all(np.isfinite(sequences)), "Non-finite values in sequences array"
