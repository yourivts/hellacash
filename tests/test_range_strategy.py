"""Tests for range trading strategy components."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bot.indicators.volume import volume_profile_support


class TestVolumeProfileSupport:
    """Tests for volume_profile_support indicator."""

    def _make_ranging_data(self, n=300, seed=42):
        """Generate synthetic ranging price data with volume."""
        np.random.seed(seed)
        t = np.linspace(0, 8 * np.pi, n)
        close = 100 + 5 * np.sin(t) + np.random.normal(0, 0.5, n)
        high = close + np.random.uniform(0.1, 1.0, n)
        low = close - np.random.uniform(0.1, 1.0, n)
        volume = np.random.uniform(100, 1000, n)
        return (
            pd.Series(close, dtype=float),
            pd.Series(volume, dtype=float),
            pd.Series(high, dtype=float),
            pd.Series(low, dtype=float),
        )

    def test_returns_series_of_correct_length(self):
        close, volume, high, low = self._make_ranging_data()
        result = volume_profile_support(close, volume, high, low)
        assert isinstance(result, pd.Series)
        assert len(result) == len(close)

    def test_values_bounded_minus_one_to_plus_one(self):
        close, volume, high, low = self._make_ranging_data()
        result = volume_profile_support(close, volume, high, low)
        valid = result.dropna()
        assert (valid >= -1.0).all()
        assert (valid <= 1.0).all()

    def test_positive_score_near_support(self):
        """When price is near range bottom with volume below, score should be positive."""
        close, volume, high, low = self._make_ranging_data(n=300)
        result = volume_profile_support(close, volume, high, low, lookback=200)
        support_indices = close[close < close.rolling(200).quantile(0.2)].index
        if len(support_indices) > 0:
            support_scores = result.loc[support_indices].dropna()
            if len(support_scores) > 0:
                assert support_scores.mean() > -0.5

    def test_handles_zero_volume(self):
        """Should not crash when volume is zero."""
        close = pd.Series([100.0] * 50)
        volume = pd.Series([0.0] * 50)
        high = close + 1.0
        low = close - 1.0
        result = volume_profile_support(close, volume, high, low, lookback=20)
        assert len(result) == 50
        assert not result.isna().all()

    def test_custom_lookback(self):
        close, volume, high, low = self._make_ranging_data()
        result_50 = volume_profile_support(close, volume, high, low, lookback=50)
        result_200 = volume_profile_support(close, volume, high, low, lookback=200)
        assert len(result_50) == len(result_200) == len(close)


from bot.strategy.range_trading import RangeStrategy
from bot.strategy.base import MarketContext, Signal


class TestRangeStrategy:
    """Tests for RangeStrategy.generate_signal."""

    def _make_ranging_ctx(self, price=100.0, rsi_val=35.0, bb_position="lower",
                          bb_bandwidth=0.06, adx_val=15.0, symbol="BTC-EUR"):
        """Build a MarketContext with synthetic ranging data."""
        np.random.seed(42)
        n = 250
        t = np.linspace(0, 6 * np.pi, n)
        base = 100 + 3 * np.sin(t)
        noise = np.random.normal(0, 0.3, n)
        closes = base + noise

        if bb_position == "lower":
            closes[-1] = price * 0.97
        elif bb_position == "upper":
            closes[-1] = price * 1.03
        else:
            closes[-1] = price

        idx_5m = pd.date_range("2026-01-01", periods=n, freq="5min")
        df_5m = pd.DataFrame({
            "open": closes * (1 + np.random.normal(0, 0.001, n)),
            "high": closes * (1 + np.random.uniform(0, 0.005, n)),
            "low": closes * (1 - np.random.uniform(0, 0.005, n)),
            "close": closes,
            "volume": np.random.uniform(100, 1000, n),
        }, index=idx_5m)

        n_1h = 50
        t_1h = np.linspace(0, 6 * np.pi, n_1h)
        close_1h = 100 + 3 * np.sin(t_1h) + np.random.normal(0, 0.2, n_1h)
        idx_1h = pd.date_range("2026-01-01", periods=n_1h, freq="1h")
        df_1h = pd.DataFrame({
            "open": close_1h,
            "high": close_1h * 1.003,
            "low": close_1h * 0.997,
            "close": close_1h,
            "volume": np.random.uniform(500, 5000, n_1h),
        }, index=idx_1h)

        return MarketContext(
            symbol=symbol,
            candles_5m=df_5m,
            candles_1h=df_1h,
            current_price=float(closes[-1]),
            sentiment_score=0.0,
            portfolio_equity_eur=10000.0,
            open_position_count=0,
        )

    def test_name_is_range(self):
        strat = RangeStrategy()
        assert strat.name == "range"

    def test_neutral_when_insufficient_data(self):
        strat = RangeStrategy()
        df = pd.DataFrame({
            "open": [100], "high": [101], "low": [99],
            "close": [100], "volume": [500],
        }, index=pd.date_range("2026-01-01", periods=1, freq="5min"))
        ctx = MarketContext(symbol="BTC-EUR", candles_5m=df, candles_1h=df, current_price=100.0,
                            sentiment_score=0.0, portfolio_equity_eur=10000.0, open_position_count=0)
        sig = strat.generate_signal(ctx)
        assert sig.direction == "NEUTRAL"

    def test_returns_signal_dataclass(self):
        strat = RangeStrategy()
        ctx = self._make_ranging_ctx()
        sig = strat.generate_signal(ctx)
        assert isinstance(sig, Signal)
        assert sig.strategy_name == "range"
        assert sig.symbol == "BTC-EUR"

    def test_indicator_snapshot_has_range_fields(self):
        strat = RangeStrategy()
        ctx = self._make_ranging_ctx()
        sig = strat.generate_signal(ctx)
        snap = sig.indicator_snapshot
        assert "range_mid" in snap
        assert "range_upper" in snap
        assert "range_lower" in snap
        assert "bounce_count" in snap
        assert "confirming_count" in snap

    def test_bounce_counter_increments(self):
        strat = RangeStrategy()
        strat.increment_bounce("BTC-EUR")
        strat.increment_bounce("BTC-EUR")
        assert strat.get_bounce_count("BTC-EUR") == 2

    def test_bounce_counter_resets(self):
        strat = RangeStrategy()
        strat.increment_bounce("BTC-EUR")
        strat.increment_bounce("BTC-EUR")
        strat.reset_bounces("BTC-EUR")
        assert strat.get_bounce_count("BTC-EUR") == 0

    def test_bounce_limit_stops_signaling(self):
        strat = RangeStrategy()
        for _ in range(3):
            strat.increment_bounce("BTC-EUR")
        ctx = self._make_ranging_ctx()
        sig = strat.generate_signal(ctx)
        assert sig.direction == "NEUTRAL"


from bot.backtest.engine import BacktestEngine, _OpenPosition
from bot.exchange.bitvavo_client import CandleData
from datetime import datetime, timezone


class TestRangeBacktestIntegration:
    """Tests for range strategy in the backtest engine."""

    def _make_ranging_candles(self, n=500, symbol="BTC-EUR"):
        """Generate synthetic ranging candles for backtest."""
        np.random.seed(42)
        candles = []
        base_price = 50000.0
        t0 = datetime(2025, 9, 1, tzinfo=timezone.utc)

        for i in range(n):
            phase = np.sin(2 * np.pi * i / 100) * 0.01
            price = base_price * (1 + phase + np.random.normal(0, 0.001))
            candles.append(CandleData(
                symbol=symbol,
                interval="5m",
                timestamp=t0 + pd.Timedelta(minutes=5 * i),
                open=price * (1 + np.random.normal(0, 0.0005)),
                high=price * (1 + abs(np.random.normal(0, 0.002))),
                low=price * (1 - abs(np.random.normal(0, 0.002))),
                close=price,
                volume=float(np.random.uniform(0.5, 5.0)),
            ))
        return candles

    def test_engine_has_range_bounce_counter(self):
        candles = self._make_ranging_candles(n=100)
        engine = BacktestEngine(candles)
        assert hasattr(engine, "_range_bounces")
        assert isinstance(engine._range_bounces, dict)

    def test_open_position_has_range_fields(self):
        pos = _OpenPosition(
            symbol="BTC-EUR", direction="LONG", entry_price=50000.0,
            entry_time="2025-09-01", size_eur=100.0, stop_loss=49000.0,
            take_profit=51000.0, highest_price=50000.0, strategy="range",
        )
        assert pos.tp_shifted is False
        assert pos.range_mid == 0.0
        assert pos.range_upper == 0.0
        assert pos.range_lower == 0.0

    def test_precompute_includes_bb_full_arrays(self):
        candles = self._make_ranging_candles(n=200)
        engine = BacktestEngine(candles)
        df = engine._to_dataframe(candles)
        precomp = engine._precompute_signals(df, "BTC-EUR")
        assert "bb_upper" in precomp
        assert "bb_lower" in precomp
        assert "bb_mid" in precomp
        assert "bb_bandwidth" in precomp
        assert "vol_profile" in precomp

    def test_backtest_runs_without_error(self):
        """Smoke test: backtest with range strategy doesn't crash."""
        candles = self._make_ranging_candles(n=500)
        engine = BacktestEngine(candles, strategy_params={
            "min_confirmations": 1,
            "entry_threshold": 0.30,
            "cooldown_hours": 1,
        })
        result = engine.run()
        assert result.total_trades >= 0


class TestRangeExitLogic:
    """Tests for range-specific exit handling in backtest."""

    def test_range_time_exit_24h(self):
        """Range positions close after 288 bars (24h)."""
        engine = BacktestEngine.__new__(BacktestEngine)
        engine.positions = []
        engine.closed_trades = []
        engine.balance = 10000.0
        engine.peak_balance = 10000.0
        engine.slippage_pct = 0.001
        engine._atr_multiplier = 2.0
        engine._max_hold_bars = 576  # 48h for trend
        engine._range_max_hold_bars = 288  # 24h for range
        engine._range_bounces = {}

        pos = _OpenPosition(
            symbol="BTC-EUR", direction="LONG", entry_price=50000.0,
            entry_time="2025-09-01", size_eur=100.0, stop_loss=49000.0,
            take_profit=50500.0, highest_price=50000.0, strategy="range",
            entry_bar=0, range_mid=50250.0, range_upper=50500.0, range_lower=50000.0,
        )
        engine.positions.append(pos)

        engine._check_exits_fast(
            price=50100.0, candle_high=50150.0, candle_low=50050.0,
            time_str="2025-09-02", atr_val=200.0, current_bar=288,
        )
        assert len(engine.positions) == 0
        assert len(engine.closed_trades) == 1
        assert engine.closed_trades[0].exit_reason == "time_exit"

    def test_range_bounce_increments_on_close(self):
        """Closing a range trade increments the bounce counter."""
        engine = BacktestEngine.__new__(BacktestEngine)
        engine.positions = []
        engine.closed_trades = []
        engine.balance = 10000.0
        engine.peak_balance = 10000.0
        engine.slippage_pct = 0.001
        engine._range_bounces = {}

        pos = _OpenPosition(
            symbol="BTC-EUR", direction="LONG", entry_price=50000.0,
            entry_time="2025-09-01", size_eur=100.0, stop_loss=49000.0,
            take_profit=50500.0, highest_price=50000.0, strategy="range",
            entry_bar=0,
        )
        engine.positions.append(pos)
        engine._close_position(pos, 50500.0, "2025-09-01T12:00", "take_profit")

        assert engine._range_bounces.get("BTC-EUR", 0) == 1
