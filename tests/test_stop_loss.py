"""Tests for bot.risk.stop_loss."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bot.risk.stop_loss import initial_stops, trail_stop, check_stop_triggered


def _make_ohlcv_df(n: int = 30, base_price: float = 100.0) -> pd.DataFrame:
    """Create a simple OHLCV DataFrame for testing."""
    np.random.seed(42)
    close = base_price + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    return pd.DataFrame({"high": high, "low": low, "close": close})


class TestInitialStops:
    def test_initial_stops_sets_sl_and_tp(self):
        df = _make_ohlcv_df(30, base_price=100.0)
        entry = df["close"].iloc[-1]
        sl, tp = initial_stops(entry, df)
        assert sl < entry, "Stop loss should be below entry"
        assert tp > entry, "Take profit should be above entry"

    def test_initial_stops_fallback_short_df(self):
        df = _make_ohlcv_df(5, base_price=100.0)
        entry = 100.0
        sl, tp = initial_stops(entry, df)
        assert sl == pytest.approx(98.0)
        assert tp == pytest.approx(103.0)


class TestTrailStop:
    def test_trail_stop_moves_up_only(self):
        current_stop = 95.0
        atr_value = 2.0

        # Price rises -> stop should move up
        new_stop = trail_stop(
            current_price=105.0,
            highest_price=105.0,
            current_stop=current_stop,
            atr_value=atr_value,
        )
        assert new_stop == 103.0  # 105 - 2

        # Price falls -> stop should NOT move down
        final_stop = trail_stop(
            current_price=100.0,
            highest_price=100.0,
            current_stop=new_stop,
            atr_value=atr_value,
        )
        assert final_stop == 103.0  # stays at previous higher value


class TestCheckStopTriggered:
    def test_check_stop_triggered_stop_loss(self):
        result = check_stop_triggered(current_price=94.0, stop_loss=95.0, take_profit=110.0)
        assert result == "stop_loss"

    def test_check_stop_triggered_take_profit(self):
        result = check_stop_triggered(current_price=111.0, stop_loss=95.0, take_profit=110.0)
        assert result == "take_profit"

    def test_check_stop_triggered_none_when_between(self):
        result = check_stop_triggered(current_price=100.0, stop_loss=95.0, take_profit=110.0)
        assert result is None
