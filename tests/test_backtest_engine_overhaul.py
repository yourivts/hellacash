"""Tests for backtest engine overhaul (Task 13)."""
from __future__ import annotations

import pytest
from datetime import datetime, timezone, timedelta

from bot.backtest.engine import (
    BacktestEngine,
    BacktestResult,
    MAKER_FEE_PCT,
    TAKER_FEE_PCT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candles(n: int = 200, base_price: float = 50000.0, spread: float = 100.0):
    """Generate n synthetic 5m candle dicts with simple oscillation."""
    candles = []
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        # Oscillate around base price to create some trading signals
        offset = spread * (1 if (i // 24) % 2 == 0 else -1) * (i % 24) / 24
        price = base_price + offset
        candles.append({
            "timestamp": start + timedelta(minutes=5 * i),
            "open": price - 10,
            "high": price + 50,
            "low": price - 50,
            "close": price,
            "volume": 100.0 + (i % 10) * 10,
        })
    return candles


# ---------------------------------------------------------------------------
# Step 2: Constants and defaults
# ---------------------------------------------------------------------------

class TestConstants:
    def test_signal_every_is_12(self):
        assert BacktestEngine.SIGNAL_EVERY == 12

    def test_max_open_positions_default_is_10(self):
        candles = _make_candles(100)
        engine = BacktestEngine(candles)
        assert engine.max_open == 10

    def test_range_max_hold_bars_is_864(self):
        candles = _make_candles(100)
        engine = BacktestEngine(candles)
        assert engine._range_max_hold_bars == 864

    def test_maker_fee_pct_exists(self):
        assert MAKER_FEE_PCT == 0.0015

    def test_taker_fee_pct_exists(self):
        assert TAKER_FEE_PCT == 0.0025


# ---------------------------------------------------------------------------
# Step 9: BacktestResult new fields
# ---------------------------------------------------------------------------

class TestBacktestResultFields:
    def test_new_fields_exist(self):
        result = BacktestResult()
        assert hasattr(result, "profit_factor")
        assert hasattr(result, "total_fees_paid")
        assert hasattr(result, "profit_per_fee")
        assert hasattr(result, "signals_generated")
        assert hasattr(result, "signals_filled")
        assert hasattr(result, "fill_rate")
        assert hasattr(result, "quiet_hours_skipped")
        assert hasattr(result, "regime_pnl")

    def test_new_fields_defaults(self):
        result = BacktestResult()
        assert result.profit_factor == 0.0
        assert result.total_fees_paid == 0.0
        assert result.profit_per_fee == 0.0
        assert result.signals_generated == 0
        assert result.signals_filled == 0
        assert result.fill_rate == 0.0
        assert result.quiet_hours_skipped == 0
        assert result.regime_pnl == {}


# ---------------------------------------------------------------------------
# Step 10b: No update_hybrid_params / position_size_modifier calls
# ---------------------------------------------------------------------------

class TestRemovedMethods:
    def test_no_position_size_modifier_attribute(self):
        """The new StrategyRouter does not have position_size_modifier."""
        from bot.strategy.router import StrategyRouter
        router = StrategyRouter()
        assert not hasattr(router, "position_size_modifier")

    def test_no_update_hybrid_params_attribute(self):
        """The new StrategyRouter does not have update_hybrid_params."""
        from bot.strategy.router import StrategyRouter
        router = StrategyRouter()
        assert not hasattr(router, "update_hybrid_params")


# ---------------------------------------------------------------------------
# Step 3: Imports exist
# ---------------------------------------------------------------------------

class TestImports:
    def test_fee_gate_importable(self):
        from bot.risk.fee_gate import check_fee_gate
        assert callable(check_fee_gate)

    def test_confluence_importable(self):
        from bot.strategy.confluence import check_confluence
        assert callable(check_confluence)

    def test_regime_importable(self):
        from bot.strategy.router import Regime
        assert hasattr(Regime, "QUIET")
        assert hasattr(Regime, "VOLATILE")
        assert hasattr(Regime, "TRENDING")
        assert hasattr(Regime, "RANGING")
        assert hasattr(Regime, "NEUTRAL")


# ---------------------------------------------------------------------------
# Engine smoke test: runs without error on synthetic data
# ---------------------------------------------------------------------------

class TestEngineSmokeTest:
    def test_engine_runs_without_error(self):
        """Verify the engine can run on synthetic candles without crashing."""
        candles = _make_candles(300, base_price=50000.0)
        engine = BacktestEngine(candles, initial_capital=10_000.0)
        result = engine.run()
        assert isinstance(result, BacktestResult)
        # Result should have the new fields populated
        assert isinstance(result.fill_rate, (int, float))
        assert isinstance(result.regime_pnl, dict)

    def test_engine_too_few_candles(self):
        """Less than WARMUP candles should return empty result."""
        candles = _make_candles(30)
        engine = BacktestEngine(candles)
        result = engine.run()
        assert result.total_trades == 0

    def test_custom_max_open_positions(self):
        candles = _make_candles(100)
        engine = BacktestEngine(candles, max_open_positions=5)
        assert engine.max_open == 5

    def test_custom_strategy_params(self):
        candles = _make_candles(100)
        engine = BacktestEngine(candles, strategy_params={"cooldown_hours": 24})
        assert engine._cooldown_bars == 24 * 12


# ---------------------------------------------------------------------------
# TradeRecord regime field
# ---------------------------------------------------------------------------

class TestTradeRecordRegime:
    def test_trade_record_has_regime_field(self):
        from bot.backtest.engine import TradeRecord
        tr = TradeRecord(
            symbol="BTC-EUR", direction="LONG",
            entry_price=50000, exit_price=50500,
            entry_time="2025-01-01", exit_time="2025-01-02",
            size_eur=1000, pnl_eur=10, pnl_pct=1.0,
            exit_reason="take_profit", strategy="orderflow",
            regime="trending",
        )
        assert tr.regime == "trending"

    def test_trade_record_regime_default(self):
        from bot.backtest.engine import TradeRecord
        tr = TradeRecord(
            symbol="BTC-EUR", direction="LONG",
            entry_price=50000, exit_price=50500,
            entry_time="2025-01-01", exit_time="2025-01-02",
            size_eur=1000, pnl_eur=10, pnl_pct=1.0,
            exit_reason="take_profit", strategy="orderflow",
        )
        assert tr.regime == "neutral"
