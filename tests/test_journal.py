"""Tests for trade journal reasoning generator."""
from __future__ import annotations
from bot.analytics.journal import generate_entry_reasoning, generate_exit_reasoning


class TestEntryReasoning:
    def test_generates_string(self):
        reasoning = generate_entry_reasoning(
            symbol="BTC-EUR", direction="LONG", strategy="hybrid",
            regime="trending", final_score=0.64, threshold=0.35,
            mtf_scores={"15m": 0.42, "1h": 0.61, "4h": 0.55, "1d": -0.12},
            mtf_agreement=0.75,
            technical_score=0.58, sentiment_score=0.31,
            onchain_score=0.22, orderbook_imbalance=0.15,
            indicator_values={"rsi": 34, "macd_histogram": 0.002},
        )
        assert "LONG" in reasoning
        assert "BTC-EUR" in reasoning
        assert "TRENDING" in reasoning or "trending" in reasoning

    def test_handles_missing_optional_scores(self):
        reasoning = generate_entry_reasoning(
            symbol="ETH-EUR", direction="SHORT", strategy="hybrid",
            regime="ranging", final_score=-0.45, threshold=0.35,
            mtf_scores={"15m": -0.3, "1h": -0.5},
            mtf_agreement=1.0,
            technical_score=-0.5, sentiment_score=-0.1,
        )
        assert "SHORT" in reasoning


class TestExitReasoning:
    def test_generates_string(self):
        reasoning = generate_exit_reasoning(
            exit_reason="take_profit", hold_seconds=3600,
            entry_price=50000, exit_price=51500,
            net_pnl=47.20, roi_pct=2.3,
        )
        assert "take_profit" in reasoning or "take profit" in reasoning
        assert "47" in reasoning
