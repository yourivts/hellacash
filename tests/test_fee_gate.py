# tests/test_fee_gate.py
from bot.risk.fee_gate import check_fee_gate, FeeGateResult


class TestFeeGate:
    def test_passes_when_profit_exceeds_fees(self):
        result = check_fee_gate(
            position_size=100.0, tp_distance_pct=2.0,
            maker_fee_pct=0.15, taker_fee_pct=0.25,
            min_profit_multiple=3.0, is_short=False, expected_hold_hours=0,
        )
        assert result.approved is True
        assert result.fee_ratio >= 3.0

    def test_rejects_when_profit_below_threshold(self):
        result = check_fee_gate(
            position_size=50.0, tp_distance_pct=0.5,
            maker_fee_pct=0.15, taker_fee_pct=0.25,
            min_profit_multiple=3.0, is_short=False, expected_hold_hours=0,
        )
        assert result.approved is False

    def test_short_includes_borrow_fee(self):
        result = check_fee_gate(
            position_size=100.0, tp_distance_pct=2.0,
            maker_fee_pct=0.15, taker_fee_pct=0.25,
            min_profit_multiple=3.0, is_short=True, expected_hold_hours=48,
            annual_borrow_rate=0.03,
        )
        assert result.approved is True
        assert result.total_fees > 0.40

    def test_zero_position_rejects(self):
        result = check_fee_gate(
            position_size=0.0, tp_distance_pct=2.0,
            maker_fee_pct=0.15, taker_fee_pct=0.25,
            min_profit_multiple=3.0,
        )
        assert result.approved is False
