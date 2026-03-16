# tests/test_fixed_fractional.py
from bot.risk.position_sizer import fixed_fractional_size

class TestFixedFractional:
    def test_basic_sizing(self):
        """3% risk on 1000 EUR with 2% stop, capped at 30% of equity."""
        size = fixed_fractional_size(
            equity=1000.0, base_risk_pct=3.0, stop_distance_pct=2.0,
            atr_pct=1.5, median_atr_pct=1.5,
        )
        assert abs(size - 300.0) < 0.01

    def test_volatility_scaling_up(self):
        size = fixed_fractional_size(
            equity=1000.0, base_risk_pct=3.0, stop_distance_pct=4.0,
            atr_pct=3.0, median_atr_pct=1.5,
        )
        assert abs(size - 300.0) < 0.01

    def test_volatility_scaling_down(self):
        size = fixed_fractional_size(
            equity=1000.0, base_risk_pct=3.0, stop_distance_pct=1.0,
            atr_pct=0.5, median_atr_pct=1.5,
        )
        assert abs(size - 300.0) < 0.01

    def test_minimum_position(self):
        size = fixed_fractional_size(
            equity=100.0, base_risk_pct=1.0, stop_distance_pct=20.0,
            atr_pct=1.0, median_atr_pct=1.0,
        )
        assert abs(size - 10.0) < 0.01

    def test_max_cap_30_percent(self):
        size = fixed_fractional_size(
            equity=500.0, base_risk_pct=5.0, stop_distance_pct=0.5,
            atr_pct=1.5, median_atr_pct=1.5,
        )
        assert abs(size - 150.0) < 0.01

    def test_confluence_bonus(self):
        size = fixed_fractional_size(
            equity=1000.0, base_risk_pct=4.5, stop_distance_pct=3.0,
            atr_pct=1.5, median_atr_pct=1.5,
        )
        assert abs(size - 300.0) < 0.01
