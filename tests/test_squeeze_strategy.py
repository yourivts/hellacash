from bot.strategy.squeeze import SqueezeStrategy


class TestSqueeze:
    def test_long_breakout(self):
        strat = SqueezeStrategy()
        direction, strength = strat.evaluate_1h(
            bb_bandwidth_prev=0.018, bb_bandwidth=0.035, price=102.0,
            bb_upper=101.5, bb_lower=98.5, volume_surge=1.8, ema50_slope=0.1)
        assert direction == "LONG"
        assert strength > 0.0

    def test_short_breakout(self):
        strat = SqueezeStrategy()
        direction, strength = strat.evaluate_1h(
            bb_bandwidth_prev=0.015, bb_bandwidth=0.04, price=97.0,
            bb_upper=101.0, bb_lower=99.0, volume_surge=2.0, ema50_slope=-0.1)
        assert direction == "SHORT"
        assert strength > 0.0

    def test_no_squeeze_without_compression(self):
        strat = SqueezeStrategy()
        direction, _ = strat.evaluate_1h(
            bb_bandwidth_prev=0.03, bb_bandwidth=0.035, price=102.0,
            bb_upper=101.5, bb_lower=98.5, volume_surge=1.8, ema50_slope=0.1)
        assert direction == "NEUTRAL"

    def test_no_squeeze_without_expansion(self):
        strat = SqueezeStrategy()
        direction, _ = strat.evaluate_1h(
            bb_bandwidth_prev=0.015, bb_bandwidth=0.025, price=102.0,
            bb_upper=101.5, bb_lower=98.5, volume_surge=1.8, ema50_slope=0.1)
        assert direction == "NEUTRAL"

    def test_no_squeeze_without_volume(self):
        strat = SqueezeStrategy()
        direction, _ = strat.evaluate_1h(
            bb_bandwidth_prev=0.015, bb_bandwidth=0.04, price=102.0,
            bb_upper=101.5, bb_lower=98.5, volume_surge=1.2, ema50_slope=0.1)
        assert direction == "NEUTRAL"

    def test_no_squeeze_against_trend(self):
        strat = SqueezeStrategy()
        direction, _ = strat.evaluate_1h(
            bb_bandwidth_prev=0.015, bb_bandwidth=0.04, price=102.0,
            bb_upper=101.5, bb_lower=98.5, volume_surge=2.0, ema50_slope=-0.1)
        assert direction == "NEUTRAL"

    def test_tp_is_2x_squeeze_width(self):
        strat = SqueezeStrategy()
        _, strength = strat.evaluate_1h(
            bb_bandwidth_prev=0.018, bb_bandwidth=0.04, price=102.0,
            bb_upper=101.5, bb_lower=98.5, volume_surge=2.0, ema50_slope=0.1)
        assert strat.last_tp_distance == 6.0  # 2 * (101.5 - 98.5)
