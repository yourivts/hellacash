from bot.strategy.range_trading import RangeStrategy


class TestRange1h:
    def test_long_at_lower_band(self):
        strat = RangeStrategy()
        direction, strength = strat.evaluate_1h(
            price=98.5, bb_lower=98.0, bb_upper=102.0, bb_mid=100.0,
            bb_bandwidth=0.04, adx_4h=18.0, rsi_1h=30.0)
        assert direction == "LONG"
        assert strength > 0.0

    def test_short_at_upper_band(self):
        strat = RangeStrategy()
        direction, strength = strat.evaluate_1h(
            price=101.8, bb_lower=98.0, bb_upper=102.0, bb_mid=100.0,
            bb_bandwidth=0.04, adx_4h=18.0, rsi_1h=70.0)
        assert direction == "SHORT"
        assert strength > 0.0

    def test_neutral_when_trending(self):
        strat = RangeStrategy()
        direction, _ = strat.evaluate_1h(
            price=98.5, bb_lower=98.0, bb_upper=102.0, bb_mid=100.0,
            bb_bandwidth=0.04, adx_4h=28.0, rsi_1h=30.0)
        assert direction == "NEUTRAL"

    def test_neutral_when_bandwidth_too_wide(self):
        strat = RangeStrategy()
        direction, _ = strat.evaluate_1h(
            price=98.5, bb_lower=90.0, bb_upper=110.0, bb_mid=100.0,
            bb_bandwidth=0.10, adx_4h=18.0, rsi_1h=30.0)
        assert direction == "NEUTRAL"

    def test_neutral_when_bandwidth_too_narrow(self):
        strat = RangeStrategy()
        direction, _ = strat.evaluate_1h(
            price=99.8, bb_lower=99.5, bb_upper=100.5, bb_mid=100.0,
            bb_bandwidth=0.01, adx_4h=18.0, rsi_1h=30.0)
        assert direction == "NEUTRAL"

    def test_neutral_when_price_not_near_band(self):
        strat = RangeStrategy()
        direction, _ = strat.evaluate_1h(
            price=99.0, bb_lower=97.0, bb_upper=103.0, bb_mid=100.0,
            bb_bandwidth=0.06, adx_4h=18.0, rsi_1h=30.0)
        assert direction == "NEUTRAL"
