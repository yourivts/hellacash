from bot.strategy.orderflow import OrderFlowStrategy


def _make_ctx(**kwargs):
    defaults = dict(body_ratio=0.2, wick_lower_ratio=0.5, wick_upper_ratio=0.1,
                    volume_surge=2.0, rsi_1h=50.0, cmf=0.1, obv_divergence=1.0)
    defaults.update(kwargs)
    return defaults


class TestOrderflow1h:
    def test_buying_absorption_long(self):
        strat = OrderFlowStrategy()
        direction, strength = strat.evaluate_1h(**_make_ctx(
            body_ratio=0.15, wick_lower_ratio=0.55, volume_surge=1.8,
            rsi_1h=45.0, cmf=0.15, obv_divergence=1.0))
        assert direction == "LONG"
        assert strength > 0.0

    def test_selling_absorption_short(self):
        strat = OrderFlowStrategy()
        direction, strength = strat.evaluate_1h(**_make_ctx(
            body_ratio=0.15, wick_upper_ratio=0.55, wick_lower_ratio=0.1,
            volume_surge=1.8, rsi_1h=55.0, cmf=-0.15, obv_divergence=-1.0))
        assert direction == "SHORT"
        assert strength > 0.0

    def test_rejects_large_body(self):
        strat = OrderFlowStrategy()
        direction, _ = strat.evaluate_1h(**_make_ctx(body_ratio=0.40))
        assert direction == "NEUTRAL"

    def test_rejects_low_volume(self):
        strat = OrderFlowStrategy()
        direction, _ = strat.evaluate_1h(**_make_ctx(volume_surge=1.2))
        assert direction == "NEUTRAL"

    def test_rejects_extreme_rsi(self):
        strat = OrderFlowStrategy()
        direction, _ = strat.evaluate_1h(**_make_ctx(rsi_1h=75.0))
        assert direction == "NEUTRAL"
