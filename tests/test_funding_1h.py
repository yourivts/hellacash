from bot.strategy.funding_contrarian import FundingContrarianStrategy


class TestFunding1h:
    def test_overbought_short(self):
        strat = FundingContrarianStrategy()
        direction, strength = strat.evaluate_1h(
            rsi_1h=75.0, rsi_4h=65.0, macd_hist=0.5, macd_hist_prev=1.2)
        assert direction == "SHORT"
        assert strength > 0.0

    def test_oversold_long(self):
        strat = FundingContrarianStrategy()
        direction, strength = strat.evaluate_1h(
            rsi_1h=22.0, rsi_4h=35.0, macd_hist=-0.5, macd_hist_prev=-1.2)
        assert direction == "LONG"
        assert strength > 0.0

    def test_neutral_when_not_extreme(self):
        strat = FundingContrarianStrategy()
        direction, _ = strat.evaluate_1h(
            rsi_1h=50.0, rsi_4h=50.0, macd_hist=0.1, macd_hist_prev=0.1)
        assert direction == "NEUTRAL"

    def test_no_signal_without_4h_confirmation(self):
        strat = FundingContrarianStrategy()
        direction, _ = strat.evaluate_1h(
            rsi_1h=75.0, rsi_4h=45.0, macd_hist=0.5, macd_hist_prev=1.2)
        assert direction == "NEUTRAL"

    def test_no_signal_without_macd_fading(self):
        strat = FundingContrarianStrategy()
        direction, _ = strat.evaluate_1h(
            rsi_1h=75.0, rsi_4h=65.0, macd_hist=1.5, macd_hist_prev=1.0)
        assert direction == "NEUTRAL"
