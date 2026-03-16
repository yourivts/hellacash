from bot.strategy.confluence import check_confluence, ConfluenceResult

class TestConfluence:
    def test_two_agree_triggers_confluence(self):
        signals = [
            {"strategy": "orderflow", "direction": "LONG", "strength": 0.7},
            {"strategy": "funding_contrarian", "direction": "LONG", "strength": 0.6},
        ]
        result = check_confluence(signals)
        assert result.triggered is True
        assert result.direction == "LONG"
        assert result.agreeing_strategies == ["orderflow", "funding_contrarian"]

    def test_disagreement_no_confluence(self):
        signals = [
            {"strategy": "orderflow", "direction": "LONG", "strength": 0.7},
            {"strategy": "funding_contrarian", "direction": "SHORT", "strength": 0.6},
        ]
        result = check_confluence(signals)
        assert result.triggered is False

    def test_single_signal_no_confluence(self):
        signals = [{"strategy": "orderflow", "direction": "LONG", "strength": 0.7}]
        result = check_confluence(signals)
        assert result.triggered is False

    def test_neutral_signals_ignored(self):
        signals = [
            {"strategy": "orderflow", "direction": "LONG", "strength": 0.7},
            {"strategy": "range", "direction": "NEUTRAL", "strength": 0.0},
            {"strategy": "funding_contrarian", "direction": "NEUTRAL", "strength": 0.0},
        ]
        result = check_confluence(signals)
        assert result.triggered is False

    def test_best_strength_used(self):
        signals = [
            {"strategy": "orderflow", "direction": "SHORT", "strength": 0.5},
            {"strategy": "squeeze", "direction": "SHORT", "strength": 0.9},
        ]
        result = check_confluence(signals)
        assert result.triggered is True
        assert result.strength == 0.9

    def test_empty_signals(self):
        result = check_confluence([])
        assert result.triggered is False
