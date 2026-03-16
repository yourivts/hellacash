import pandas as pd
import numpy as np
from bot.strategy.router import detect_regime, Regime, StrategyRouter


def _make_df(adx: float, atr_pct: float, rows: int = 50) -> pd.DataFrame:
    close = np.full(rows, 100.0)
    return pd.DataFrame({
        "close": close,
        "adx": np.full(rows, adx),
        "atr": np.full(rows, atr_pct),
        "high": close * (1 + atr_pct / 100),
        "low": close * (1 - atr_pct / 100),
    })


class TestRegimeDetection:
    def test_quiet_regime(self):
        df = _make_df(adx=30.0, atr_pct=0.5)
        assert detect_regime(df) == Regime.QUIET

    def test_volatile_regime(self):
        df = _make_df(adx=30.0, atr_pct=5.0)
        assert detect_regime(df) == Regime.VOLATILE

    def test_trending_regime(self):
        df = _make_df(adx=30.0, atr_pct=2.0)
        assert detect_regime(df) == Regime.TRENDING

    def test_ranging_regime(self):
        df = _make_df(adx=15.0, atr_pct=2.0)
        assert detect_regime(df) == Regime.RANGING

    def test_neutral_regime(self):
        df = _make_df(adx=22.0, atr_pct=2.0)
        assert detect_regime(df) == Regime.NEUTRAL


class TestStrategyRouter:
    def test_quiet_returns_empty(self):
        router = StrategyRouter()
        strats = router.get_strategies(Regime.QUIET)
        assert len(strats) == 0

    def test_trending_strategies(self):
        router = StrategyRouter()
        strats = router.get_strategies(Regime.TRENDING)
        names = {s.name for s in strats}
        assert names == {"orderflow", "funding_contrarian", "squeeze"}

    def test_ranging_strategies(self):
        router = StrategyRouter()
        strats = router.get_strategies(Regime.RANGING)
        names = {s.name for s in strats}
        assert names == {"range", "orderflow", "funding_contrarian"}

    def test_volatile_strategies(self):
        router = StrategyRouter()
        strats = router.get_strategies(Regime.VOLATILE)
        names = {s.name for s in strats}
        assert names == {"funding_contrarian", "squeeze"}

    def test_neutral_strategies(self):
        router = StrategyRouter()
        strats = router.get_strategies(Regime.NEUTRAL)
        names = {s.name for s in strats}
        assert names == {"funding_contrarian", "orderflow"}
