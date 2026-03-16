"""Integration test: different param sets produce differentiated backtest results."""
import numpy as np
import pandas as pd
from bot.backtest.engine import BacktestEngine


def _make_synthetic_candles(n_5m=6000):
    """Generate synthetic 5m candles with trends, reversals, and volume spikes.

    6000 candles = ~20 days — enough to generate multiple signals across strategies.
    """
    candles = []
    base = 80000.0
    rng = np.random.RandomState(42)
    for i in range(n_5m):
        ts = pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(minutes=5 * i)
        # Multiple overlapping cycles for varied price action
        trend = i * 0.5
        fast_cycle = np.sin(i / 30) * 800    # fast oscillation
        slow_cycle = np.sin(i / 200) * 3000  # slow trend
        noise = rng.normal(0, 150)
        price = base + trend + fast_cycle + slow_cycle + noise
        # Volume spikes every ~100 candles to trigger breakout signals
        vol_spike = 5.0 if i % 100 < 3 else 1.0
        candles.append({
            "timestamp": ts.isoformat(),
            "open": price - 10,
            "high": price + abs(noise) + 30,
            "low": price - abs(noise) - 30,
            "close": price,
            "volume": (500 + abs(noise) * 5) * vol_spike,
        })
    return candles



def test_atr_multiplier_changes_stop_levels():
    """Different atr_multiplier should change stop-loss distances."""
    candles = _make_synthetic_candles(500)

    engine_tight = BacktestEngine(candles, strategy_params={"atr_multiplier": 1.0})
    engine_wide = BacktestEngine(candles, strategy_params={"atr_multiplier": 4.0})

    df = engine_tight._to_dataframe(candles)
    from bot.risk.stop_loss import initial_stops
    window = df.iloc[200:401]
    price = float(df["close"].iloc[400])

    sl_tight, tp_tight = initial_stops(price, window, direction="LONG",
                                        atr_multiplier=engine_tight._atr_multiplier)
    sl_wide, tp_wide = initial_stops(price, window, direction="LONG",
                                      atr_multiplier=engine_wide._atr_multiplier)

    # Wider multiplier = wider stop (further from entry)
    assert sl_wide < sl_tight, (
        f"Wider ATR multiplier should set stop further: tight={sl_tight}, wide={sl_wide}"
    )
    assert tp_wide > tp_tight, (
        f"Wider ATR multiplier should set TP further: tight={tp_tight}, wide={tp_wide}"
    )
