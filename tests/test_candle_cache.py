"""Tests for CandleCache expansion."""
from bot.data_loader import CandleCache, MAX_CANDLES


class TestCandleCacheCapacity:
    def test_max_candles_is_2000(self):
        assert MAX_CANDLES == 2000

    def test_cache_holds_2000_candles(self):
        cc = CandleCache()
        for i in range(2100):
            cc.cache_candle("BTC-EUR", "5m", {
                "timestamp": f"2026-01-01T00:{i:04d}",
                "open": 100, "high": 101, "low": 99, "close": 100, "volume": 10,
            })
        rows = cc._cache["BTC-EUR"]["5m"]
        assert len(rows) == 2000
