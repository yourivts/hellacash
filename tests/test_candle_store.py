"""Tests for CandleStore and Candle1m model."""
from bot.data.models import Candle1m


def test_candle_1m_model_exists():
    """Candle1m model has correct table name and columns."""
    assert Candle1m.__tablename__ == "candle_1m"
    cols = {c.name for c in Candle1m.__table__.columns}
    assert cols == {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
    # Verify composite primary key
    pk_cols = {c.name for c in Candle1m.__table__.primary_key.columns}
    assert pk_cols == {"symbol", "timestamp"}
