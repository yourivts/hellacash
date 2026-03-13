"""CRUD repositories for all ORM models."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import delete, desc, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from bot.data.models import (
    Asset,
    Candle,
    LearningEvent,
    Order,
    PortfolioSnapshot,
    Position,
    SentimentScore,
    Signal,
    StrategyParams,
    Trade,
)


# ── Assets ────────────────────────────────────────────────────────────────────


async def upsert_asset(session: AsyncSession, symbol: str, base: str, quote: str, **kwargs) -> Asset:
    result = await session.execute(select(Asset).where(Asset.symbol == symbol))
    asset = result.scalar_one_or_none()
    if asset is None:
        asset = Asset(symbol=symbol, base=base, quote=quote, **kwargs)
        session.add(asset)
    else:
        for k, v in kwargs.items():
            setattr(asset, k, v)
    return asset


async def get_active_assets(session: AsyncSession) -> List[Asset]:
    result = await session.execute(select(Asset).where(Asset.is_active == True))
    return list(result.scalars().all())


# ── Candles ───────────────────────────────────────────────────────────────────


async def upsert_candles(session: AsyncSession, candles: List[Dict[str, Any]]) -> None:
    for c in candles:
        result = await session.execute(
            select(Candle).where(
                Candle.symbol == c["symbol"],
                Candle.interval == c["interval"],
                Candle.timestamp == c["timestamp"],
            )
        )
        row = result.scalar_one_or_none()
        if row is None:
            session.add(Candle(**c))
        else:
            for k, v in c.items():
                setattr(row, k, v)


async def get_candles(
    session: AsyncSession,
    symbol: str,
    interval: str,
    limit: int = 500,
) -> List[Candle]:
    result = await session.execute(
        select(Candle)
        .where(Candle.symbol == symbol, Candle.interval == interval)
        .order_by(desc(Candle.timestamp))
        .limit(limit)
    )
    rows = list(result.scalars().all())
    return list(reversed(rows))  # chronological order


# ── Signals ───────────────────────────────────────────────────────────────────


async def save_signal(session: AsyncSession, **kwargs) -> Signal:
    sig = Signal(**kwargs)
    session.add(sig)
    await session.flush()
    return sig


async def get_recent_signals(
    session: AsyncSession, symbol: str, limit: int = 50
) -> List[Signal]:
    result = await session.execute(
        select(Signal)
        .where(Signal.symbol == symbol)
        .order_by(desc(Signal.created_at))
        .limit(limit)
    )
    return list(result.scalars().all())


# ── Orders ────────────────────────────────────────────────────────────────────


async def save_order(session: AsyncSession, **kwargs) -> Order:
    order = Order(**kwargs)
    session.add(order)
    await session.flush()
    return order


async def get_order_by_bitvavo_id(session: AsyncSession, bitvavo_id: str) -> Optional[Order]:
    result = await session.execute(
        select(Order).where(Order.bitvavo_order_id == bitvavo_id)
    )
    return result.scalar_one_or_none()


async def update_order_status(
    session: AsyncSession,
    order_id: int,
    status: str,
    fill_price: Optional[float] = None,
    filled_amount: Optional[float] = None,
    fee: Optional[float] = None,
) -> None:
    values: Dict[str, Any] = {"status": status}
    if fill_price is not None:
        values["fill_price"] = fill_price
    if filled_amount is not None:
        values["filled_amount"] = filled_amount
    if fee is not None:
        values["fee"] = fee
    await session.execute(update(Order).where(Order.id == order_id).values(**values))


# ── Positions ─────────────────────────────────────────────────────────────────


async def save_position(session: AsyncSession, **kwargs) -> Position:
    pos = Position(**kwargs)
    session.add(pos)
    await session.flush()
    return pos


async def get_open_position(session: AsyncSession, symbol: str) -> Optional[Position]:
    result = await session.execute(
        select(Position).where(Position.symbol == symbol)
    )
    return result.scalar_one_or_none()


async def get_all_open_positions(session: AsyncSession) -> List[Position]:
    result = await session.execute(select(Position))
    return list(result.scalars().all())


async def update_position(session: AsyncSession, position_id: int, **kwargs) -> None:
    await session.execute(
        update(Position).where(Position.id == position_id).values(**kwargs)
    )


async def delete_position(session: AsyncSession, symbol: str) -> None:
    await session.execute(delete(Position).where(Position.symbol == symbol))


# ── Trades ────────────────────────────────────────────────────────────────────


async def save_trade(session: AsyncSession, **kwargs) -> Trade:
    trade = Trade(**kwargs)
    session.add(trade)
    await session.flush()
    return trade


async def get_trades(
    session: AsyncSession,
    symbol: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[Trade]:
    q = select(Trade)
    if symbol:
        q = q.where(Trade.symbol == symbol)
    q = q.order_by(desc(Trade.created_at)).limit(limit).offset(offset)
    result = await session.execute(q)
    return list(result.scalars().all())


async def get_trades_since(session: AsyncSession, since: datetime) -> List[Trade]:
    result = await session.execute(
        select(Trade).where(Trade.created_at >= since).order_by(Trade.created_at)
    )
    return list(result.scalars().all())


async def count_trades_since(session: AsyncSession, since: datetime) -> int:
    result = await session.execute(
        select(Trade).where(Trade.created_at >= since)
    )
    return len(result.scalars().all())


# ── Portfolio snapshots ───────────────────────────────────────────────────────


async def save_snapshot(session: AsyncSession, **kwargs) -> PortfolioSnapshot:
    snap = PortfolioSnapshot(**kwargs)
    session.add(snap)
    await session.flush()
    return snap


async def get_snapshots(
    session: AsyncSession, days: int = 7
) -> List[PortfolioSnapshot]:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    result = await session.execute(
        select(PortfolioSnapshot)
        .where(PortfolioSnapshot.snapshot_at >= since)
        .order_by(PortfolioSnapshot.snapshot_at)
    )
    return list(result.scalars().all())


# ── Sentiment scores ──────────────────────────────────────────────────────────


async def save_sentiment(session: AsyncSession, **kwargs) -> SentimentScore:
    ss = SentimentScore(**kwargs)
    session.add(ss)
    await session.flush()
    return ss


async def get_latest_sentiment(
    session: AsyncSession, symbol: str, source: str = "aggregate"
) -> Optional[SentimentScore]:
    result = await session.execute(
        select(SentimentScore)
        .where(SentimentScore.symbol == symbol, SentimentScore.source == source)
        .order_by(desc(SentimentScore.computed_at))
        .limit(1)
    )
    return result.scalar_one_or_none()


async def get_sentiment_history(
    session: AsyncSession, symbol: str, hours: int = 24
) -> List[SentimentScore]:
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    result = await session.execute(
        select(SentimentScore)
        .where(
            SentimentScore.symbol == symbol,
            SentimentScore.source == "aggregate",
            SentimentScore.computed_at >= since,
        )
        .order_by(SentimentScore.computed_at)
    )
    return list(result.scalars().all())


# ── Strategy params ───────────────────────────────────────────────────────────


async def get_strategy_params(
    session: AsyncSession, strategy_name: str
) -> Optional[StrategyParams]:
    result = await session.execute(
        select(StrategyParams).where(StrategyParams.strategy_name == strategy_name)
    )
    return result.scalar_one_or_none()


async def upsert_strategy_params(
    session: AsyncSession, strategy_name: str, params: Dict[str, Any], **kwargs
) -> StrategyParams:
    result = await session.execute(
        select(StrategyParams).where(StrategyParams.strategy_name == strategy_name)
    )
    row = result.scalar_one_or_none()
    if row is None:
        row = StrategyParams(strategy_name=strategy_name, params=params, **kwargs)
        session.add(row)
    else:
        row.params = params
        row.version = (row.version or 1) + 1
        for k, v in kwargs.items():
            setattr(row, k, v)
    await session.flush()
    return row


async def get_all_strategy_params(session: AsyncSession) -> List[StrategyParams]:
    result = await session.execute(
        select(StrategyParams).where(StrategyParams.is_active == True)
    )
    return list(result.scalars().all())


# ── Learning events ───────────────────────────────────────────────────────────


async def save_learning_event(session: AsyncSession, **kwargs) -> LearningEvent:
    ev = LearningEvent(**kwargs)
    session.add(ev)
    await session.flush()
    return ev


async def get_learning_events(
    session: AsyncSession, strategy_name: str, limit: int = 200
) -> List[LearningEvent]:
    result = await session.execute(
        select(LearningEvent)
        .where(LearningEvent.strategy_name == strategy_name)
        .order_by(desc(LearningEvent.recorded_at))
        .limit(limit)
    )
    return list(result.scalars().all())


async def count_learning_events(session: AsyncSession, strategy_name: str) -> int:
    result = await session.execute(
        select(LearningEvent).where(LearningEvent.strategy_name == strategy_name)
    )
    return len(result.scalars().all())
