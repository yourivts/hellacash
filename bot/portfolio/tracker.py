"""Real-time portfolio state tracker (in-memory + DB sync)."""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

from bot.data.database import get_session
from bot.data.repositories import (
    delete_position,
    get_all_open_positions,
    get_trades_since,
    save_snapshot,
    save_trade,
    update_position,
)
from bot.exchange.bitvavo_client import BitvavoClient

logger = logging.getLogger(__name__)


class PortfolioTracker:
    """
    Maintains real-time view of portfolio state.
    Syncs with DB for persistence.
    """

    def __init__(self, client: BitvavoClient) -> None:
        self.client = client
        self._positions: Dict[str, Dict[str, Any]] = {}  # symbol → position dict
        self._peak_equity: float = 0.0
        self._daily_realized_loss: float = 0.0
        self._daily_loss_date: date = date.today()

    async def refresh(self) -> None:
        """Reload positions from DB and update prices from exchange."""
        async with get_session() as session:
            positions = await get_all_open_positions(session)

        self._positions = {}
        for p in positions:
            ticker = self.client.get_ticker(p.symbol)
            current_price = ticker.price if ticker else p.current_price
            self._positions[p.symbol] = {
                "id": p.id,
                "symbol": p.symbol,
                "strategy_name": p.strategy_name,
                "entry_price": p.entry_price,
                "current_price": current_price,
                "quantity": p.quantity,
                "stop_loss_price": p.stop_loss_price,
                "take_profit_price": p.take_profit_price,
                "trailing_stop_pct": p.trailing_stop_pct,
                "highest_price": p.highest_price,
                "entry_order_id": p.entry_order_id,
                "paper_trade": p.paper_trade,
                "opened_at": p.opened_at,
                "unrealized_pnl": (current_price - p.entry_price) * p.quantity,
            }

    def get_equity_eur(self) -> float:
        """Total portfolio value in EUR (cash + positions)."""
        cash = self.client.get_balance_eur()
        positions_value = sum(
            p["current_price"] * p["quantity"] for p in self._positions.values()
        )
        return cash + positions_value

    def get_cash_eur(self) -> float:
        return self.client.get_balance_eur()

    def get_positions_value_eur(self) -> float:
        return sum(p["current_price"] * p["quantity"] for p in self._positions.values())

    def open_position_count(self) -> int:
        return len(self._positions)

    def has_position(self, symbol: str) -> bool:
        return symbol in self._positions

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        return self._positions.get(symbol)

    def get_all_positions(self) -> List[Dict[str, Any]]:
        return list(self._positions.values())

    def update_price(self, symbol: str, price: float) -> None:
        if symbol in self._positions:
            pos = self._positions[symbol]
            pos["current_price"] = price
            pos["unrealized_pnl"] = (price - pos["entry_price"]) * pos["quantity"]
            if price > pos["highest_price"]:
                pos["highest_price"] = price

    def current_drawdown_pct(self) -> float:
        equity = self.get_equity_eur()
        if self._peak_equity <= 0:
            return 0.0
        return max(0.0, (self._peak_equity - equity) / self._peak_equity * 100.0)

    def daily_realized_loss_eur(self) -> float:
        today = date.today()
        if today != self._daily_loss_date:
            self._daily_realized_loss = 0.0
            self._daily_loss_date = today
        return self._daily_realized_loss

    async def add_position(
        self,
        symbol: str,
        strategy_name: str,
        entry_price: float,
        quantity: float,
        stop_loss: float,
        take_profit: float,
        entry_order_id: Optional[int],
        paper_trade: bool = True,
    ) -> None:
        from bot.data.repositories import save_position
        async with get_session() as session:
            pos = await save_position(
                session,
                symbol=symbol,
                strategy_name=strategy_name,
                entry_price=entry_price,
                current_price=entry_price,
                quantity=quantity,
                stop_loss_price=stop_loss,
                take_profit_price=take_profit,
                highest_price=entry_price,
                entry_order_id=entry_order_id,
                paper_trade=paper_trade,
            )

        self._positions[symbol] = {
            "id": pos.id,
            "symbol": symbol,
            "strategy_name": strategy_name,
            "entry_price": entry_price,
            "current_price": entry_price,
            "quantity": quantity,
            "stop_loss_price": stop_loss,
            "take_profit_price": take_profit,
            "trailing_stop_pct": None,
            "highest_price": entry_price,
            "entry_order_id": entry_order_id,
            "paper_trade": paper_trade,
            "opened_at": datetime.now(timezone.utc),
            "unrealized_pnl": 0.0,
        }
        logger.info("Position opened: %s x%.4f @ €%.4f", symbol, quantity, entry_price)

    async def close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_order_id: Optional[int],
        exit_reason: str,
        sentiment_score: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        pos = self._positions.pop(symbol, None)
        if pos is None:
            return None

        entry_price = pos["entry_price"]
        quantity = pos["quantity"]
        gross_pnl = (exit_price - entry_price) * quantity
        fee_pct = 0.0025
        fee = (entry_price + exit_price) * quantity * fee_pct
        net_pnl = gross_pnl - fee
        roi_pct = net_pnl / (entry_price * quantity) * 100.0

        if net_pnl < 0:
            self._daily_realized_loss += abs(net_pnl)

        hold_secs = int(
            (datetime.now(timezone.utc) - pos["opened_at"]).total_seconds()
        )

        async with get_session() as session:
            # Remove from positions table
            await delete_position(session, symbol)

            # Save closed trade
            trade = await save_trade(
                session,
                symbol=symbol,
                strategy_name=pos["strategy_name"],
                entry_order_id=pos["entry_order_id"],
                exit_order_id=exit_order_id,
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=quantity,
                gross_pnl=gross_pnl,
                net_pnl=net_pnl,
                roi_pct=roi_pct,
                exit_sentiment=sentiment_score,
                hold_seconds=hold_secs,
                exit_reason=exit_reason,
                paper_trade=pos["paper_trade"],
            )

        logger.info(
            "Position closed: %s @ €%.4f | P&L: €%.2f (%.2f%%) | Reason: %s",
            symbol, exit_price, net_pnl, roi_pct, exit_reason,
        )

        return {
            "trade_id": trade.id,
            "symbol": symbol,
            "net_pnl": net_pnl,
            "roi_pct": roi_pct,
            "exit_reason": exit_reason,
        }

    async def snapshot(self) -> None:
        """Save a portfolio snapshot to DB."""
        equity = self.get_equity_eur()
        if equity > self._peak_equity:
            self._peak_equity = equity

        dd = self.current_drawdown_pct()

        # Compute simple metrics from recent trades
        since = datetime.utcnow().replace(hour=0, minute=0, second=0)
        async with get_session() as session:
            today_trades = await get_trades_since(session, since)

        daily_pnl = sum(t.net_pnl for t in today_trades) if today_trades else 0.0
        wins = [t for t in today_trades if t.net_pnl > 0]
        win_rate = len(wins) / len(today_trades) if today_trades else None

        async with get_session() as session:
            await save_snapshot(
                session,
                total_equity_eur=equity,
                cash_eur=self.get_cash_eur(),
                positions_value_eur=self.get_positions_value_eur(),
                daily_pnl=daily_pnl,
                max_drawdown_pct=dd,
                win_rate=win_rate,
            )
