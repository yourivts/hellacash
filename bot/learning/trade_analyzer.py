"""Post-trade analysis: extracts features and labels for learning."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from bot.data.database import get_session
from bot.data.repositories import (
    count_learning_events,
    save_learning_event,
)

logger = logging.getLogger(__name__)


class TradeAnalyzer:
    """
    Called after every trade closes.
    Extracts a feature vector from the trade's entry snapshot and labels it.
    """

    async def process_closed_trade(
        self,
        trade_id: int,
        symbol: str,
        strategy_name: str,
        net_pnl: float,
        entry_features: Optional[Dict[str, Any]],
    ) -> None:
        if entry_features is None:
            return

        outcome = 1 if net_pnl > 0 else 0

        feature_vector = self._extract_features(entry_features)

        async with get_session() as session:
            await save_learning_event(
                session,
                trade_id=trade_id,
                symbol=symbol,
                strategy_name=strategy_name,
                feature_vector=feature_vector,
                outcome=outcome,
            )

        logger.debug(
            "Learning event saved: %s %s outcome=%d pnl=€%.2f",
            strategy_name, symbol, outcome, net_pnl,
        )

    def _extract_features(self, snapshot: Dict[str, Any]) -> Dict[str, float]:
        """Normalize indicator snapshot into a clean float feature dict."""
        keys = [
            "rsi", "macd_histogram", "macd_line", "bb_pct_b", "bb_bandwidth",
            "ema20", "ema50", "adx", "volume_surge", "cci", "supertrend",
            "technical_score", "sentiment_score", "final_score", "confirming_count",
        ]
        features: Dict[str, float] = {}
        for k in keys:
            val = snapshot.get(k)
            if val is not None:
                try:
                    features[k] = float(val)
                except (TypeError, ValueError):
                    features[k] = 0.0
        return features

    async def get_trade_count(self, strategy_name: str) -> int:
        async with get_session() as session:
            return await count_learning_events(session, strategy_name)
