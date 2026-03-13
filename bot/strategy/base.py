"""Abstract base strategy and Signal dataclass."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class Signal:
    symbol: str
    direction: str          # "LONG", "SHORT", "NEUTRAL"
    strength: float         # 0.0 – 1.0
    strategy_name: str
    technical_score: float = 0.0
    sentiment_score: float = 0.0
    indicator_snapshot: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def is_actionable(self, min_confidence: float = 0.60, min_confirmations: int = 2) -> bool:
        return (
            self.direction in ("LONG", "SHORT")
            and self.strength >= min_confidence
            and self.indicator_snapshot.get("confirming_count", 0) >= min_confirmations
        )


@dataclass
class MarketContext:
    symbol: str
    candles_5m: pd.DataFrame    # primary timeframe
    candles_1h: pd.DataFrame    # higher timeframe context
    current_price: float
    sentiment_score: float      # -1.0 to +1.0
    portfolio_equity_eur: float
    open_position_count: int
    indicator_weights: Optional[Dict[str, float]] = None


class BaseStrategy(ABC):
    name: str = "base"

    @abstractmethod
    def generate_signal(self, ctx: MarketContext) -> Signal:
        ...

    def get_default_params(self) -> Dict[str, Any]:
        return {}

    def get_param_space(self) -> Dict[str, Any]:
        """Optuna parameter search space definition."""
        return {}
