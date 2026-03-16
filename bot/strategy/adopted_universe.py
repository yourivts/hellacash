"""AdoptedUniverse: per-strategy-per-symbol params from walk-forward."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

CHAMPION_DEFAULTS: Dict[str, Any] = {
    "atr_multiplier": 3.5,
    "rr_ratio": 2.5,
    "base_risk_pct": 3.0,
    "min_profit_multiple": 3.0,
    "max_hold_hours": 120,
    "quiet_atr_threshold": 1.0,
    "regime_adx_threshold": 24,
    "ranging_adx_threshold": 20,
}

ALL_STRATEGIES = ["orderflow", "range", "squeeze", "funding_contrarian"]


@dataclass
class StrategyConfig:
    params: Dict[str, Any] = field(default_factory=dict)
    avg_sharpe: float = 0.0
    avg_pnl: float = 0.0


@dataclass
class SymbolConfig:
    strategies: Dict[str, StrategyConfig] = field(default_factory=dict)
    regime_params: Dict[str, float] = field(default_factory=dict)


class AdoptedUniverse:
    """Stores adopted strategy-symbol combos with per-combo parameters."""

    def __init__(self) -> None:
        self._adopted: Dict[str, SymbolConfig] = {}

    def is_adopted(self, symbol: str) -> bool:
        return symbol in self._adopted

    def get_strategy_params(self, symbol: str, strategy: str) -> Dict[str, Any]:
        if symbol in self._adopted and strategy in self._adopted[symbol].strategies:
            return self._adopted[symbol].strategies[strategy].params
        return dict(CHAMPION_DEFAULTS)

    def get_regime_params(self, symbol: str) -> Dict[str, float]:
        if symbol in self._adopted:
            return self._adopted[symbol].regime_params
        return {
            "quiet_atr_threshold": CHAMPION_DEFAULTS["quiet_atr_threshold"],
            "regime_adx_threshold": CHAMPION_DEFAULTS["regime_adx_threshold"],
        }

    def get_enabled_strategies(self, symbol: str) -> List[str]:
        if symbol in self._adopted:
            return list(self._adopted[symbol].strategies.keys())
        return []

    def get_strategy_metrics(self, symbol: str, strategy: str) -> Dict[str, float]:
        """Return avg_sharpe and avg_pnl for a strategy-symbol combo."""
        if symbol in self._adopted and strategy in self._adopted[symbol].strategies:
            sc = self._adopted[symbol].strategies[strategy]
            return {"avg_sharpe": sc.avg_sharpe, "avg_pnl": sc.avg_pnl}
        return {"avg_sharpe": 0.0, "avg_pnl": 0.0}

    def adopted_symbols(self) -> List[str]:
        return list(self._adopted.keys())

    def count(self) -> int:
        return len(self._adopted)

    def update(self, wf_results: Dict[str, Dict[str, Any]]) -> None:
        """Rebuild universe from walk-forward results.

        Args:
            wf_results: {symbol: {strategy_name: WFResult}}
        """
        new_adopted: Dict[str, SymbolConfig] = {}

        for symbol, strat_results in wf_results.items():
            adopted_strats: Dict[str, StrategyConfig] = {}
            best_sharpe = float("-inf")
            best_regime_params = {
                "quiet_atr_threshold": CHAMPION_DEFAULTS["quiet_atr_threshold"],
                "regime_adx_threshold": CHAMPION_DEFAULTS["regime_adx_threshold"],
            }

            for strat_name, result in strat_results.items():
                if result.adopted and result.recommended_params is not None:
                    adopted_strats[strat_name] = StrategyConfig(
                        params=dict(result.recommended_params),
                        avg_sharpe=result.avg_oos_sharpe,
                        avg_pnl=result.avg_oos_pnl,
                    )
                    if result.avg_oos_sharpe > best_sharpe:
                        best_sharpe = result.avg_oos_sharpe
                        best_regime_params = {
                            "quiet_atr_threshold": result.recommended_params.get(
                                "quiet_atr_threshold", CHAMPION_DEFAULTS["quiet_atr_threshold"]
                            ),
                            "regime_adx_threshold": result.recommended_params.get(
                                "regime_adx_threshold", CHAMPION_DEFAULTS["regime_adx_threshold"]
                            ),
                        }

            if adopted_strats:
                new_adopted[symbol] = SymbolConfig(
                    strategies=adopted_strats,
                    regime_params=best_regime_params,
                )

        if not new_adopted and self._adopted:
            logger.warning(
                "Walk-forward: zero symbols adopted — keeping previous universe (%d symbols)",
                len(self._adopted),
            )
            return

        # Log diff
        old_symbols = set(self._adopted.keys())
        new_symbols = set(new_adopted.keys())
        added = new_symbols - old_symbols
        removed = old_symbols - new_symbols
        total_combos = sum(len(sc.strategies) for sc in new_adopted.values())

        parts = []
        for s in sorted(added):
            strats = ",".join(sorted(new_adopted[s].strategies.keys()))
            parts.append(f"+{s}({strats})")
        for s in sorted(removed):
            parts.append(f"-{s}")

        logger.info(
            "Universe updated: %s — %d symbols / %d strategy-combos adopted",
            " ".join(parts) if parts else "no changes",
            len(new_adopted),
            total_combos,
        )

        self._adopted = new_adopted
