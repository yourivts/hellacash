"""Confluence Gate: detects when 2+ strategies agree on direction."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ConfluenceResult:
    triggered: bool = False
    direction: str = "NEUTRAL"
    strength: float = 0.0
    agreeing_strategies: List[str] = field(default_factory=list)


def check_confluence(signals: List[Dict]) -> ConfluenceResult:
    if not signals:
        return ConfluenceResult()

    directional = [s for s in signals if s.get("direction") not in ("NEUTRAL", None)]
    if len(directional) < 2:
        return ConfluenceResult()

    direction_counts = Counter(s["direction"] for s in directional)
    most_common_dir, count = direction_counts.most_common(1)[0]

    if count < 2:
        return ConfluenceResult()

    agreeing = [s for s in directional if s["direction"] == most_common_dir]
    best_strength = max(s.get("strength", 0.0) for s in agreeing)

    return ConfluenceResult(
        triggered=True,
        direction=most_common_dir,
        strength=best_strength,
        agreeing_strategies=[s["strategy"] for s in agreeing],
    )
