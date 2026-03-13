"""Fear & Greed Index scraper (no credentials required)."""
from __future__ import annotations

import json
import logging
import urllib.request

logger = logging.getLogger(__name__)

FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1"


class FearGreedScraper:
    """Fetches the Crypto Fear & Greed Index and exposes it as a sentiment score."""

    def __init__(self) -> None:
        pass

    def get_score(self) -> float:
        """Fetch Fear & Greed Index and return score mapped to -1.0 to +1.0."""
        try:
            req = urllib.request.Request(
                FEAR_GREED_URL,
                headers={"User-Agent": "hellacash/1.0"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            value = int(data["data"][0]["value"])  # 0 (extreme fear) – 100 (extreme greed)
            score = (value / 50.0) - 1.0          # map to -1.0 … +1.0
            label = data["data"][0]["value_classification"]
            logger.info("Fear & Greed Index: %d (%s) → score=%.3f", value, label, score)
            return score
        except Exception as e:
            logger.error("Fear & Greed fetch error: %s", e)
            return 0.0

    @property
    def available(self) -> bool:
        return True
