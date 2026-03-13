from __future__ import annotations

import os
from functools import lru_cache
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── Exchange ──────────────────────────────────────────────────────────────
    bitvavo_api_key: str = ""
    bitvavo_api_secret: str = ""
    bitvavo_rest_url: str = "https://api.bitvavo.com/v2"
    bitvavo_ws_url: str = "wss://ws.bitvavo.com/v2/"

    # ── Trading mode ──────────────────────────────────────────────────────────
    paper_trading: bool = True        # MUST be set to false explicitly for live
    max_pairs: int = 25               # top N pairs by 24h Bitvavo volume
    base_currency: str = "EUR"
    primary_interval: str = "5m"     # candle interval for strategy decisions
    strategy_cycle_secs: int = 30    # how often the strategy loop runs

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://hellacash:hellacash@localhost:5432/hellacash"

    # ── Social media sentiment ─────────────────────────────────────────────────
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "hellacash/1.0"
    twitter_bearer_token: str = ""   # optional; snscrape fallback if empty
    sentiment_poll_interval_secs: int = 600  # 10 minutes

    # ── Risk limits (NEVER modified by learning system) ───────────────────────
    max_drawdown_pct: float = 8.0          # portfolio hard stop (%)
    max_position_size_pct: float = 20.0   # max single position (% of portfolio)
    max_daily_loss_eur: float = 200.0      # daily loss circuit breaker
    max_open_positions: int = 5
    min_trade_roi_pct: float = 0.3         # min expected ROI before fees (%)
    min_signal_confidence: float = 0.60   # min composite confidence to trade
    kelly_fraction: float = 0.25          # quarter-Kelly sizing

    # ── API server ────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── Learning ──────────────────────────────────────────────────────────────
    optimizer_min_trades: int = 50         # minimum trades before optimizer runs
    model_dir: str = "./models"

    @field_validator("database_url")
    @classmethod
    def check_db_url(cls, v: str) -> str:
        if not v:
            raise ValueError("DATABASE_URL must be set")
        return v

    @property
    def is_live(self) -> bool:
        return not self.paper_trading


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
