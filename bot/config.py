from __future__ import annotations

import os
from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── Exchange ──────────────────────────────────────────────────────────────
    bitvavo_api_key: str = Field(default="", repr=False)
    bitvavo_api_secret: str = Field(default="", repr=False)
    bitvavo_rest_url: str = "https://api.bitvavo.com/v2"
    bitvavo_ws_url: str = "wss://ws.bitvavo.com/v2/"

    # ── Trading mode ──────────────────────────────────────────────────────────
    paper_trading: bool = True        # MUST be set to false explicitly for live
    max_pairs: int = 100              # top N pairs by 24h Bitvavo volume
    base_currency: str = "EUR"
    primary_interval: str = "5m"     # candle interval for strategy decisions
    strategy_cycle_secs: int = 30    # how often the strategy loop runs
    candle_batch_size: int = 15      # concurrent candle fetches per batch
    min_volume_eur: float = 10000.0  # skip markets with < this 24h volume for trading

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://hellacash:hellacash@localhost:5432/hellacash"

    # ── Social media sentiment ─────────────────────────────────────────────────
    reddit_client_id: str = Field(default="", repr=False)
    reddit_client_secret: str = Field(default="", repr=False)
    reddit_user_agent: str = "linux:com.hellacash.sentiment:v1.0.0 (by /u/yourivts)"
    twitter_bearer_token: str = Field(default="", repr=False)  # optional; snscrape fallback if empty
    sentiment_poll_interval_secs: int = 300  # 5 minutes
    reddit_cache_ttl: int = 300      # seconds to cache reddit data
    news_cache_ttl: int = 300        # seconds to cache news feeds

    # ── Risk limits (NEVER modified by learning system) ───────────────────────
    max_drawdown_pct: float = Field(default=8.0, ge=0, le=100)    # portfolio hard stop (%)
    soft_drawdown_pct: float = Field(default=3.0, ge=0, le=100)   # reduce position sizes by 50%
    max_position_size_pct: float = Field(default=20.0, ge=0, le=100)  # max single position (% of portfolio)
    max_daily_loss_eur: float = Field(default=50.0, ge=0)          # daily loss circuit breaker
    max_open_positions: int = Field(default=10, ge=1)
    min_trade_roi_pct: float = Field(default=0.3, ge=0)            # min expected ROI before fees (%)
    min_signal_confidence: float = Field(default=0.60, ge=0, le=1) # min composite confidence to trade
    kelly_fraction: float = Field(default=0.25, ge=0, le=1)        # quarter-Kelly sizing
    trade_cooldown_hours: float = Field(default=12.0, ge=0)        # hours between trades per symbol
    taker_fee_pct: float = Field(default=0.25, ge=0)               # Bitvavo taker fee (%)
    maker_fee_pct: float = Field(default=0.15, ge=0)               # Bitvavo maker fee (%)

    # ── API server ────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str = Field(default="", repr=False)  # empty = no auth

    # ── Discord notifications ──────────────────────────────────────────────────
    discord_webhook_url: str = Field(default="", repr=False)
    discord_notify_trades: bool = False

    # ── Learning ──────────────────────────────────────────────────────────────
    optimizer_min_trades: int = 50         # minimum trades before optimizer runs
    model_dir: str = "./models"

    # ── Multi-timeframe voting ───────────────────────────────────────────────
    mtf_tech_weight: float = Field(default=0.55, ge=0, le=1)
    mtf_sent_weight: float = Field(default=0.20, ge=0, le=1)
    mtf_onchain_weight: float = Field(default=0.15, ge=0, le=1)
    mtf_book_weight: float = Field(default=0.10, ge=0, le=1)

    # ── On-chain metrics ────────────────────────────────────────────────────
    onchain_enabled: bool = True
    onchain_poll_interval_secs: int = 300

    # ── Order book ───────────────────────────────────────────────────────────
    orderbook_enabled: bool = True
    orderbook_depth_levels: int = 25

    # ── Position sizing ───────────────────────────────────────────────────────
    base_risk_pct: float = 3.0          # % of equity risked per trade (fixed fractional)
    confluence_risk_pct: float = 4.5   # % when confluence gate triggers

    # ── Fee-aware gate ────────────────────────────────────────────────────────
    min_profit_multiple: float = 3.0   # reject if expected_profit < N * total_fees

    # ── Regime detection ─────────────────────────────────────────────────────
    quiet_atr_threshold: float = 1.0   # ATR% below this = QUIET (no trading)
    regime_adx_threshold: float = 25.0 # ADX above this = TRENDING

    # ── Weekend filter ────────────────────────────────────────────────────────
    weekend_filter_enabled: bool = False

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
