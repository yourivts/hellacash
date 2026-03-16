"""One-time bootstrap: download 1m candles and train initial RL models.

Usage: python -m bot.scripts.rl_bootstrap
"""
from __future__ import annotations

import asyncio
import logging
import subprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


async def main():
    from bot.config import get_settings
    from bot.data.candle_store import CandleStore
    from bot.learning.rl_trainer import RLTrainer
    from bot.strategy.adopted_universe import ALL_STRATEGIES

    settings = get_settings()
    candle_store = CandleStore(db_url=settings.database_url)

    # Step 1: Run Alembic migration (must happen before download writes to candle_1m)
    logger.info("Running alembic upgrade head to ensure candle_1m table exists...")
    subprocess.run(["alembic", "upgrade", "head"], check=True)

    # Step 2: Download candles
    symbols = None
    try:
        from bot.main import get_tradeable_symbols
        symbols = get_tradeable_symbols()
    except Exception as e:
        logger.warning("Could not get tradeable symbols: %s", e)
    if not symbols:
        symbols = ["BTC-EUR", "ETH-EUR"]
        logger.warning("Could not get tradeable symbols, using fallback: %s", symbols)

    logger.info("Downloading 1m candles for %d symbols...", len(symbols))
    await candle_store.bulk_download(symbols)
    logger.info("Download complete.")

    # Step 3: Train models
    logger.info("Training RL models for strategies: %s", ALL_STRATEGIES)
    trainer = RLTrainer(candle_store, ALL_STRATEGIES)
    results = trainer.train()
    logger.info("Training complete: %s", results)


if __name__ == "__main__":
    asyncio.run(main())
