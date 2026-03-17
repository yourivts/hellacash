"""Monitor bootstrap + walk-forward and log progress. Restarts on failure.

Usage: docker compose exec -T bot sh -c "nohup python -m bot.scripts.monitor_bootstrap > /tmp/monitor.log 2>&1 &"
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MONITOR] %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_DIR = "models/rl_optimizer"
STRATEGIES = ["orderflow", "range", "squeeze", "funding_contrarian"]


def models_exist() -> dict:
    """Check which strategy models exist on disk."""
    found = {}
    if not os.path.isdir(MODEL_DIR):
        return found
    for s in STRATEGIES:
        path = os.path.join(MODEL_DIR, f"{s}_ppo.zip")
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1_048_576
            found[s] = size_mb
    return found


def bootstrap_running() -> bool:
    """Check if bootstrap process is still running."""
    try:
        result = subprocess.run(
            ["sh", "-c", "cat /proc/*/cmdline 2>/dev/null | tr '\\0' ' '"],
            capture_output=True, text=True, timeout=5,
        )
        return "rl_bootstrap" in result.stdout
    except Exception:
        return False


def run_bootstrap():
    """Run bootstrap and return exit code."""
    logger.info("Starting rl_bootstrap...")
    result = subprocess.run(
        [sys.executable, "-m", "bot.scripts.rl_bootstrap"],
        capture_output=False,
    )
    return result.returncode


def main():
    logger.info("=== Monitor started ===")
    logger.info("Watching for: candle download → model training → walk-forward gate")

    # Phase 1: Run bootstrap (download + train)
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        logger.info("Bootstrap attempt %d/%d", attempt, max_retries)
        exit_code = run_bootstrap()

        if exit_code == 0:
            logger.info("Bootstrap completed successfully (exit code 0)")
            break
        else:
            logger.error("Bootstrap failed with exit code %d", exit_code)
            if attempt < max_retries:
                logger.info("Retrying in 60 seconds...")
                time.sleep(60)
            else:
                logger.error("Bootstrap failed after %d attempts, giving up", max_retries)

    # Phase 2: Verify models
    models = models_exist()
    if models:
        logger.info("=== Models on disk ===")
        for strategy, size in models.items():
            logger.info("  %s: %.1f MB", strategy, size)
        logger.info("  %d/%d strategies trained", len(models), len(STRATEGIES))
    else:
        logger.error("No models found after bootstrap!")

    # Phase 3: Wait for walk-forward to pick up models and complete
    logger.info("Models saved. Walk-forward scheduler will detect them automatically.")
    logger.info("Check bot logs for 'Walk-forward: RL models ready' and 'Walk-forward optimization starting...'")
    logger.info("=== Monitor complete ===")


if __name__ == "__main__":
    main()
