"""
Pre-train the RLSignalEvaluator on historical backtest data.

Runs a full backtest with forced-trade mode (takes ALL signals regardless of
confidence), collects (observation, outcome) pairs, then does batch PPO
training. The resulting model can then be used with online_learning=True
for continued adaptation.

Usage:
    python scripts/pretrain_signal_evaluator.py
"""
import sys, os, time
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
from bot.exchange.bitvavo_client import BitvavoClient
from bot.backtest.engine import BacktestEngine
from bot.learning.rl_signal_evaluator import (
    RLSignalEvaluator, build_signal_obs, OBS_DIM, BUFFER_SIZE,
)
from bot.strategy.adopted_universe import CHAMPION_DEFAULTS

PAIRS = ["BTC-EUR", "ETH-EUR", "XRP-EUR"]
YEARS = 5
DAYS = YEARS * 365
CAPITAL = 10_000.0
MODEL_PATH = "models/signal_evaluator/pretrained.npz"

# Training hyperparameters
PRETRAIN_EPOCHS = 10  # number of full passes over collected data
BATCH_SIZE = 64


def collect_training_data(candles, pair, ml_signal_generator=None):
    """Run a backtest that takes ALL signals and collects (obs, reward) pairs.

    We use the engine with signal_evaluator=None (champion defaults) and
    reconstruct the observations + rewards from the trade log.
    """
    # Run champion backtest to get trade outcomes
    engine = BacktestEngine(
        candles, initial_capital=CAPITAL, max_open_positions=10,
        strategy_params=dict(CHAMPION_DEFAULTS),
        ml_signal_generator=ml_signal_generator,
    )
    result = engine.run()

    # Now run again with a dummy evaluator that always says "take"
    # to collect observations for each signal
    collector = _ObsCollector(ml_signal_generator=ml_signal_generator)
    engine2 = BacktestEngine(
        candles, initial_capital=CAPITAL, max_open_positions=10,
        strategy_params=dict(CHAMPION_DEFAULTS),
        signal_evaluator=collector,
        online_learning=True,
        ml_signal_generator=ml_signal_generator,
    )
    engine2.run()

    return collector.experiences


class _ObsCollector(RLSignalEvaluator):
    """A modified evaluator that always takes the trade and records everything."""

    def __init__(self, ml_signal_generator=None):
        super().__init__()
        self._ml_signal_generator = ml_signal_generator
        self.experiences = []  # list of (obs, reward) tuples

    def evaluate(self, obs, deterministic=False):
        """Always return take_trade=True to collect all signals."""
        from bot.learning.rl_signal_evaluator import EvalResult
        # Use the parent's forward pass to get action values, but force take_trade=True
        result = super().evaluate(obs, deterministic=False)
        result.take_trade = True  # force all trades
        return result

    def record_outcome(self, eval_result, reward):
        """Record the experience for batch training later."""
        if eval_result.obs is not None:
            self.experiences.append((
                eval_result.obs.copy(),
                eval_result.action.copy(),
                eval_result.log_prob,
                reward,
            ))
        return False  # don't trigger PPO during collection


def batch_train(evaluator, all_experiences, epochs=PRETRAIN_EPOCHS):
    """Train the evaluator on collected experiences using batch PPO."""
    n = len(all_experiences)
    if n < 10:
        print(f"  Too few experiences ({n}) to train")
        return

    obs_arr = np.stack([e[0] for e in all_experiences])
    act_arr = np.stack([e[1] for e in all_experiences])
    logp_arr = np.array([e[2] for e in all_experiences], dtype=np.float32)
    rew_arr = np.array([e[3] for e in all_experiences], dtype=np.float32)

    print(f"  Training on {n} experiences for {epochs} epochs...")
    print(f"  Reward stats: mean={rew_arr.mean():.4f}, std={rew_arr.std():.4f}, "
          f"positive={(rew_arr > 0).sum()}/{n} ({(rew_arr > 0).mean()*100:.1f}%)")

    for epoch in range(epochs):
        # Shuffle
        indices = np.random.permutation(n)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        batches = 0

        for start in range(0, n - BATCH_SIZE + 1, BATCH_SIZE):
            batch_idx = indices[start:start + BATCH_SIZE]

            # Load batch into evaluator buffer
            for idx in batch_idx:
                obs_i = obs_arr[idx]
                act_i = act_arr[idx]
                lp_i = logp_arr[idx]
                rew_i = rew_arr[idx]

                _, _, value = evaluator._forward(obs_i)
                evaluator._obs_buf.append(obs_i)
                evaluator._act_buf.append(act_i)
                evaluator._logp_buf.append(lp_i)
                evaluator._rew_buf.append(rew_i)
                evaluator._val_buf.append(value)

            # Trigger PPO update
            evaluator._ppo_update()
            batches += 1

        # Evaluate on full dataset
        correct_takes = 0
        correct_skips = 0
        for i in range(min(n, 500)):
            result = evaluator.evaluate(obs_arr[i], deterministic=True)
            if rew_arr[i] > 0 and result.take_trade:
                correct_takes += 1
            elif rew_arr[i] <= 0 and not result.take_trade:
                correct_skips += 1

        accuracy = (correct_takes + correct_skips) / min(n, 500) * 100
        print(f"  Epoch {epoch+1}/{epochs}: {batches} batches, "
              f"accuracy={accuracy:.1f}% (takes={correct_takes}, skips={correct_skips})")


def main():
    print("=" * 100)
    print(f"  PRE-TRAINING SIGNAL EVALUATOR on {YEARS}-year historical data")
    print(f"  Pairs: {', '.join(PAIRS)} | Capital: EUR {CAPITAL:,.0f}")
    print("=" * 100)

    # Optionally load ML signal generator
    ml_gen = None
    model_dir = "models/ml_signals"
    if Path(model_dir).exists() and (Path(model_dir) / "lstm.pt").exists():
        from bot.learning.ml_signal_generator import MLSignalGenerator
        ml_gen = MLSignalGenerator(model_dir=model_dir)
        print(f"  Using ML Signal Generator for pre-training")
    else:
        print(f"  ML models not found — using strategy-based signals for pre-training")

    # Fetch candles
    all_experiences = []
    for pair in PAIRS:
        client = BitvavoClient(api_key="", api_secret="", paper_trading=True)
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=DAYS)
        print(f"\n  Fetching {pair}...", end="", flush=True)
        candles = client.get_candles_range(pair, interval="5m", start=start, end=end)
        if not candles or len(candles) < 1000:
            print(f" skipped (insufficient data)")
            continue
        print(f" {len(candles):,} candles")

        print(f"  Collecting training data for {pair}...")
        t0 = time.time()
        experiences = collect_training_data(candles, pair, ml_signal_generator=ml_gen)
        elapsed = time.time() - t0
        print(f"  Got {len(experiences)} trade experiences ({elapsed:.0f}s)")
        all_experiences.extend(experiences)
        time.sleep(1)

    print(f"\n  Total experiences collected: {len(all_experiences)}")

    # Create and train evaluator
    evaluator = RLSignalEvaluator()

    # Batch training
    batch_train(evaluator, all_experiences, epochs=PRETRAIN_EPOCHS)

    # Save
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    evaluator.save(MODEL_PATH)
    print(f"\n  Model saved to {MODEL_PATH}")

    # Verify: run a quick backtest with the pre-trained model
    print(f"\n  Verification: running 6-month backtest with pre-trained model...")
    client = BitvavoClient(api_key="", api_secret="", paper_trading=True)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=180)
    candles = client.get_candles_range("BTC-EUR", interval="5m", start=start, end=end)

    # Baseline
    engine_base = BacktestEngine(
        candles, initial_capital=CAPITAL, strategy_params=dict(CHAMPION_DEFAULTS),
        ml_signal_generator=ml_gen,
    )
    bt_base = engine_base.run()

    # Pre-trained RL
    pretrained = RLSignalEvaluator()
    pretrained.load(MODEL_PATH)
    engine_rl = BacktestEngine(
        candles, initial_capital=CAPITAL, strategy_params=dict(CHAMPION_DEFAULTS),
        signal_evaluator=pretrained, online_learning=True,
        ml_signal_generator=ml_gen,
    )
    bt_rl = engine_rl.run()

    print(f"  Champion Defaults: {bt_base.total_trades} trades, EUR {bt_base.total_pnl:+.2f}, "
          f"Sharpe={bt_base.sharpe_ratio:+.2f}")
    print(f"  Pre-trained RL:    {bt_rl.total_trades} trades, EUR {bt_rl.total_pnl:+.2f}, "
          f"Sharpe={bt_rl.sharpe_ratio:+.2f}")
    stats = pretrained.stats
    print(f"  Online updates during verification: {stats['updates']}")

    print(f"\n  Done! Use the pre-trained model with:")
    print(f"    evaluator = RLSignalEvaluator()")
    print(f"    evaluator.load('{MODEL_PATH}')")


if __name__ == "__main__":
    main()
