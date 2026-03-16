"""PPO training loop for trading parameter optimization."""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class _TrainingLogger:
    """Callback to log PPO training progress at regular intervals."""

    def __init__(self, strategy: str, total_timesteps: int, log_every: int = 1024):
        self._strategy = strategy
        self._total = total_timesteps
        self._log_every = log_every
        self._step = 0
        self._start = time.time()

    def __call__(self, locals_: dict, globals_: dict) -> bool:
        self._step += 1
        if self._step % self._log_every == 0:
            elapsed = time.time() - self._start
            pct = (self._step / self._total) * 100 if self._total else 0
            # Extract PPO training stats
            ep_rew = locals_.get("self")
            info = ""
            if ep_rew and hasattr(ep_rew, "logger") and hasattr(ep_rew.logger, "name_to_value"):
                vals = ep_rew.logger.name_to_value
                parts = []
                for key in ["train/entropy_loss", "train/policy_gradient_loss",
                            "train/value_loss", "train/approx_kl"]:
                    if key in vals:
                        parts.append(f"{key.split('/')[-1]}={vals[key]:.4f}")
                if parts:
                    info = " | " + ", ".join(parts)
            logger.info(
                "[%s] step %d/%d (%.0f%%) elapsed=%.0fs%s",
                self._strategy, self._step, self._total, pct, elapsed, info,
            )
        return True  # continue training


class RLTrainer:
    """Train and validate PPO models for each strategy."""

    def __init__(self, candle_store, strategies: List[str],
                 model_dir: str = "models/rl_optimizer") -> None:
        self._candle_store = candle_store
        self._strategies = strategies
        self._model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def train(self, total_timesteps: int = 10_000) -> Dict[str, float]:
        """Full training run. Returns {strategy: avg_reward}."""
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

        from bot.learning.rl_environment import TradingParamEnv

        symbols = self._candle_store.available_symbols()
        if not symbols:
            logger.warning("RLTrainer: no symbols with candle data, skipping")
            return {}

        logger.info("RLTrainer: %d symbols available for training: %s",
                     len(symbols), ", ".join(symbols[:10]) + ("..." if len(symbols) > 10 else ""))

        n_envs = min(os.cpu_count() or 1, 4)
        results = {}
        overall_start = time.time()

        for strategy in self._strategies:
            strat_start = time.time()
            logger.info(
                "=== Training %s model === timesteps=%d, envs=%d, symbols=%d",
                strategy, total_timesteps, n_envs, len(symbols),
            )

            def make_env(s=strategy):
                return TradingParamEnv(self._candle_store, symbols, s)

            # Try SubprocVecEnv, fall back to DummyVecEnv
            try:
                vec_env = SubprocVecEnv([make_env for _ in range(n_envs)])
                logger.info("[%s] Using SubprocVecEnv (%d workers)", strategy, n_envs)
            except Exception as e:
                logger.warning("[%s] SubprocVecEnv failed (%s), using DummyVecEnv", strategy, e)
                vec_env = DummyVecEnv([make_env for _ in range(n_envs)])

            try:
                model = PPO(
                    "MlpPolicy", vec_env,
                    learning_rate=3e-4,
                    n_steps=1024,
                    batch_size=64,
                    policy_kwargs={"net_arch": [256, 256, 256]},
                    verbose=0,
                )

                train_cb = _TrainingLogger(strategy, total_timesteps, log_every=1024)
                model.learn(total_timesteps=total_timesteps, callback=train_cb)
                train_elapsed = time.time() - strat_start
                logger.info("[%s] Training done in %.1fs", strategy, train_elapsed)

                # Validate
                avg_reward, val_stats = self._validate(model, strategy, symbols)
                results[strategy] = avg_reward
                logger.info(
                    "[%s] Validation: avg_reward=%.3f, avg_sharpe=%.2f, avg_pnl=€%.2f, "
                    "avg_pf=%.2f, avg_trades=%.0f, avg_win_rate=%.1f%%, avg_drawdown=%.1f%%",
                    strategy, avg_reward, val_stats["avg_sharpe"], val_stats["avg_pnl"],
                    val_stats["avg_pf"], val_stats["avg_trades"],
                    val_stats["avg_win_rate"], val_stats["avg_drawdown"],
                )

                # Check against existing model
                existing_path = os.path.join(self._model_dir, f"{strategy}_ppo.zip")
                if os.path.exists(existing_path):
                    old_model = PPO.load(existing_path)
                    old_reward, old_stats = self._validate(old_model, strategy, symbols)
                    logger.info(
                        "[%s] Old model: avg_reward=%.3f, avg_sharpe=%.2f, avg_pnl=€%.2f",
                        strategy, old_reward, old_stats["avg_sharpe"], old_stats["avg_pnl"],
                    )
                    if avg_reward <= old_reward:
                        logger.warning(
                            "[%s] New model (%.3f) not better than old (%.3f), keeping old",
                            strategy, avg_reward, old_reward,
                        )
                        continue

                # Atomic save
                tmp_path = os.path.join(self._model_dir, f"{strategy}_ppo.tmp.zip")
                final_path = os.path.join(self._model_dir, f"{strategy}_ppo.zip")
                model.save(tmp_path)
                os.replace(tmp_path, final_path)
                logger.info("[%s] Model saved (avg_reward=%.3f, time=%.1fs)",
                            strategy, avg_reward, time.time() - strat_start)

            finally:
                vec_env.close()

        total_elapsed = time.time() - overall_start
        logger.info(
            "=== RLTrainer complete === strategies=%d, total_time=%.1fs, results=%s",
            len(self._strategies), total_elapsed,
            {s: f"{r:.3f}" for s, r in results.items()},
        )
        return results

    def retrain(self, fine_tune_months: int = 6,
                total_timesteps: int = 5_000) -> Dict[str, float]:
        """Weekly incremental retrain on recent data."""
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        from bot.learning.rl_environment import TradingParamEnv

        symbols = self._candle_store.available_symbols()
        if not symbols:
            return {}

        results = {}
        overall_start = time.time()

        for strategy in self._strategies:
            model_path = os.path.join(self._model_dir, f"{strategy}_ppo.zip")
            if not os.path.exists(model_path):
                logger.warning("No existing model for %s, skipping retrain", strategy)
                continue

            strat_start = time.time()
            logger.info("=== Retraining %s === timesteps=%d", strategy, total_timesteps)

            env = DummyVecEnv([
                lambda s=strategy: TradingParamEnv(self._candle_store, symbols, s)
            ])

            try:
                model = PPO.load(model_path, env=env)

                train_cb = _TrainingLogger(strategy, total_timesteps, log_every=1024)
                model.learn(total_timesteps=total_timesteps, callback=train_cb)
                logger.info("[%s] Retrain done in %.1fs", strategy, time.time() - strat_start)

                avg_reward, val_stats = self._validate(model, strategy, symbols)
                results[strategy] = avg_reward
                logger.info(
                    "[%s] Retrain validation: avg_reward=%.3f, avg_sharpe=%.2f, "
                    "avg_pnl=€%.2f, avg_pf=%.2f, avg_trades=%.0f",
                    strategy, avg_reward, val_stats["avg_sharpe"],
                    val_stats["avg_pnl"], val_stats["avg_pf"], val_stats["avg_trades"],
                )

                # Validation gate
                old_model = PPO.load(model_path)
                old_reward, old_stats = self._validate(old_model, strategy, symbols)
                logger.info(
                    "[%s] Old model: avg_reward=%.3f, avg_sharpe=%.2f, avg_pnl=€%.2f",
                    strategy, old_reward, old_stats["avg_sharpe"], old_stats["avg_pnl"],
                )
                if avg_reward <= old_reward:
                    logger.warning(
                        "[%s] Retrain: new (%.3f) not better than old (%.3f), keeping old",
                        strategy, avg_reward, old_reward,
                    )
                    continue

                tmp_path = os.path.join(self._model_dir, f"{strategy}_ppo.tmp.zip")
                model.save(tmp_path)
                os.replace(tmp_path, model_path)
                logger.info("[%s] Retrained model saved (%.3f)", strategy, avg_reward)

            finally:
                env.close()

        total_elapsed = time.time() - overall_start
        logger.info(
            "=== Retrain complete === time=%.1fs, results=%s",
            total_elapsed, {s: f"{r:.3f}" for s, r in results.items()},
        )
        return results

    def _validate(self, model, strategy: str, symbols: List[str],
                  n_episodes: int = 20) -> tuple:
        """Run model on validation episodes, return (avg_reward, stats_dict)."""
        from bot.learning.rl_environment import TradingParamEnv

        env = TradingParamEnv(self._candle_store, symbols, strategy,
                              validation_mode=True)
        rewards = []
        sharpes = []
        pnls = []
        pfs = []
        trade_counts = []
        win_rates = []
        drawdowns = []

        for ep in range(n_episodes):
            obs, _ = env.reset()
            action, _ = model.predict(obs, deterministic=True)
            _, reward, _, _, info = env.step(action)
            rewards.append(reward)

            result = info.get("result")
            if result:
                sharpes.append(result.sharpe_ratio)
                pnls.append(result.total_pnl)
                pf = result.profit_factor
                pfs.append(pf if not (pf == float('inf') or pf != pf) else 0.0)
                trade_counts.append(result.total_trades)
                win_rates.append(result.win_rate)
                drawdowns.append(result.max_drawdown_pct)

                if (ep + 1) % 5 == 0 or ep == 0:
                    logger.info(
                        "  [%s] val ep %d/%d: reward=%.2f, sharpe=%.2f, pnl=€%.2f, "
                        "pf=%.2f, trades=%d, win=%.0f%%, dd=%.1f%%",
                        strategy, ep + 1, n_episodes, reward,
                        result.sharpe_ratio, result.total_pnl,
                        result.profit_factor, result.total_trades,
                        result.win_rate, result.max_drawdown_pct,
                    )

        avg_reward = np.mean(rewards) if rewards else 0.0
        stats = {
            "avg_sharpe": np.mean(sharpes) if sharpes else 0.0,
            "avg_pnl": np.mean(pnls) if pnls else 0.0,
            "avg_pf": np.mean(pfs) if pfs else 0.0,
            "avg_trades": np.mean(trade_counts) if trade_counts else 0.0,
            "avg_win_rate": np.mean(win_rates) if win_rates else 0.0,
            "avg_drawdown": np.mean(drawdowns) if drawdowns else 0.0,
        }
        return avg_reward, stats
