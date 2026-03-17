"""PPO training loop for trading parameter optimization."""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


_GPU_STRATEGIES = {
    "orderflow", "range", "squeeze", "funding_contrarian",
    "breakout", "trend_following", "momentum", "mean_reversion",
}


def _gpu_available() -> bool:
    """Check if GPU acceleration is available (CuPy + CUDA)."""
    try:
        import cupy as cp
        cp.cuda.Device(0).compute_capability
        return True
    except Exception:
        return False


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
        from stable_baselines3.common.vec_env import DummyVecEnv

        from bot.learning.rl_environment import TradingParamEnv

        symbols = self._candle_store.available_symbols()
        if not symbols:
            logger.warning("RLTrainer: no symbols with candle data, skipping")
            return {}

        logger.info("RLTrainer: %d symbols available for training: %s",
                     len(symbols), ", ".join(symbols[:10]) + ("..." if len(symbols) > 10 else ""))

        use_gpu = _gpu_available()
        if use_gpu:
            from bot.learning.gpu_vec_env import GpuTradingVecEnv
            from bot.learning.gpu_backtest_kernel import compile_kernel, prepare_gpu_data
            n_envs = 512
            logger.info("GPU detected -- using GpuTradingVecEnv (%d envs)", n_envs)

            # Prepare GPU data ONCE for all strategies
            gpu_prep_start = time.time()
            gpu_kernel = compile_kernel()
            gpu_data = prepare_gpu_data(self._candle_store, symbols)
            logger.info("GPU data prepared in %.1fs", time.time() - gpu_prep_start)
        else:
            n_envs = min(os.cpu_count() or 1, 4)

        results = {}
        overall_start = time.time()

        # Warm signal cache if any strategies need CPU env
        has_cpu_strats = any(s not in _GPU_STRATEGIES for s in self._strategies)
        if not use_gpu or has_cpu_strats:
            cache_dir = os.path.join(self._model_dir, ".signal_cache")
            logger.info("Warming signal cache for %d symbols -> %s", len(symbols), cache_dir)
            cache_start = time.time()
            TradingParamEnv.warm_signal_cache(self._candle_store, symbols, cache_dir)
            logger.info("Signal cache ready in %.1fs", time.time() - cache_start)

        for strategy in self._strategies:
            # Skip strategies that already have a trained model (bootstrap resume)
            existing_path = os.path.join(self._model_dir, f"{strategy}_ppo.zip")
            if os.path.exists(existing_path):
                logger.info("=== Skipping %s — model already exists at %s ===",
                            strategy, existing_path)
                continue

            strat_start = time.time()
            strat_gpu = use_gpu and strategy in _GPU_STRATEGIES
            strat_n_envs = n_envs if strat_gpu else min(os.cpu_count() or 1, 4)
            logger.info(
                "=== Training %s model === timesteps=%d, envs=%d, symbols=%d, gpu=%s",
                strategy, total_timesteps, strat_n_envs, len(symbols), strat_gpu,
            )

            if strat_gpu:
                vec_env = GpuTradingVecEnv(
                    self._candle_store, symbols, strategy,
                    n_envs=n_envs, n_segments=3,
                    _gpu_data=gpu_data, _kernel=gpu_kernel,
                )
                logger.info("[%s] Using GpuTradingVecEnv (%d envs)", strategy, n_envs)
                ppo_n_steps = 128
                ppo_batch_size = 256
            else:
                def make_env(s=strategy):
                    return TradingParamEnv(self._candle_store, symbols, s,
                                           n_segments=3)
                vec_env = DummyVecEnv([make_env for _ in range(n_envs)])
                logger.info("[%s] Using DummyVecEnv (%d envs)", strategy, n_envs)
                ppo_n_steps = 512
                ppo_batch_size = 64

            try:
                model = PPO(
                    "MlpPolicy", vec_env,
                    learning_rate=3e-4,
                    n_steps=ppo_n_steps,
                    batch_size=ppo_batch_size,
                    policy_kwargs={"net_arch": [256, 256]},
                    verbose=0,
                )

                train_cb = _TrainingLogger(strategy, total_timesteps, log_every=2048)
                model.learn(total_timesteps=total_timesteps, callback=train_cb)
                train_elapsed = time.time() - strat_start
                logger.info("[%s] Training done in %.1fs", strategy, train_elapsed)

                # Validate
                gpu_kw = dict(gpu_data=gpu_data, gpu_kernel=gpu_kernel) if strat_gpu else {}
                avg_reward, val_stats = self._validate(model, strategy, symbols, **gpu_kw)
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
                    old_model = PPO.load(existing_path, device="cpu")
                    old_reward, old_stats = self._validate(old_model, strategy, symbols, **gpu_kw)
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
                total_timesteps: int = 50_000) -> Dict[str, float]:
        """Weekly incremental retrain on recent data."""
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        from bot.learning.rl_environment import TradingParamEnv

        symbols = self._candle_store.available_symbols()
        if not symbols:
            return {}

        use_gpu = _gpu_available()
        if use_gpu:
            from bot.learning.gpu_vec_env import GpuTradingVecEnv
            from bot.learning.gpu_backtest_kernel import compile_kernel, prepare_gpu_data

            gpu_prep_start = time.time()
            gpu_kernel = compile_kernel()
            gpu_data = prepare_gpu_data(self._candle_store, symbols)
            logger.info("GPU data prepared in %.1fs", time.time() - gpu_prep_start)

        results = {}
        overall_start = time.time()

        has_cpu_strats = any(s not in _GPU_STRATEGIES for s in self._strategies)
        if not use_gpu or has_cpu_strats:
            cache_dir = os.path.join(self._model_dir, ".signal_cache")
            logger.info("Warming signal cache for retrain (%d symbols)", len(symbols))
            TradingParamEnv.warm_signal_cache(self._candle_store, symbols, cache_dir)

        for strategy in self._strategies:
            model_path = os.path.join(self._model_dir, f"{strategy}_ppo.zip")
            if not os.path.exists(model_path):
                logger.warning("No existing model for %s, skipping retrain", strategy)
                continue

            strat_start = time.time()
            strat_gpu = use_gpu and strategy in _GPU_STRATEGIES
            logger.info("=== Retraining %s === timesteps=%d, gpu=%s",
                        strategy, total_timesteps, strat_gpu)

            if strat_gpu:
                env = GpuTradingVecEnv(
                    self._candle_store, symbols, strategy,
                    n_envs=512, n_segments=3,
                    _gpu_data=gpu_data, _kernel=gpu_kernel,
                )
            else:
                env = DummyVecEnv([
                    lambda s=strategy: TradingParamEnv(self._candle_store, symbols, s,
                                                        n_segments=3)
                ])

            try:
                model = PPO.load(model_path, env=env)

                train_cb = _TrainingLogger(strategy, total_timesteps, log_every=1024)
                model.learn(total_timesteps=total_timesteps, callback=train_cb)
                logger.info("[%s] Retrain done in %.1fs", strategy, time.time() - strat_start)

                gpu_kw = dict(gpu_data=gpu_data, gpu_kernel=gpu_kernel) if strat_gpu else {}
                avg_reward, val_stats = self._validate(model, strategy, symbols, **gpu_kw)
                results[strategy] = avg_reward
                logger.info(
                    "[%s] Retrain validation: avg_reward=%.3f, avg_sharpe=%.2f, "
                    "avg_pnl=€%.2f, avg_pf=%.2f, avg_trades=%.0f",
                    strategy, avg_reward, val_stats["avg_sharpe"],
                    val_stats["avg_pnl"], val_stats["avg_pf"], val_stats["avg_trades"],
                )

                # Validation gate
                old_model = PPO.load(model_path, device="cpu")
                old_reward, old_stats = self._validate(old_model, strategy, symbols, **gpu_kw)
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
                  n_episodes: int = 20, gpu_data=None, gpu_kernel=None) -> tuple:
        """Run model on validation episodes, return (avg_reward, stats_dict).

        Uses GPU VecEnv when gpu_data/gpu_kernel are provided, otherwise
        falls back to CPU TradingParamEnv.
        """
        if gpu_data is not None and gpu_kernel is not None:
            return self._validate_gpu(
                model, strategy, symbols, n_episodes, gpu_data, gpu_kernel,
            )

        from bot.learning.rl_environment import TradingParamEnv

        env = TradingParamEnv(self._candle_store, symbols, strategy,
                              validation_mode=True, n_segments=3)
        rewards = []
        sharpes = []
        pnls = []
        pfs = []
        trade_counts = []
        win_rates = []
        drawdowns = []

        for ep in range(n_episodes):
            obs, _ = env.reset()
            ep_reward = 0.0
            ep_pnl = 0.0
            ep_trades = 0
            last_result = None

            # Run through all segments (1 for single-step, n_segments for multi-step)
            terminated = False
            while not terminated:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, _, info = env.step(action)
                ep_reward += reward
                result = info.get("result")
                if result:
                    last_result = result
                    ep_pnl += result.total_pnl
                    ep_trades += result.total_trades

            rewards.append(ep_reward)

            if last_result:
                sharpes.append(last_result.sharpe_ratio)
                pnls.append(ep_pnl)
                pf = last_result.profit_factor
                pfs.append(pf if not (pf == float('inf') or pf != pf) else 0.0)
                trade_counts.append(ep_trades)
                win_rates.append(last_result.win_rate)
                drawdowns.append(last_result.max_drawdown_pct)

                if (ep + 1) % 5 == 0 or ep == 0:
                    logger.info(
                        "  [%s] val ep %d/%d: reward=%.2f, pnl=€%.2f, "
                        "trades=%d, win=%.0f%%, dd=%.1f%%",
                        strategy, ep + 1, n_episodes, ep_reward,
                        ep_pnl, ep_trades,
                        last_result.win_rate, last_result.max_drawdown_pct,
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

    def _validate_gpu(self, model, strategy: str, symbols: List[str],
                      n_episodes: int, gpu_data, gpu_kernel) -> tuple:
        """GPU-accelerated validation using GpuTradingVecEnv."""
        from bot.learning.gpu_vec_env import GpuTradingVecEnv

        env = GpuTradingVecEnv(
            self._candle_store, symbols, strategy,
            n_envs=n_episodes, n_segments=3,
            validation_mode=True,
            _gpu_data=gpu_data, _kernel=gpu_kernel,
        )

        obs = env.reset()
        all_rewards = np.zeros(n_episodes, dtype=np.float64)
        all_pnl = np.zeros(n_episodes, dtype=np.float64)
        all_trades = np.zeros(n_episodes, dtype=np.int32)
        all_sharpe = np.zeros(n_episodes, dtype=np.float64)
        all_dd = np.zeros(n_episodes, dtype=np.float64)
        all_pf = np.zeros(n_episodes, dtype=np.float64)

        # Run through all 3 segments
        for seg in range(3):
            actions, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(actions)
            all_rewards += rewards

            for i in range(n_episodes):
                all_pnl[i] += infos[i].get("segment_pnl", 0.0)
                all_trades[i] += infos[i].get("segment_trades", 0)
                all_sharpe[i] = infos[i].get("segment_sharpe", 0.0)
                all_dd[i] = max(all_dd[i], infos[i].get("segment_max_dd", 0.0))
                all_pf[i] = infos[i].get("segment_profit_factor", 0.0)

        env.close()

        # Compute win rate from trades and wins (approximate from PF)
        avg_reward = float(np.mean(all_rewards))
        stats = {
            "avg_sharpe": float(np.mean(all_sharpe)),
            "avg_pnl": float(np.mean(all_pnl)),
            "avg_pf": float(np.mean(np.clip(all_pf, 0.0, 5.0))),
            "avg_trades": float(np.mean(all_trades)),
            "avg_win_rate": 0.0,  # not tracked per-position in GPU kernel
            "avg_drawdown": float(np.mean(all_dd)),
        }

        logger.info(
            "  [%s] GPU val: avg_reward=%.2f, pnl=%.2f, trades=%.0f, "
            "sharpe=%.2f, dd=%.1f%%",
            strategy, avg_reward, stats["avg_pnl"], stats["avg_trades"],
            stats["avg_sharpe"], stats["avg_drawdown"],
        )

        return avg_reward, stats
