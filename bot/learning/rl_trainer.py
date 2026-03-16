"""PPO training loop for trading parameter optimization."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class RLTrainer:
    """Train and validate PPO models for each strategy."""

    def __init__(self, candle_store, strategies: List[str],
                 model_dir: str = "models/rl_optimizer") -> None:
        self._candle_store = candle_store
        self._strategies = strategies
        self._model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def train(self, total_timesteps: int = 25_000) -> Dict[str, float]:
        """Full training run. Returns {strategy: avg_reward}."""
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

        from bot.learning.rl_environment import TradingParamEnv

        symbols = self._candle_store.available_symbols()
        if not symbols:
            logger.warning("RLTrainer: no symbols with candle data, skipping")
            return {}

        n_envs = min(os.cpu_count() or 1, 4)
        results = {}

        for strategy in self._strategies:
            logger.info("Training %s model (%d timesteps, %d envs)...",
                        strategy, total_timesteps, n_envs)

            def make_env(s=strategy):
                return TradingParamEnv(self._candle_store, symbols, s)

            # Try SubprocVecEnv, fall back to DummyVecEnv
            try:
                vec_env = SubprocVecEnv([make_env for _ in range(n_envs)])
            except Exception as e:
                logger.warning("SubprocVecEnv failed (%s), using DummyVecEnv", e)
                vec_env = DummyVecEnv([make_env for _ in range(n_envs)])

            try:
                model = PPO(
                    "MlpPolicy", vec_env,
                    learning_rate=3e-4,
                    n_steps=2048,
                    batch_size=64,
                    policy_kwargs={"net_arch": [256, 256, 256]},
                    verbose=0,
                )
                model.learn(total_timesteps=total_timesteps)

                # Validate
                avg_reward = self._validate(model, strategy, symbols)
                results[strategy] = avg_reward

                # Check against existing model
                existing_path = os.path.join(self._model_dir, f"{strategy}_ppo.zip")
                if os.path.exists(existing_path):
                    old_model = PPO.load(existing_path)
                    old_reward = self._validate(old_model, strategy, symbols)
                    if avg_reward <= old_reward:
                        logger.warning(
                            "%s: new model (%.3f) not better than old (%.3f), keeping old",
                            strategy, avg_reward, old_reward,
                        )
                        continue

                # Atomic save
                tmp_path = os.path.join(self._model_dir, f"{strategy}_ppo.tmp.zip")
                final_path = os.path.join(self._model_dir, f"{strategy}_ppo.zip")
                model.save(tmp_path)
                os.replace(tmp_path, final_path)
                logger.info("%s: model saved (avg_reward=%.3f)", strategy, avg_reward)

            finally:
                vec_env.close()

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
        for strategy in self._strategies:
            model_path = os.path.join(self._model_dir, f"{strategy}_ppo.zip")
            if not os.path.exists(model_path):
                logger.warning("No existing model for %s, skipping retrain", strategy)
                continue

            env = DummyVecEnv([
                lambda s=strategy: TradingParamEnv(self._candle_store, symbols, s)
            ])

            try:
                model = PPO.load(model_path, env=env)
                model.learn(total_timesteps=total_timesteps)

                avg_reward = self._validate(model, strategy, symbols)
                results[strategy] = avg_reward

                # Validation gate
                old_model = PPO.load(model_path)
                old_reward = self._validate(old_model, strategy, symbols)
                if avg_reward <= old_reward:
                    logger.warning(
                        "%s retrain: new (%.3f) not better than old (%.3f), keeping old",
                        strategy, avg_reward, old_reward,
                    )
                    continue

                tmp_path = os.path.join(self._model_dir, f"{strategy}_ppo.tmp.zip")
                model.save(tmp_path)
                os.replace(tmp_path, model_path)
                logger.info("%s: retrained model saved (%.3f)", strategy, avg_reward)

            finally:
                env.close()

        return results

    def _validate(self, model, strategy: str, symbols: List[str],
                  n_episodes: int = 20) -> float:
        """Run model on validation episodes, return average reward."""
        from bot.learning.rl_environment import TradingParamEnv

        env = TradingParamEnv(self._candle_store, symbols, strategy)
        total_reward = 0.0
        for _ in range(n_episodes):
            obs, _ = env.reset()
            action, _ = model.predict(obs, deterministic=True)
            _, reward, _, _, _ = env.step(action)
            total_reward += reward
        return total_reward / max(n_episodes, 1)
