"""SAC agent wrapper over Stable-Baselines3 (comparison agent)."""

import logging
import os
from typing import Optional

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class SACAgent(BaseAgent):
    """
    SAC (Soft Actor-Critic) agent for comparison experiments.

    SAC advantages over PPO:
    - Higher sample efficiency via off-policy replay buffer.
    - Automatic entropy tuning.

    SAC risks in finance:
    - Replay buffer mixes experiences from different market regimes (staleness).
    - Use PPO as default; SAC for sample-efficiency ablation studies.
    """

    def __init__(self, config: dict) -> None:
        agent_cfg = config.get("agent", {})
        self._config = config
        self._model: Optional[SAC] = None

        self._sac_kwargs = {
            "policy": "MlpPolicy",
            "learning_rate": agent_cfg.get("learning_rate", 3e-4),
            "buffer_size": agent_cfg.get("buffer_size", 100_000),
            "batch_size": agent_cfg.get("batch_size", 256),
            "gamma": agent_cfg.get("gamma", 0.99),
            "tau": agent_cfg.get("tau", 0.005),
            "ent_coef": "auto",
            "policy_kwargs": {
                "net_arch": agent_cfg.get("net_arch", [256, 128]),
            },
            "tensorboard_log": agent_cfg.get("tensorboard_log", None),
            "verbose": 0,
        }

    def initialize(self, env: gym.Env) -> "SACAgent":
        self._model = SAC(env=env, **self._sac_kwargs)
        logger.info("SAC initialized | obs=%s | act=%s", env.observation_space.shape, env.action_space.shape)
        return self

    def train(
        self,
        env: gym.Env,
        total_timesteps: int,
        eval_env: Optional[gym.Env] = None,
        checkpoint_dir: Optional[str] = None,
    ) -> "SACAgent":
        if self._model is None:
            self.initialize(env)
        else:
            self._model.set_env(env)

        callbacks = []
        if eval_env is not None and checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
            callbacks.append(EvalCallback(
                eval_env,
                best_model_save_path=checkpoint_dir,
                log_path=checkpoint_dir,
                eval_freq=max(total_timesteps // 20, 252),
                n_eval_episodes=3,
                deterministic=True,
                verbose=0,
            ))
        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
            callbacks.append(CheckpointCallback(
                save_freq=max(total_timesteps // 10, 252),
                save_path=checkpoint_dir,
                name_prefix="sac",
                verbose=0,
            ))

        logger.info("Training SAC for %d timesteps...", total_timesteps)
        self._model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks or None,
            reset_num_timesteps=False,
            progress_bar=True,
        )
        return self

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Agent not initialized. Call initialize() or load() first.")
        action, _ = self._model.predict(obs, deterministic=deterministic)
        return np.clip(action, 0.0, 1.0).astype(np.float32)

    def save(self, path: str) -> None:
        if self._model:
            self._model.save(path)

    def load(self, path: str, env: Optional[gym.Env] = None) -> "SACAgent":
        self._model = SAC.load(path, env=env)
        return self
