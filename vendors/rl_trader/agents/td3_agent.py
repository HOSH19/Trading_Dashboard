"""TD3 agent wrapper over Stable-Baselines3."""

import logging
import os
from typing import Optional

import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.noise import NormalActionNoise

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class TD3Agent(BaseAgent):
    def __init__(self, config: dict) -> None:
        agent_cfg = config.get("agent", {})
        self._config = config
        self._model: Optional[TD3] = None

        self._td3_kwargs = {
            "policy": "MlpPolicy",
            "learning_rate": agent_cfg.get("learning_rate", 3e-4),
            "buffer_size": agent_cfg.get("buffer_size", 100_000),
            "batch_size": agent_cfg.get("batch_size", 256),
            "gamma": agent_cfg.get("gamma", 0.99),
            "tau": agent_cfg.get("tau", 0.005),
            "policy_delay": agent_cfg.get("policy_delay", 2),
            "target_policy_noise": agent_cfg.get("target_policy_noise", 0.2),
            "target_noise_clip": agent_cfg.get("target_noise_clip", 0.5),
            "policy_kwargs": {
                "net_arch": agent_cfg.get("net_arch", [256, 128]),
            },
            "tensorboard_log": agent_cfg.get("tensorboard_log", None),
            "verbose": 0,
        }

    def initialize(self, env: gym.Env) -> "TD3Agent":
        n_actions = env.action_space.shape[0]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions),
        )
        self._model = TD3(env=env, action_noise=action_noise, **self._td3_kwargs)
        logger.info("TD3 initialized | obs=%s | act=%s", env.observation_space.shape, env.action_space.shape)
        return self

    def train(
        self,
        env: gym.Env,
        total_timesteps: int,
        eval_env: Optional[gym.Env] = None,
        checkpoint_dir: Optional[str] = None,
    ) -> "TD3Agent":
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
                name_prefix="td3",
                verbose=0,
            ))

        logger.info("Training TD3 for %d timesteps...", total_timesteps)
        self._model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks or None,
            reset_num_timesteps=False,
            progress_bar=True,
        )
        return self

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Agent not initialized.")
        action, _ = self._model.predict(obs, deterministic=deterministic)
        return np.clip(action, 0.0, 1.0).astype(np.float32)

    def save(self, path: str) -> None:
        if self._model:
            self._model.save(path)

    def load(self, path: str, env: Optional[gym.Env] = None) -> "TD3Agent":
        self._model = TD3.load(path, env=env)
        return self
