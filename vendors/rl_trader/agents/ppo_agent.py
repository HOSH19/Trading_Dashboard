"""PPO agent wrapper over Stable-Baselines3."""

import logging
import os
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
import warnings
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class PPOAgent(BaseAgent):
    """
    PPO agent for continuous portfolio weight allocation.

    Why PPO:
    - On-policy updates avoid replay buffer staleness in non-stationary markets.
    - clip_range=0.2 prevents destructive updates from noisy financial rewards.
    - Entropy bonus (ent_coef) keeps policy exploratory during training.
    """

    def __init__(self, config: dict) -> None:
        agent_cfg = config.get("agent", {})
        self._config = config
        self._model: Optional[PPO] = None

        self._ppo_kwargs = {
            "policy": "MlpPolicy",
            "learning_rate": agent_cfg.get("learning_rate", 3e-4),
            "n_steps": agent_cfg.get("n_steps", 252),
            "batch_size": agent_cfg.get("batch_size", 64),
            "n_epochs": agent_cfg.get("n_epochs", 10),
            "gamma": agent_cfg.get("gamma", 0.99),
            "gae_lambda": agent_cfg.get("gae_lambda", 0.95),
            "clip_range": agent_cfg.get("clip_range", 0.2),
            "ent_coef": agent_cfg.get("ent_coef", 0.01),
            "vf_coef": agent_cfg.get("vf_coef", 0.5),
            "max_grad_norm": agent_cfg.get("max_grad_norm", 0.5),
            "policy_kwargs": {
                "net_arch": agent_cfg.get("net_arch", [256, 128]),
                "activation_fn": torch.nn.Tanh,
            },
            "tensorboard_log": agent_cfg.get("tensorboard_log", None),
            "verbose": 0,
        }

    def initialize(self, env: gym.Env) -> "PPOAgent":
        """Create a fresh PPO model. Call after env is available."""
        self._model = PPO(env=env, **self._ppo_kwargs)
        logger.info(
            "PPO initialized | obs=%s | act=%s | net=%s",
            env.observation_space.shape,
            env.action_space.shape,
            self._ppo_kwargs["policy_kwargs"]["net_arch"],
        )
        return self

    def train(
        self,
        env: gym.Env,
        total_timesteps: int,
        eval_env: Optional[gym.Env] = None,
        checkpoint_dir: Optional[str] = None,
    ) -> "PPOAgent":
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
                name_prefix="ppo",
                verbose=0,
            ))

        logger.info("Training PPO for %d timesteps...", total_timesteps)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Monitor.*")
            self._model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks or None,
                reset_num_timesteps=False,
                progress_bar=True,
            )
        logger.info("Training complete.")
        return self

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Agent not initialized. Call initialize() or load() first.")
        action, _ = self._model.predict(obs, deterministic=deterministic)
        return np.clip(action, 0.0, 1.0).astype(np.float32)

    def save(self, path: str) -> None:
        if self._model:
            self._model.save(path)
            logger.info("PPO model saved to %s", path)

    def load(self, path: str, env: Optional[gym.Env] = None) -> "PPOAgent":
        self._model = PPO.load(path, env=env)
        logger.info("PPO model loaded from %s", path)
        return self
