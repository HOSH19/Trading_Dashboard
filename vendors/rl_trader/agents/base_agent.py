"""Abstract base class for RL agents."""

from abc import ABC, abstractmethod
from typing import Optional

import gymnasium as gym
import numpy as np


class BaseAgent(ABC):
    """Shared interface for PPO, SAC, and any future RL agents."""

    @abstractmethod
    def train(
        self,
        env: gym.Env,
        total_timesteps: int,
        eval_env: Optional[gym.Env] = None,
        checkpoint_dir: Optional[str] = None,
    ) -> "BaseAgent":
        """Train the agent for total_timesteps steps."""

    @abstractmethod
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Return action for the given observation."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist model to disk."""

    @abstractmethod
    def load(self, path: str, env: Optional[gym.Env] = None) -> "BaseAgent":
        """Load model from disk."""

    def portfolio_weights(
        self,
        obs: np.ndarray,
        symbols: list,
        deterministic: bool = True,
    ) -> dict:
        """
        Convenience: return {symbol: weight} dict from a raw observation.
        The last action element is the cash weight (excluded from the dict).
        """
        action = self.predict(obs, deterministic=deterministic)
        return {sym: float(action[i]) for i, sym in enumerate(symbols)}
