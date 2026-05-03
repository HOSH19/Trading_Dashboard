"""Observation space builder: market features + portfolio state → flat numpy vector."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from env.portfolio import PortfolioState


_MAX_DAYS_HELD = 252.0
_MAX_DAYS_SINCE_REBALANCE = 63.0  # one quarter


class ObservationBuilder:
    """
    Concatenates three blocks into a flat float32 observation vector:

      Block 1 — Market features:  F features × N assets (flattened, z-scored)
      Block 2 — Portfolio state:  weights, P&L, drawdown, cash, timing
      Block 3 — Episode context:  progress through episode

    All values are clipped to [-5, 5] after assembly to keep observations
    bounded for the neural network.
    """

    def __init__(self, symbols: List[str], n_features: int) -> None:
        self.symbols = symbols
        self.n_features = n_features
        # market: n_features * n_symbols
        # portfolio: weights(n) + pnl(n) + [drawdown, cash_frac, days_norm, cash_frac] = n*2 + 4
        # context: 1 (episode progress)
        self._obs_dim = n_features * len(symbols) + len(symbols) * 2 + 4 + 1

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    def build(
        self,
        feature_rows: Dict[str, np.ndarray],
        portfolio: PortfolioState,
        episode_len: int,
        noise_sigma: float = 0.0,
    ) -> np.ndarray:
        """
        Build a flat observation vector.

        Args:
            feature_rows:  {symbol: 1D array of length n_features} for current bar.
            portfolio:     Current PortfolioState from Portfolio.state().
            episode_len:   Total episode length (for progress normalization).
            noise_sigma:   Std of Gaussian noise added to market block (data augmentation).

        Returns:
            float32 array of shape (obs_dim,).
        """
        # Block 1: Market features (n_features × n_symbols), flattened
        market_blocks = []
        for sym in self.symbols:
            row = feature_rows.get(sym, np.zeros(self.n_features))
            row = np.nan_to_num(row, nan=0.0, posinf=3.0, neginf=-3.0)
            if noise_sigma > 0:
                row = row + np.random.normal(0, noise_sigma, size=row.shape)
            market_blocks.append(row)
        market_block = np.concatenate(market_blocks)

        # Block 2: Portfolio state
        weights = np.array([portfolio.weights.get(sym, 0.0) for sym in self.symbols])
        pnl_pcts = np.array([
            np.clip(portfolio.unrealized_pnl_pct.get(sym, 0.0), -1.0, 1.0)
            for sym in self.symbols
        ])
        portfolio_block = np.concatenate([
            weights,
            pnl_pcts,
            [
                np.clip(portfolio.drawdown_from_peak, -1.0, 0.0),
                np.clip(portfolio.cash_fraction, 0.0, 1.0),
                min(portfolio.days_since_rebalance / _MAX_DAYS_SINCE_REBALANCE, 1.0),
                portfolio.cash_fraction,
            ],
        ])

        # Block 3: Episode context
        progress = portfolio.step / max(episode_len, 1)
        context_block = np.array([progress], dtype=np.float32)

        obs = np.concatenate([market_block, portfolio_block, context_block]).astype(np.float32)
        return np.clip(obs, -5.0, 5.0)
