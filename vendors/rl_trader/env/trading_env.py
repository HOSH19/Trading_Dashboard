"""
Core Gymnasium trading environment for multi-asset RL portfolio allocation.

Design principles:
- One step = one trading day (bar).
- Action: raw weight logits → softmax-projected to valid portfolio weights.
- Fill delay: action at bar T fills at bar T+1 open (no lookahead bias).
- Transaction costs and slippage are modeled identically to the backtester.
- Early termination when drawdown exceeds max_dd_terminate.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd

from data.features import compute_features, feature_columns
from env.observation import ObservationBuilder
from env.portfolio import Portfolio
from env.reward import RewardComponents, get_reward_fn

logger = logging.getLogger(__name__)


class TradingEnv(gym.Env):
    """
    Multi-asset portfolio allocation environment.

    observation_space: Box(shape=(obs_dim,), dtype=float32)
    action_space:      Box(low=0, high=1, shape=(n_assets+1,), dtype=float32)
                       Last element is the cash weight; softmax-normalized internally.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        bars_by_symbol: Dict[str, pd.DataFrame],
        config: dict,
        macro: Optional[pd.DataFrame] = None,
        noise_sigma: float = 0.0,
        equity_jitter: float = 0.0,
    ) -> None:
        """
        Args:
            bars_by_symbol: {symbol: OHLCV DataFrame} for the episode window.
            config:         Full settings dict (environment, reward, risk sections).
            macro:          Optional macro DataFrame (vix, yield_spread, credit_proxy).
            noise_sigma:    Gaussian noise added to features during training (data augmentation).
            equity_jitter:  ±fraction for random starting capital variation.
        """
        super().__init__()
        self._bars = bars_by_symbol
        self._cfg = config.get("environment", {})
        self._reward_cfg = config.get("reward", {})
        self._risk_cfg = config.get("risk", {})
        self._macro = macro
        self._noise_sigma = noise_sigma
        self._equity_jitter = equity_jitter

        self.symbols = sorted(bars_by_symbol.keys())
        self._n_assets = len(self.symbols) + 1  # +1 for cash

        self._episode_len = self._cfg.get("episode_len", 252)
        self._initial_capital = self._cfg.get("initial_capital", 100_000.0)
        self._transaction_cost = self._cfg.get("transaction_cost", 0.001)
        self._slippage = self._cfg.get("slippage", 0.0005)
        self._fill_delay = self._cfg.get("fill_delay", 1)
        self._max_dd_terminate = self._cfg.get("max_dd_terminate", 0.25)
        self._max_turnover = self._risk_cfg.get("max_turnover_per_step", 1.0)
        reward_name = self._cfg.get("reward_fn", "risk_adjusted")
        self._reward_fn = get_reward_fn(reward_name)

        use_macro = macro is not None
        self._feature_cols = feature_columns(use_macro=use_macro)
        self._n_features = len(self._feature_cols)

        self._obs_builder = ObservationBuilder(self.symbols, self._n_features)

        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self._n_assets,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_builder.obs_dim,),
            dtype=np.float32,
        )

        # Precompute feature matrices (done once; sliced per step)
        self._features: Dict[str, pd.DataFrame] = {}
        self._common_index: Optional[pd.DatetimeIndex] = None
        self._portfolio: Optional[Portfolio] = None
        self._step_idx: int = 0
        self._start_idx: int = 0
        self._reward_history: List[float] = []

        self._precompute_features()

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # Random starting equity variation (data augmentation)
        jitter = 1.0
        if self._equity_jitter > 0:
            jitter = 1.0 + self.np_random.uniform(-self._equity_jitter, self._equity_jitter)
        start_capital = self._initial_capital * jitter

        # Random episode start within available data
        max_start = max(0, len(self._common_index) - self._episode_len - self._fill_delay)
        self._start_idx = int(self.np_random.integers(0, max_start + 1)) if max_start > 0 else 0
        self._step_idx = 0

        self._portfolio = Portfolio(
            symbols=self.symbols,
            initial_capital=start_capital,
            transaction_cost=self._transaction_cost,
            slippage=self._slippage,
        )
        self._reward_history = []
        n_eq = len(self.symbols)
        self._last_weights = np.array([0.0] * n_eq + [1.0], dtype=np.float32)  # start all-cash

        obs = self._get_obs()
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        bar_idx = self._start_idx + self._step_idx
        if bar_idx >= len(self._common_index):
            obs = self._get_obs()
            return obs, 0.0, True, False, {}

        date = self._common_index[bar_idx]
        prices = self._prices_at(bar_idx)
        prev_equity = self._portfolio.equity(prices)

        # Project action → valid portfolio weights
        target_weights = self._project_action(action)

        # Rebalance using next bar's open (fill delay)
        fill_bar_idx = bar_idx + self._fill_delay
        total_cost = 0.0
        if fill_bar_idx < len(self._common_index):
            fill_prices = self._prices_at(fill_bar_idx, use_open=True)
            total_cost = self._portfolio.rebalance(
                target_weights={sym: target_weights[i] for i, sym in enumerate(self.symbols)},
                prices=prices,
                fill_prices=fill_prices,
            )

        # Advance one bar
        next_bar_idx = bar_idx + 1
        if next_bar_idx < len(self._common_index):
            next_prices = self._prices_at(next_bar_idx)
        else:
            next_prices = prices
        curr_equity = self._portfolio.step_forward(next_prices)

        pstate = self._portfolio.state(next_prices)
        reward_components = self._reward_fn(
            prev_equity=prev_equity,
            curr_equity=curr_equity,
            drawdown_from_peak=pstate.drawdown_from_peak,
            turnover=total_cost / max(prev_equity, 1.0),
            transaction_cost=total_cost,
            weights=[target_weights[i] for i in range(len(self.symbols))],
            **self._reward_cfg,
        )
        self._reward_history.append(reward_components.total)
        self._last_weights = target_weights.copy()
        self._step_idx += 1

        terminated = bool(
            self._step_idx >= self._episode_len
            or bar_idx + 1 >= len(self._common_index)
            or abs(pstate.drawdown_from_peak) >= self._max_dd_terminate
        )

        obs = self._get_obs()
        info = {
            "date": date,
            "equity": curr_equity,
            "drawdown": pstate.drawdown_from_peak,
            "weights": pstate.weights,
            "reward_components": reward_components,
            "transaction_cost": total_cost,
        }
        return obs, float(reward_components.total), terminated, False, info

    def render(self) -> None:
        if self._portfolio is None:
            return
        bar_idx = self._start_idx + self._step_idx
        if bar_idx < len(self._common_index):
            prices = self._prices_at(bar_idx)
            pstate = self._portfolio.state(prices)
            print(
                f"Step {self._step_idx:3d} | "
                f"Equity: ${pstate.equity:,.0f} | "
                f"Drawdown: {pstate.drawdown_from_peak:.1%} | "
                f"Weights: {pstate.weights}"
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _precompute_features(self) -> None:
        """Compute feature matrices once at init; slice per step during episodes."""
        indices = []
        for sym, bars in self._bars.items():
            feat = compute_features(bars, macro=self._macro)
            self._features[sym] = feat
            indices.append(feat.index)

        if indices:
            common = indices[0]
            for idx in indices[1:]:
                common = common.intersection(idx)
            self._common_index = common.sort_values()

    def _prices_at(self, bar_idx: int, use_open: bool = False) -> Dict[str, float]:
        date = self._common_index[bar_idx]
        col = "open" if use_open else "close"
        result = {}
        for sym in self.symbols:
            bars = self._bars[sym]
            if date in bars.index:
                result[sym] = float(bars.loc[date, col])
            else:
                result[sym] = 1.0
        return result

    def _get_obs(self) -> np.ndarray:
        bar_idx = self._start_idx + self._step_idx
        bar_idx = min(bar_idx, len(self._common_index) - 1)
        date = self._common_index[bar_idx]

        feature_rows = {}
        for sym in self.symbols:
            feat_df = self._features[sym]
            if date in feat_df.index:
                row = feat_df.loc[date, self._feature_cols].values.astype(np.float32)
            else:
                row = np.zeros(self._n_features, dtype=np.float32)
            feature_rows[sym] = row

        prices = self._prices_at(bar_idx)
        pstate = self._portfolio.state(prices) if self._portfolio else _zero_state(self.symbols)

        return self._obs_builder.build(
            feature_rows=feature_rows,
            portfolio=pstate,
            episode_len=self._episode_len,
            noise_sigma=self._noise_sigma,
        )

    def _project_action(self, action: np.ndarray) -> np.ndarray:
        """Softmax-project raw logits to valid portfolio weights summing to 1."""
        clipped = np.clip(action, 0.0, 1.0).astype(np.float64)
        total = clipped.sum()
        if total < 1e-8:
            weights = np.ones(len(action)) / len(action)
        else:
            weights = clipped / total

        # Apply risk constraints (clip max single position, ensure min cash)
        max_pos = self._risk_cfg.get("max_single_position", 0.20)
        min_cash = self._risk_cfg.get("min_cash", 0.05)

        equity_weights = weights[:-1]
        equity_weights = np.clip(equity_weights, 0.0, max_pos)
        equity_sum = equity_weights.sum()
        max_equity = 1.0 - min_cash
        if equity_sum > max_equity:
            equity_weights = equity_weights * max_equity / equity_sum
        cash_weight = 1.0 - equity_weights.sum()
        weights = np.append(equity_weights, cash_weight)

        # Enforce max turnover: clip each weight's move from current holdings
        current = self._last_weights.astype(np.float64)
        delta = weights - current
        turnover = np.abs(delta).sum() / 2.0
        if turnover > self._max_turnover:
            weights = current + delta * (self._max_turnover / turnover)
            weights = np.clip(weights, 0.0, 1.0)
            weights /= weights.sum()

        return weights.astype(np.float32)


def _zero_state(symbols):
    from env.portfolio import PortfolioState
    return PortfolioState(
        equity=0.0,
        cash_fraction=1.0,
        weights={sym: 0.0 for sym in symbols},
        unrealized_pnl_pct={sym: 0.0 for sym in symbols},
        drawdown_from_peak=0.0,
        days_since_rebalance=0,
        step=0,
    )
