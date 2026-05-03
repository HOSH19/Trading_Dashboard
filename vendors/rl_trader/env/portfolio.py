"""Stateful portfolio tracker for the trading environment."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class Trade:
    """Record of a single rebalance event."""
    timestamp: pd.Timestamp
    symbol: str
    old_weight: float
    new_weight: float
    price: float
    turnover: float
    cost: float


@dataclass
class PortfolioState:
    """Full portfolio state at a single timestep."""
    equity: float
    cash_fraction: float
    weights: Dict[str, float]
    unrealized_pnl_pct: Dict[str, float]
    drawdown_from_peak: float
    days_since_rebalance: int
    step: int


class Portfolio:
    """
    Multi-asset portfolio tracker.

    Tracks cash, per-asset shares and weights, equity curve, and drawdown.
    Called by TradingEnv.step() to apply rebalancing and compute P&L.
    """

    def __init__(
        self,
        symbols: List[str],
        initial_capital: float = 100_000.0,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
    ) -> None:
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage

        self._cash = initial_capital
        self._shares: Dict[str, float] = {sym: 0.0 for sym in symbols}
        self._entry_prices: Dict[str, float] = {sym: 0.0 for sym in symbols}
        self._peak_equity = initial_capital
        self._days_since_rebalance = 0
        self._step = 0
        self._trade_log: List[Trade] = []
        self._equity_curve: List[float] = [initial_capital]

    def reset(self, initial_capital: Optional[float] = None) -> None:
        cap = initial_capital or self.initial_capital
        self._cash = cap
        self._shares = {sym: 0.0 for sym in self.symbols}
        self._entry_prices = {sym: 0.0 for sym in self.symbols}
        self._peak_equity = cap
        self._days_since_rebalance = 0
        self._step = 0
        self._trade_log = []
        self._equity_curve = [cap]

    def equity(self, prices: Dict[str, float]) -> float:
        asset_value = sum(self._shares[sym] * prices[sym] for sym in self.symbols)
        return self._cash + asset_value

    def weights(self, prices: Dict[str, float]) -> Dict[str, float]:
        eq = self.equity(prices)
        if eq <= 0:
            return {sym: 0.0 for sym in self.symbols}
        return {sym: self._shares[sym] * prices[sym] / eq for sym in self.symbols}

    def cash_fraction(self, prices: Dict[str, float]) -> float:
        eq = self.equity(prices)
        return self._cash / eq if eq > 0 else 1.0

    def unrealized_pnl_pct(self, prices: Dict[str, float]) -> Dict[str, float]:
        result = {}
        for sym in self.symbols:
            entry = self._entry_prices.get(sym, 0.0)
            if entry > 0 and self._shares[sym] > 0:
                result[sym] = (prices[sym] - entry) / entry
            else:
                result[sym] = 0.0
        return result

    def drawdown(self, prices: Dict[str, float]) -> float:
        eq = self.equity(prices)
        self._peak_equity = max(self._peak_equity, eq)
        return (eq - self._peak_equity) / self._peak_equity  # always <= 0

    def rebalance(
        self,
        target_weights: Dict[str, float],
        prices: Dict[str, float],
        fill_prices: Dict[str, float],
    ) -> float:
        """
        Execute a rebalance to target_weights using fill_prices (T+1 open).

        Prices are used for equity computation (current close); fill_prices
        are used for actual order execution (next open + slippage).

        Returns total transaction cost incurred.
        """
        eq = self.equity(prices)
        current_weights = self.weights(prices)
        total_cost = 0.0

        for sym in self.symbols:
            target_w = target_weights.get(sym, 0.0)
            current_w = current_weights.get(sym, 0.0)
            delta_w = target_w - current_w
            if abs(delta_w) < 1e-4:
                continue

            target_value = target_w * eq
            fill_price = fill_prices[sym] * (1 + np.sign(delta_w) * self.slippage)
            current_shares = self._shares[sym]
            target_shares = target_value / fill_price if fill_price > 0 else 0.0
            delta_shares = target_shares - current_shares

            turnover = abs(delta_shares * fill_price)
            cost = turnover * self.transaction_cost
            total_cost += cost

            self._shares[sym] = max(0.0, target_shares)
            self._cash -= delta_shares * fill_price + cost

            if delta_shares > 0 and self._entry_prices[sym] == 0.0:
                self._entry_prices[sym] = fill_price
            elif self._shares[sym] == 0.0:
                self._entry_prices[sym] = 0.0

            self._trade_log.append(Trade(
                timestamp=pd.Timestamp.now(),
                symbol=sym,
                old_weight=current_w,
                new_weight=target_w,
                price=fill_price,
                turnover=turnover / eq if eq > 0 else 0.0,
                cost=cost,
            ))

        self._days_since_rebalance = 0
        return total_cost

    def step_forward(self, prices: Dict[str, float]) -> float:
        """
        Advance one bar without rebalancing.

        Returns current equity.
        """
        self._step += 1
        self._days_since_rebalance += 1
        eq = self.equity(prices)
        self._equity_curve.append(eq)
        self._peak_equity = max(self._peak_equity, eq)
        return eq

    def state(self, prices: Dict[str, float]) -> PortfolioState:
        return PortfolioState(
            equity=self.equity(prices),
            cash_fraction=self.cash_fraction(prices),
            weights=self.weights(prices),
            unrealized_pnl_pct=self.unrealized_pnl_pct(prices),
            drawdown_from_peak=self.drawdown(prices),
            days_since_rebalance=self._days_since_rebalance,
            step=self._step,
        )

    @property
    def equity_curve(self) -> List[float]:
        return self._equity_curve

    @property
    def trade_log(self) -> List[Trade]:
        return self._trade_log
