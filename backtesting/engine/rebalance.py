"""Rebalance order execution — exit stale positions and enter/adjust targets."""

from __future__ import annotations

import pandas as pd

from backtesting.trade_record import TradeRecord


def _exit_stale(
    date: pd.Timestamp,
    target: dict,
    holdings: dict,
    stop_prices: dict,
    prices: dict,
    trades: list,
    cash: float,
    commission: float,
) -> float:
    for sym in list(holdings):
        if sym not in target or target[sym] < 0.001:
            p = prices.get(sym)
            if p is None:
                continue
            shares = holdings.pop(sym)
            stop_prices.pop(sym, None)
            proceeds = shares * p * (1 - commission)
            cash += proceeds
            trades.append(TradeRecord(date, sym, "sell", shares, p, proceeds))
    return cash


def _adjust_position(
    date: pd.Timestamp,
    sym: str,
    weight: float,
    holdings: dict,
    stop_prices: dict,
    prices: dict,
    trades: list,
    cash: float,
    portfolio_value: float,
    commission: float,
    state: dict,
) -> float:
    p = prices[sym]
    delta = portfolio_value * weight - holdings.get(sym, 0) * p
    if abs(delta) < portfolio_value * 0.005:
        return cash
    if delta > 0 and cash >= delta:
        shares = delta / p
        cost = shares * p * (1 + commission)
        cash -= cost
        holdings[sym] = holdings.get(sym, 0) + shares
        trail_pct = state.get(f"trail_{sym}")
        if trail_pct is not None:
            stop_prices.setdefault(sym, p * (1 - trail_pct))
        trades.append(TradeRecord(date, sym, "buy", shares, p, cost))
    elif delta < 0 and sym in holdings:
        shares = min(abs(delta) / p, holdings[sym])
        proceeds = shares * p * (1 - commission)
        cash += proceeds
        holdings[sym] -= shares
        if holdings[sym] < 1e-6:
            del holdings[sym]
            stop_prices.pop(sym, None)
        trades.append(TradeRecord(date, sym, "sell", shares, p, proceeds))
    return cash


def rebalance(
    date: pd.Timestamp,
    target: dict,
    holdings: dict,
    stop_prices: dict,
    prices: dict,
    trades: list,
    cash: float,
    portfolio_value: float,
    commission: float,
    state: dict,
) -> float:
    """Exit stale positions, then enter or adjust to reach target weights; return updated cash."""
    cash = _exit_stale(date, target, holdings, stop_prices, prices, trades, cash, commission)
    for sym, weight in target.items():
        if weight >= 0.001 and sym in prices:
            cash = _adjust_position(
                date, sym, weight, holdings, stop_prices, prices, trades,
                cash, portfolio_value, commission, state,
            )
    return cash
