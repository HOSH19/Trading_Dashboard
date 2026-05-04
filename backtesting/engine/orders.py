"""Order execution helpers — stop processing, trailing stops, and rebalancing."""

from __future__ import annotations

import pandas as pd

from backtesting.trade_record import TradeRecord


def process_stops(
    date: pd.Timestamp,
    holdings: dict,
    stop_prices: dict,
    prices: dict,
    lows: dict,
    trades: list,
    cash: float,
    commission: float,
) -> float:
    """Exit any position whose low touched its stop price; return updated cash."""
    to_exit = [s for s in holdings if s in prices and lows.get(s, prices[s]) <= stop_prices.get(s, 0)]
    for sym in to_exit:
        exit_price = max(stop_prices[sym], lows.get(sym, prices[sym]))
        shares = holdings.pop(sym)
        stop_prices.pop(sym, None)
        proceeds = shares * exit_price * (1 - commission)
        cash += proceeds
        trades.append(TradeRecord(date, sym, "sell", shares, exit_price, proceeds))
    return cash


def update_trailing_stops(holdings: dict, stop_prices: dict, prices: dict, state: dict) -> None:
    """Ratchet trailing stops upward when price has moved in our favour."""
    for sym in list(holdings):
        trail_pct = state.get(f"trail_{sym}")
        if trail_pct is None or sym not in prices:
            continue
        stop_prices[sym] = max(stop_prices.get(sym, 0), prices[sym] * (1 - trail_pct))


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

    for sym, weight in target.items():
        if weight < 0.001 or sym not in prices:
            continue
        p = prices[sym]
        delta = portfolio_value * weight - holdings.get(sym, 0) * p
        if abs(delta) < portfolio_value * 0.005:
            continue
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
