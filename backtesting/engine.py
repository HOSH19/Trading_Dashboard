"""Backtest simulation loop — plugs signal functions into market data."""

from __future__ import annotations

from typing import Callable

import pandas as pd

from backtesting.metrics import BacktestResult, TradeRecord


def run_simulation(
    ohlcv: dict[str, pd.DataFrame],
    signal_fn: Callable[[pd.Timestamp, dict[str, pd.DataFrame], dict], dict[str, float]],
    strategy_name: str,
    initial_capital: float = 100_000.0,
    commission: float = 0.0005,
    rebalance_every: int = 5,
) -> BacktestResult:
    """
    ohlcv       : {symbol: DataFrame(open/high/low/close/volume, DatetimeIndex)}
    signal_fn   : (date, ohlcv_slice, state) -> {symbol: target_weight}
    rebalance_every: call signal_fn every N bars
    """
    all_dates = pd.DatetimeIndex(sorted(set.union(*[set(df.index) for df in ohlcv.values()])))

    cash = initial_capital
    holdings: dict[str, float] = {}
    stop_prices: dict[str, float] = {}
    equity_curve: dict[pd.Timestamp, float] = {}
    trades: list[TradeRecord] = []
    state: dict = {}

    for bar_idx, date in enumerate(all_dates):
        prices = {s: ohlcv[s].loc[date, "close"] for s in ohlcv if date in ohlcv[s].index}
        lows   = {s: ohlcv[s].loc[date, "low"]   for s in ohlcv if date in ohlcv[s].index}

        cash = _process_stops(date, holdings, stop_prices, prices, lows, trades, cash, commission)
        _update_trailing_stops(holdings, stop_prices, prices, state)

        if bar_idx % rebalance_every == 0:
            slice_ohlcv = {s: ohlcv[s].loc[:date] for s in ohlcv if date in ohlcv[s].index}
            target = signal_fn(date, slice_ohlcv, state)
            portfolio_value = cash + sum(holdings.get(s, 0) * prices.get(s, 0) for s in holdings)
            cash = _rebalance(date, target, holdings, stop_prices, prices, trades, cash, portfolio_value, commission, state)

        equity_curve[date] = cash + sum(holdings.get(s, 0) * prices.get(s, 0) for s in holdings)

    series = pd.Series(equity_curve, index=pd.DatetimeIndex(list(equity_curve)))
    return BacktestResult(strategy=strategy_name, equity_curve=series, trade_log=trades)


def _process_stops(date, holdings, stop_prices, prices, lows, trades, cash, commission) -> float:
    to_exit = [s for s in holdings if s in prices and lows.get(s, prices[s]) <= stop_prices.get(s, 0)]
    for sym in to_exit:
        exit_price = max(stop_prices[sym], lows.get(sym, prices[sym]))
        shares = holdings.pop(sym)
        stop_prices.pop(sym, None)
        proceeds = shares * exit_price * (1 - commission)
        cash += proceeds
        trades.append(TradeRecord(date, sym, "sell", shares, exit_price, proceeds))
    return cash


def _update_trailing_stops(holdings, stop_prices, prices, state) -> None:
    for sym in list(holdings):
        trail_pct = state.get(f"trail_{sym}")
        if trail_pct is None or sym not in prices:
            continue
        new_stop = prices[sym] * (1 - trail_pct)
        stop_prices[sym] = max(stop_prices.get(sym, 0), new_stop)


def _rebalance(date, target, holdings, stop_prices, prices, trades, cash, portfolio_value, commission, state) -> float:
    # Exit stale positions
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

    # Enter / adjust
    for sym, weight in target.items():
        if weight < 0.001 or sym not in prices:
            continue
        p = prices[sym]
        target_value  = portfolio_value * weight
        current_value = holdings.get(sym, 0) * p
        delta = target_value - current_value

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
