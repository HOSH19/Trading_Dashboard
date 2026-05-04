"""Backtest simulation loop — plugs signal functions into market data."""

from __future__ import annotations

from typing import Callable

import pandas as pd

from backtesting.backtest_result import BacktestResult
from backtesting.engine.orders import process_stops, rebalance, update_trailing_stops
from backtesting.trade_record import TradeRecord


def run_simulation(
    ohlcv: dict[str, pd.DataFrame],
    signal_fn: Callable[[pd.Timestamp, dict[str, pd.DataFrame], dict], dict[str, float]],
    strategy_name: str,
    initial_capital: float = 100_000.0,
    commission: float = 0.0005,
    rebalance_every: int = 5,
) -> BacktestResult:
    """Run a daily bar simulation.

    ohlcv          : {symbol: DataFrame(open/high/low/close/volume, DatetimeIndex)}
    signal_fn      : (date, ohlcv_slice, state) -> {symbol: target_weight}
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

        cash = process_stops(date, holdings, stop_prices, prices, lows, trades, cash, commission)
        update_trailing_stops(holdings, stop_prices, prices, state)

        if bar_idx % rebalance_every == 0:
            slice_ohlcv = {s: ohlcv[s].loc[:date] for s in ohlcv if date in ohlcv[s].index}
            portfolio_value = cash + sum(holdings.get(s, 0) * prices.get(s, 0) for s in holdings)
            state.update({"__equity__": portfolio_value, "__holdings__": dict(holdings), "__cash__": cash})
            target = signal_fn(date, slice_ohlcv, state)
            cash = rebalance(date, target, holdings, stop_prices, prices, trades, cash, portfolio_value, commission, state)

        equity_curve[date] = cash + sum(holdings.get(s, 0) * prices.get(s, 0) for s in holdings)

    series = pd.Series(equity_curve, index=pd.DatetimeIndex(list(equity_curve)))
    return BacktestResult(strategy=strategy_name, equity_curve=series, trade_log=trades)
