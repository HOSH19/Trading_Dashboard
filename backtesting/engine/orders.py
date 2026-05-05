"""Stop processing and trailing-stop helpers."""

from __future__ import annotations

import pandas as pd

from backtesting.engine.rebalance import rebalance  # noqa: F401 — re-exported for callers
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
