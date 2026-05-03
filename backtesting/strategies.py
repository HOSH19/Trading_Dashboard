"""Strategy entry points — re-exports real agent implementations + SPY benchmark."""

from __future__ import annotations

import pandas as pd

from backtesting.engine import run_simulation
from backtesting.metrics import BacktestResult
from backtesting.strategies_real_claudebot import run_claudebot_api as run_claudebot
from backtesting.strategies_real_regime import run_regime_trader
from backtesting.strategies_real_rl import run_rl_trader
from fetchers.market_data import load_ohlcv

__all__ = ["run_regime_trader", "run_rl_trader", "run_claudebot", "run_spy_benchmark"]


def run_spy_benchmark(start: str, end: str) -> BacktestResult:
    ohlcv = load_ohlcv(start, end, ["SPY"])
    return run_simulation(ohlcv, lambda d, o, s: {"SPY": 0.99}, "SPY B&H", rebalance_every=999) if ohlcv else BacktestResult("SPY B&H", pd.Series(dtype=float), [])
