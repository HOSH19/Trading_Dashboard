"""Strategy entry points — re-exports real agent implementations + SPY benchmark."""

from __future__ import annotations

import pandas as pd

from backtesting.engine import run_simulation
from backtesting.metrics import BacktestResult
from backtesting.strategies_real_claudebot import run_claudebot_api as run_claudebot
from backtesting.strategies_real_regime import run_regime_trader
from backtesting.strategies_real_rl import run_rl_trader
from fetchers.market_data import load_ohlcv

__all__ = ["run_regime_trader", "run_rl_trader", "run_claudebot", "run_spy_benchmark", "run_etf_benchmark"]

_ETF_OPTIONS = ["SPY", "QQQ", "IWM", "DIA", "GLD", "TLT", "BTC-USD"]


def run_spy_benchmark(start: str, end: str) -> BacktestResult:
    return run_etf_benchmark("SPY", start, end, label="SPY B&H")


def run_etf_benchmark(ticker: str, start: str, end: str, label: str | None = None) -> BacktestResult:
    """Buy-and-hold a single ETF — buys once at open, never rebalances."""
    name = label or f"{ticker} B&H"
    ohlcv = load_ohlcv(start, end, [ticker])
    if not ohlcv or ticker not in ohlcv:
        return BacktestResult(name, pd.Series(dtype=float), [])
    df = ohlcv[ticker]
    # Compute equity directly — no trade engine needed, no spurious trade records
    scale = 100_000.0 / float(df["close"].iloc[0])
    equity = df["close"] * scale
    return BacktestResult(name, equity, [])
