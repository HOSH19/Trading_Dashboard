"""Strategy entry points — re-exports real agent implementations + ETF benchmark."""

from __future__ import annotations

import pandas as pd

from backtesting.agents.claudebot.claudebot import run_claudebot_api as run_claudebot
from backtesting.agents.regime.regime_trader import run_regime_trader
from backtesting.agents.rl.rl_trader import run_rl_trader
from backtesting.backtest_result import BacktestResult
from backtesting.engine import run_simulation
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
        return BacktestResult.empty(name)
    df = ohlcv[ticker]
    scale = 100_000.0 / float(df["close"].iloc[0])
    equity = df["close"] * scale
    return BacktestResult(name, equity, [])
