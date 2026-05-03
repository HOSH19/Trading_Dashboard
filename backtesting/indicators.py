"""Technical indicators used by backtest strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - 100 / (1 + rs)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(window).mean()


def sma(prices: pd.Series, window: int) -> pd.Series:
    return prices.rolling(window).mean()


def rolling_sharpe(returns: pd.Series, window: int = 60) -> pd.Series:
    mu = returns.rolling(window).mean()
    sigma = returns.rolling(window).std()
    return (mu / (sigma + 1e-9)) * np.sqrt(252)
