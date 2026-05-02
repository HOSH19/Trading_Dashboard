"""Fetch historical OHLCV data from Yahoo Finance for backtesting."""

from __future__ import annotations

import pandas as pd


def load_ohlcv(start: str, end: str, symbols: list[str] | None = None) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    from config import UNIVERSE

    syms = symbols or UNIVERSE
    raw = yf.download(syms, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        return {}

    is_multi = isinstance(raw.columns, pd.MultiIndex)
    result = {}
    for sym in syms:
        try:
            df = raw.xs(sym, axis=1, level=1).copy() if is_multi else raw.copy()
            df.columns = [c.lower() for c in df.columns]
            df = df.dropna(subset=["close"])
            if not df.empty:
                result[sym] = df
        except KeyError:
            continue
    return result
