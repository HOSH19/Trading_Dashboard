"""TradeRecord dataclass — one executed buy or sell."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class TradeRecord:
    date: pd.Timestamp
    symbol: str
    side: str   # "buy" | "sell"
    qty: float
    price: float
    value: float
