"""Alpaca data models — Portfolio and Position snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Position:
    symbol: str
    qty: float
    avg_entry: float
    current_price: float
    market_value: float
    unrealized_pl: float
    unrealized_pl_pct: float


@dataclass
class PortfolioSnapshot:
    strategy: str
    equity: float
    cash: float
    buying_power: float
    today_pl: float
    today_pl_pct: float
    positions: list[Position] = field(default_factory=list)
    error: str | None = None
