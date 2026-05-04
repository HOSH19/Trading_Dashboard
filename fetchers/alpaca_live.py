"""Fetch live portfolio snapshots and equity history from Alpaca paper accounts."""

from __future__ import annotations

import pandas as pd

from fetchers.alpaca_models import Position, PortfolioSnapshot


def _fetch_snapshot(name: str, api_key: str, secret_key: str) -> PortfolioSnapshot:
    """Return a PortfolioSnapshot for a named strategy; error field set on failure."""
    if not api_key or not secret_key:
        return PortfolioSnapshot(strategy=name, equity=0, cash=0, buying_power=0,
                                 today_pl=0, today_pl_pct=0, error="API keys not configured")
    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(api_key, secret_key, paper=True)
        account = client.get_account()
        equity = float(account.equity)
        last_equity = float(account.last_equity)
        today_pl = equity - last_equity
        positions = [
            Position(
                symbol=p.symbol, qty=float(p.qty),
                avg_entry=float(p.avg_entry_price), current_price=float(p.current_price),
                market_value=float(p.market_value), unrealized_pl=float(p.unrealized_pl),
                unrealized_pl_pct=float(p.unrealized_plpc) * 100,
            )
            for p in client.get_all_positions()
        ]
        return PortfolioSnapshot(
            strategy=name, equity=equity, cash=float(account.cash),
            buying_power=float(account.buying_power),
            today_pl=today_pl, today_pl_pct=(today_pl / last_equity * 100) if last_equity else 0,
            positions=positions,
        )
    except Exception as exc:
        return PortfolioSnapshot(strategy=name, equity=0, cash=0, buying_power=0,
                                 today_pl=0, today_pl_pct=0, error=str(exc))


def fetch_portfolio_history(name: str, api_key: str, secret_key: str, period: str = "all") -> pd.Series:
    """Return daily equity Series; zeros (market-closed / bot-not-running days) are dropped.

    period: '1M' | '3M' | '6M' | '1A' | 'all'
    """
    if not api_key or not secret_key:
        return pd.Series(dtype=float, name=name)
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetPortfolioHistoryRequest
        client = TradingClient(api_key, secret_key, paper=True)
        h = client.get_portfolio_history(GetPortfolioHistoryRequest(period=period, timeframe="1D"))
        ts = pd.to_datetime(h.timestamp, unit="s", utc=True)
        series = pd.Series([e or 0.0 for e in h.equity], index=ts, name=name, dtype=float)
        return series[series > 0]
    except Exception:
        return pd.Series(dtype=float, name=name)


def fetch_all_history(strategies: list, period: str = "all") -> dict[str, pd.Series]:
    from config import alpaca_keys
    result = {}
    for s in strategies:
        key, secret = alpaca_keys(s.env_prefix)
        series = fetch_portfolio_history(s.name, key, secret, period)
        if not series.empty:
            result[s.name] = series
    return result


def fetch_all_snapshots(strategies: list) -> list[PortfolioSnapshot]:
    from config import alpaca_keys
    return [_fetch_snapshot(s.name, *alpaca_keys(s.env_prefix)) for s in strategies]


def positions_to_df(positions: list[Position]) -> pd.DataFrame:
    """Convert a list of Position objects to a display DataFrame."""
    if not positions:
        return pd.DataFrame(columns=["Symbol", "Qty", "Avg Entry", "Price", "Mkt Value", "Unreal P&L", "P&L %"])
    return pd.DataFrame([
        {
            "Symbol": p.symbol, "Qty": p.qty,
            "Avg Entry": f"${p.avg_entry:,.2f}", "Price": f"${p.current_price:,.2f}",
            "Mkt Value": f"${p.market_value:,.2f}", "Unreal P&L": f"${p.unrealized_pl:+,.2f}",
            "P&L %": f"{p.unrealized_pl_pct:+.2f}%",
        }
        for p in sorted(positions, key=lambda x: x.market_value, reverse=True)
    ])
