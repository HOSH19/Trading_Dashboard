"""BacktestResult dataclass."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtesting.trade_record import TradeRecord
from backtesting.trade_stats import fmt_f2, fmt_int, fmt_pct, trade_stats


@dataclass
class BacktestResult:
    strategy: str
    equity_curve: pd.Series
    trade_log: list[TradeRecord]
    initial_capital: float = 100_000.0

    @classmethod
    def empty(cls, name: str) -> "BacktestResult":
        return cls(strategy=name, equity_curve=pd.Series(dtype=float), trade_log=[])

    @classmethod
    def trim_and_scale(cls, name: str, equity: pd.Series, trades: list, start: str) -> "BacktestResult":
        """Trim equity/trades to start date and normalise equity to $100k at first bar."""
        start_ts = pd.Timestamp(start)
        if equity.index.tz is not None:
            start_ts = start_ts.tz_localize(equity.index.tz)
        eq = equity[equity.index >= start_ts]
        if eq.empty:
            return cls.empty(name)
        scale = 100_000.0 / eq.iloc[0]
        return cls(name, eq * scale, [t for t in trades if t.date >= start_ts])

    def metrics(self) -> dict:
        """Compute and return all performance metrics as a formatted dict."""
        eq = self.equity_curve.dropna()
        if len(eq) < 2:
            return {}

        rets = eq.pct_change().dropna()
        total_return = eq.iloc[-1] / eq.iloc[0] - 1
        n_years = len(eq) / 252
        cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / max(n_years, 1e-6)) - 1

        daily_rf = 0.045 / 252  # 4.5% annualised, matching regime-trader's performance.py
        excess = rets - daily_rf
        ann_vol = rets.std() * np.sqrt(252)
        sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0
        downside_std = excess[excess < 0].std()
        sortino = float(excess.mean() / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0

        roll_max = eq.cummax()
        max_dd = ((eq - roll_max) / (roll_max + 1e-9)).min()
        ts = trade_stats(self.trade_log, self.initial_capital)
        sells = sum(1 for t in self.trade_log if t.side == "sell")

        return {
            "Total Return":       f"{total_return:.1%}",
            "CAGR":               f"{cagr:.1%}",
            "Sharpe":             f"{sharpe:.2f}",
            "Sortino":            f"{sortino:.2f}",
            "Max Drawdown":       f"{max_dd:.1%}",
            "Calmar":             f"{cagr / (abs(max_dd) + 1e-9):.2f}",
            "Ann. Volatility":    f"{ann_vol:.1%}",
            "Total Trades":       sells,
            "Win Rate":           fmt_pct(ts["win_rate"]),
            "Profit Factor":      fmt_f2(ts["profit_factor"]),
            "Avg Win %":          fmt_pct(ts["avg_win"]),
            "Avg Loss %":         fmt_pct(ts["avg_loss"]),
            "Expectancy $":       f"${ts['expectancy']:,.0f}" if ts["expectancy"] is not None else "—",
            "Best Trade %":       fmt_pct(ts["best"]),
            "Worst Trade %":      fmt_pct(ts["worst"]),
            "Avg Hold Days":      f"{ts['avg_hold']:.0f}" if ts["avg_hold"] is not None else "—",
            "Max Consec. Losses": fmt_int(ts["max_consec_losses"]),
        }
