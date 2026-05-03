"""BacktestResult dataclass and all performance metric calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TradeRecord:
    date: pd.Timestamp
    symbol: str
    side: str   # "buy" | "sell"
    qty: float
    price: float
    value: float


@dataclass
class BacktestResult:
    strategy: str
    equity_curve: pd.Series
    trade_log: list[TradeRecord]
    initial_capital: float = 100_000.0

    def metrics(self) -> dict:
        eq = self.equity_curve.dropna()
        if len(eq) < 2:
            return {}

        rets = eq.pct_change().dropna()
        total_return = eq.iloc[-1] / eq.iloc[0] - 1
        n_years = (eq.index[-1] - eq.index[0]).days / 365.25
        cagr = (1 + total_return) ** (1 / max(n_years, 1e-6)) - 1

        ann_ret = rets.mean() * 252
        ann_vol = rets.std() * np.sqrt(252)
        sharpe = ann_ret / (ann_vol + 1e-9)
        sortino = ann_ret / (rets[rets < 0].std() * np.sqrt(252) + 1e-9)

        roll_max = eq.cummax()
        max_dd = ((eq - roll_max) / (roll_max + 1e-9)).min()
        calmar = cagr / (abs(max_dd) + 1e-9)

        ts = _trade_stats(self.trade_log, self.initial_capital)
        sells = sum(1 for t in self.trade_log if t.side == "sell")

        def _fmt_pct(v): return f"{v:.1%}" if v is not None else "—"
        def _fmt_f2(v):  return f"{v:.2f}" if v is not None else "—"
        def _fmt_int(v): return v if v is not None else "—"

        return {
            "Total Return":       f"{total_return:.1%}",
            "CAGR":               f"{cagr:.1%}",
            "Sharpe":             f"{sharpe:.2f}",
            "Sortino":            f"{sortino:.2f}",
            "Max Drawdown":       f"{max_dd:.1%}",
            "Calmar":             f"{calmar:.2f}",
            "Ann. Volatility":    f"{ann_vol:.1%}",
            "Total Trades":       sells,
            "Win Rate":           _fmt_pct(ts["win_rate"]),
            "Profit Factor":      _fmt_f2(ts["profit_factor"]),
            "Avg Win %":          _fmt_pct(ts["avg_win"]),
            "Avg Loss %":         _fmt_pct(ts["avg_loss"]),
            "Expectancy $":       f"${ts['expectancy']:,.0f}" if ts["expectancy"] is not None else "—",
            "Best Trade %":       _fmt_pct(ts["best"]),
            "Worst Trade %":      _fmt_pct(ts["worst"]),
            "Avg Hold Days":      f"{ts['avg_hold']:.0f}" if ts["avg_hold"] is not None else "—",
            "Max Consec. Losses": _fmt_int(ts["max_consec_losses"]),
        }


def _matched_trades(trade_log: list[TradeRecord]):
    """Yield (entry_price, exit_price, entry_date, exit_date, entry_value) per closed round-trip."""
    entries: dict[str, list[tuple]] = {}
    for t in sorted(trade_log, key=lambda x: x.date):
        if t.side == "buy":
            entries.setdefault(t.symbol, []).append((t.price, t.date, t.value))
        elif t.side == "sell" and entries.get(t.symbol):
            ep, ed, ev = entries[t.symbol].pop(0)
            yield ep, t.price, ed, t.date, ev


def _trade_stats(trade_log: list[TradeRecord], initial_capital: float) -> dict:
    win_rets, loss_rets, hold_days, win_vals, loss_vals, seq = [], [], [], [], [], []

    for ep, xp, ed, xd, ev in _matched_trades(trade_log):
        ret = xp / ep - 1
        hold_days.append((xd - ed).days)
        if ret >= 0:
            win_rets.append(ret); win_vals.append(ev * ret); seq.append(1)
        else:
            loss_rets.append(ret); loss_vals.append(ev * abs(ret)); seq.append(0)

    if not seq:
        return {k: None for k in ("win_rate", "profit_factor", "avg_win", "avg_loss",
                                   "expectancy", "best", "worst", "avg_hold", "max_consec_losses")}

    n = len(seq)
    wr = len(win_rets) / n
    avg_win  = float(np.mean(win_rets))  if win_rets  else 0.0
    avg_loss = float(np.mean(loss_rets)) if loss_rets else 0.0

    max_consec = cur = 0
    for r in seq:
        cur = cur + 1 if r == 0 else 0
        max_consec = max(max_consec, cur)

    gross_loss = sum(loss_vals)
    all_rets = win_rets + loss_rets
    return {
        "win_rate":          wr,
        "profit_factor":     sum(win_vals) / gross_loss if gross_loss > 0 else None,
        "avg_win":           avg_win  if win_rets  else None,
        "avg_loss":          avg_loss if loss_rets else None,
        "expectancy":        (wr * avg_win + (1 - wr) * avg_loss) * initial_capital,
        "best":              max(all_rets),
        "worst":             min(all_rets),
        "avg_hold":          float(np.mean(hold_days)),
        "max_consec_losses": max_consec,
    }
