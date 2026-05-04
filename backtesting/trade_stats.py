"""Round-trip trade statistics helpers."""

from __future__ import annotations

import numpy as np

from backtesting.trade_record import TradeRecord


def fmt_pct(v) -> str:
    return f"{v:.1%}" if v is not None else "—"


def fmt_f2(v) -> str:
    return f"{v:.2f}" if v is not None else "—"


def fmt_int(v):
    return v if v is not None else "—"


def matched_trades(trade_log: list[TradeRecord]):
    """Yield (entry_price, exit_price, entry_date, exit_date, entry_value) for each closed round-trip."""
    entries: dict[str, list[tuple]] = {}
    for t in sorted(trade_log, key=lambda x: x.date):
        if t.side == "buy":
            entries.setdefault(t.symbol, []).append((t.price, t.date, t.value))
        elif t.side == "sell" and entries.get(t.symbol):
            ep, ed, ev = entries[t.symbol].pop(0)
            yield ep, t.price, ed, t.date, ev


def _max_consec_losses(seq: list[int]) -> int:
    max_run = cur = 0
    for r in seq:
        cur = cur + 1 if r == 0 else 0
        max_run = max(max_run, cur)
    return max_run


def trade_stats(trade_log: list[TradeRecord], initial_capital: float) -> dict:
    """Compute win rate, profit factor, expectancy, and related stats from a trade log."""
    win_rets, loss_rets, hold_days, win_vals, loss_vals, seq = [], [], [], [], [], []

    for ep, xp, ed, xd, ev in matched_trades(trade_log):
        ret = xp / ep - 1
        hold_days.append((xd - ed).days)
        if ret >= 0:
            win_rets.append(ret); win_vals.append(ev * ret); seq.append(1)
        else:
            loss_rets.append(ret); loss_vals.append(ev * abs(ret)); seq.append(0)

    if not seq:
        return {k: None for k in ("win_rate", "profit_factor", "avg_win", "avg_loss",
                                   "expectancy", "best", "worst", "avg_hold", "max_consec_losses")}

    wr = len(win_rets) / len(seq)
    avg_win  = float(np.mean(win_rets))  if win_rets  else 0.0
    avg_loss = float(np.mean(loss_rets)) if loss_rets else 0.0
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
        "max_consec_losses": _max_consec_losses(seq),
    }
