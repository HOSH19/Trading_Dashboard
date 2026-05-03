"""
Shared backtest simulation engine and performance metrics.

All strategies plug into run_simulation() by providing a SignalGenerator
that returns target position weights each bar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
import pandas as pd


# ── Indicators (computed once, shared across strategies) ────────────────────

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


# ── Core simulation ──────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    date: pd.Timestamp
    symbol: str
    side: str          # "buy" | "sell"
    qty: float
    price: float
    value: float


@dataclass
class BacktestResult:
    strategy: str
    equity_curve: pd.Series          # indexed by date
    trade_log: list[TradeRecord]
    initial_capital: float = 100_000.0

    # ── Metrics ────────────────────────────────────────────────────────────

    def metrics(self) -> dict:
        eq = self.equity_curve.dropna()
        if len(eq) < 2:
            return {}

        rets = eq.pct_change().dropna()
        total_return = (eq.iloc[-1] / eq.iloc[0]) - 1
        n_years = (eq.index[-1] - eq.index[0]).days / 365.25
        cagr = (1 + total_return) ** (1 / max(n_years, 1e-6)) - 1

        ann_ret = rets.mean() * 252
        ann_vol = rets.std() * np.sqrt(252)
        sharpe = ann_ret / (ann_vol + 1e-9)

        downside = rets[rets < 0].std() * np.sqrt(252)
        sortino = ann_ret / (downside + 1e-9)

        roll_max = eq.cummax()
        drawdown = (eq - roll_max) / (roll_max + 1e-9)
        max_dd = drawdown.min()
        calmar = cagr / (abs(max_dd) + 1e-9)

        win_rate = self._compute_win_rate()
        trade_stats = self._compute_trade_stats()

        sells = [t for t in self.trade_log if t.side == "sell"]

        return {
            "Total Return": f"{total_return:.1%}",
            "CAGR": f"{cagr:.1%}",
            "Sharpe": f"{sharpe:.2f}",
            "Sortino": f"{sortino:.2f}",
            "Max Drawdown": f"{max_dd:.1%}",
            "Calmar": f"{calmar:.2f}",
            "Ann. Volatility": f"{ann_vol:.1%}",
            "Total Trades": len(sells),
            "Win Rate": f"{win_rate:.1%}" if win_rate is not None else "—",
            "Profit Factor": f"{trade_stats['profit_factor']:.2f}" if trade_stats["profit_factor"] is not None else "—",
            "Avg Win %": f"{trade_stats['avg_win']:.1%}" if trade_stats["avg_win"] is not None else "—",
            "Avg Loss %": f"{trade_stats['avg_loss']:.1%}" if trade_stats["avg_loss"] is not None else "—",
            "Expectancy $": f"${trade_stats['expectancy']:,.0f}" if trade_stats["expectancy"] is not None else "—",
            "Best Trade %": f"{trade_stats['best']:.1%}" if trade_stats["best"] is not None else "—",
            "Worst Trade %": f"{trade_stats['worst']:.1%}" if trade_stats["worst"] is not None else "—",
            "Avg Hold Days": f"{trade_stats['avg_hold']:.0f}" if trade_stats["avg_hold"] is not None else "—",
            "Max Consec. Losses": trade_stats["max_consec_losses"] if trade_stats["max_consec_losses"] is not None else "—",
        }

    def _matched_trades(self):
        """Yield (entry_price, exit_price, entry_date, exit_date, value) for each closed round-trip."""
        entry_prices: dict[str, list[tuple]] = {}
        for t in sorted(self.trade_log, key=lambda x: x.date):
            if t.side == "buy":
                entry_prices.setdefault(t.symbol, []).append((t.price, t.date, t.value))
            elif t.side == "sell" and entry_prices.get(t.symbol):
                ep, ed, ev = entry_prices[t.symbol].pop(0)
                yield ep, t.price, ed, t.date, ev

    def _compute_win_rate(self) -> float | None:
        wins = losses = 0
        for ep, xp, *_ in self._matched_trades():
            if xp >= ep:
                wins += 1
            else:
                losses += 1
        total = wins + losses
        return wins / total if total > 0 else None

    def _compute_trade_stats(self) -> dict:
        win_rets, loss_rets, hold_days, win_vals, loss_vals = [], [], [], [], []
        results_seq = []  # 1=win, 0=loss for consecutive loss calc

        for ep, xp, ed, xd, ev in self._matched_trades():
            ret = xp / ep - 1
            days = (xd - ed).days
            hold_days.append(days)
            if ret >= 0:
                win_rets.append(ret)
                win_vals.append(ev * ret)
                results_seq.append(1)
            else:
                loss_rets.append(ret)
                loss_vals.append(ev * abs(ret))
                results_seq.append(0)

        if not results_seq:
            return {k: None for k in ("profit_factor", "avg_win", "avg_loss", "expectancy", "best", "worst", "avg_hold", "max_consec_losses")}

        gross_win = sum(win_vals)
        gross_loss = sum(loss_vals)
        profit_factor = gross_win / gross_loss if gross_loss > 0 else None

        n = len(results_seq)
        wr = len(win_rets) / n
        avg_win = float(np.mean(win_rets)) if win_rets else 0.0
        avg_loss = float(np.mean(loss_rets)) if loss_rets else 0.0
        expectancy = (wr * avg_win + (1 - wr) * avg_loss) * self.initial_capital if results_seq else None

        max_consec = cur = 0
        for r in results_seq:
            cur = cur + 1 if r == 0 else 0
            max_consec = max(max_consec, cur)

        all_rets = win_rets + loss_rets
        return {
            "profit_factor": profit_factor,
            "avg_win": avg_win if win_rets else None,
            "avg_loss": avg_loss if loss_rets else None,
            "expectancy": expectancy,
            "best": max(all_rets) if all_rets else None,
            "worst": min(all_rets) if all_rets else None,
            "avg_hold": float(np.mean(hold_days)) if hold_days else None,
            "max_consec_losses": max_consec if results_seq else None,
        }


def run_simulation(
    ohlcv: dict[str, pd.DataFrame],
    signal_fn: Callable[[pd.Timestamp, dict[str, pd.DataFrame], dict], dict[str, float]],
    strategy_name: str,
    initial_capital: float = 100_000.0,
    commission: float = 0.0005,
    rebalance_every: int = 5,
) -> BacktestResult:
    """
    ohlcv: {symbol: DataFrame with columns open/high/low/close/volume, DatetimeIndex}
    signal_fn: (date, ohlcv_slice, state) -> {symbol: target_weight}
               Called every `rebalance_every` bars. Weights sum ≤ 1.0.
    state: mutable dict passed through to signal_fn for per-strategy bookkeeping
    """
    all_dates = sorted(set.union(*[set(df.index) for df in ohlcv.values()]))
    all_dates = pd.DatetimeIndex(all_dates)

    cash = initial_capital
    holdings: dict[str, float] = {}   # symbol -> shares
    stop_prices: dict[str, float] = {}  # symbol -> stop price (trailing)
    equity_curve: dict[pd.Timestamp, float] = {}
    trades: list[TradeRecord] = []
    state: dict = {}

    for bar_idx, date in enumerate(all_dates):
        prices = {sym: ohlcv[sym].loc[date, "close"] for sym in ohlcv if date in ohlcv[sym].index}
        lows = {sym: ohlcv[sym].loc[date, "low"] for sym in ohlcv if date in ohlcv[sym].index}

        # ── Check stops on open (use low as proxy for intraday stop hit) ──
        to_exit = []
        for sym, shares in holdings.items():
            if sym not in prices:
                continue
            stop = stop_prices.get(sym, 0)
            if lows.get(sym, prices[sym]) <= stop:
                to_exit.append(sym)

        for sym in to_exit:
            exit_price = max(stop_prices[sym], lows.get(sym, prices[sym]))
            shares = holdings.pop(sym)
            stop_prices.pop(sym, None)
            proceeds = shares * exit_price * (1 - commission)
            cash += proceeds
            trades.append(TradeRecord(date, sym, "sell", shares, exit_price, proceeds))

        # ── Update trailing stops (only for symbols that opted in via state) ──
        for sym in list(holdings):
            trail_pct = state.get(f"trail_{sym}")
            if trail_pct is None or sym not in prices:
                continue
            new_stop = prices[sym] * (1 - trail_pct)
            stop_prices[sym] = max(stop_prices.get(sym, 0), new_stop)

        # ── Rebalance ──
        if bar_idx % rebalance_every == 0:
            slice_ohlcv = {sym: ohlcv[sym].loc[:date] for sym in ohlcv if date in ohlcv[sym].index}
            target_weights = signal_fn(date, slice_ohlcv, state)

            portfolio_value = cash + sum(
                holdings.get(sym, 0) * prices.get(sym, 0) for sym in holdings
            )

            # Exit positions no longer in target
            for sym in list(holdings):
                if sym not in target_weights or target_weights[sym] < 0.001:
                    p = prices.get(sym)
                    if p is None:
                        continue
                    shares = holdings.pop(sym)
                    stop_prices.pop(sym, None)
                    proceeds = shares * p * (1 - commission)
                    cash += proceeds
                    trades.append(TradeRecord(date, sym, "sell", shares, p, proceeds))

            # Enter / adjust positions
            for sym, weight in target_weights.items():
                if weight < 0.001 or sym not in prices:
                    continue
                target_value = portfolio_value * weight
                current_value = holdings.get(sym, 0) * prices[sym]
                delta_value = target_value - current_value

                if abs(delta_value) < portfolio_value * 0.005:
                    continue  # skip trivial rebalances

                p = prices[sym]
                if delta_value > 0 and cash >= delta_value:
                    shares = delta_value / p
                    cost = shares * p * (1 + commission)
                    cash -= cost
                    holdings[sym] = holdings.get(sym, 0) + shares
                    trail_pct = state.get(f"trail_{sym}")
                    if trail_pct is not None:
                        stop_prices.setdefault(sym, p * (1 - trail_pct))
                    trades.append(TradeRecord(date, sym, "buy", shares, p, cost))

                elif delta_value < 0 and sym in holdings:
                    shares = min(abs(delta_value) / p, holdings[sym])
                    proceeds = shares * p * (1 - commission)
                    cash += proceeds
                    holdings[sym] -= shares
                    if holdings[sym] < 1e-6:
                        del holdings[sym]
                        stop_prices.pop(sym, None)
                    trades.append(TradeRecord(date, sym, "sell", shares, p, proceeds))

        equity = cash + sum(holdings.get(sym, 0) * prices.get(sym, 0) for sym in holdings)
        equity_curve[date] = equity

    series = pd.Series(equity_curve)
    series.index = pd.DatetimeIndex(series.index)
    return BacktestResult(strategy=strategy_name, equity_curve=series, trade_log=trades)
