"""Reusable Streamlit UI components and data-display helpers."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from backtesting.engine import BacktestResult

MIN_BARS_FOR_METRICS = 10  # annualised stats are meaningless on < 2 weeks of data


def highlight_best(s: pd.Series) -> list[str]:
    """Pandas Styler column — green background on the best (max) value."""
    try:
        nums = s.str.rstrip("%").str.replace("+", "", regex=False).astype(float)
        best_idx = nums.idxmax()
        return ["background-color: #1a3a2a" if i == best_idx else "" for i in s.index]
    except Exception:
        return [""] * len(s)


_ALL_METRICS = [
    "Total Return", "CAGR", "Sharpe", "Sortino", "Max Drawdown", "Calmar", "Ann. Volatility",
    "Total Trades", "Win Rate", "Profit Factor", "Avg Win %", "Avg Loss %",
    "Expectancy $", "Best Trade %", "Worst Trade %", "Avg Hold Days", "Max Consec. Losses",
]
_DEFAULT_METRICS = ["Total Return", "CAGR", "Sharpe", "Max Drawdown", "Total Trades", "Win Rate"]


def metrics_table(
    results: list[BacktestResult],
    min_bars: int = MIN_BARS_FOR_METRICS,
    key: str = "metrics_cols",
) -> None:
    """Render a styled performance metrics table from a list of BacktestResults."""
    selected_cols = st.multiselect(
        "Metrics to display",
        _ALL_METRICS,
        default=_DEFAULT_METRICS,
        key=key,
    )

    metric_rows, skipped = [], []
    for result in results:
        if len(result.equity_curve) < min_bars:
            skipped.append(f"{result.strategy} ({len(result.equity_curve)} days)")
            continue
        m = result.metrics()
        if m:
            m["Strategy"] = result.strategy
            metric_rows.append(m)

    if skipped:
        st.caption(f"Metrics hidden for strategies with < {min_bars} trading days: {', '.join(skipped)}")

    if metric_rows:
        mdf = pd.DataFrame(metric_rows).set_index("Strategy")
        cols = [c for c in selected_cols if c in mdf.columns]
        if cols:
            st.dataframe(mdf[cols].style.apply(highlight_best, axis=0), use_container_width=True)
            st.caption("Green highlight = best value per metric column.")
        else:
            st.info("Select at least one metric above.")
    elif not skipped:
        st.info("Not enough data yet to compute metrics.")


def metrics_table_from_curves(
    curves: dict[str, pd.Series],
    min_bars: int = MIN_BARS_FOR_METRICS,
) -> None:
    """Wrapper for callers that only have equity curves (no trade log)."""
    results = [BacktestResult(strategy=name, equity_curve=s, trade_log=[]) for name, s in curves.items()]
    metrics_table(results, min_bars=min_bars, key="metrics_cols_hist")


def monthly_heatmaps_row(curves: dict[str, pd.Series]) -> None:
    """Render a row of monthly-return heatmaps, one per strategy."""
    from ui.charts import monthly_heatmap

    cols = st.columns(min(len(curves), 3))
    for col, (name, series) in zip(cols * 2, curves.items()):
        fig = monthly_heatmap(series, title=name)
        if fig:
            with col:
                st.plotly_chart(fig, use_container_width=True)


def download_trade_log(results: list[BacktestResult]) -> None:
    """Render a CSV download button for the full trade log."""
    all_trades = []
    for result in results:
        for t in result.trade_log:
            all_trades.append({
                "Strategy": result.strategy,
                "Date": t.date.date() if hasattr(t.date, "date") else t.date,
                "Symbol": t.symbol,
                "Side": t.side.upper(),
                "Qty": round(t.qty, 4),
                "Price": round(t.price, 2),
                "Value": round(t.value, 2),
            })
    if not all_trades:
        return
    df = pd.DataFrame(all_trades)
    st.download_button(
        label="Download trade log (CSV)",
        data=df.to_csv(index=False).encode(),
        file_name="backtest_trade_log.csv",
        mime="text/csv",
    )


def trade_log_table(results: list[BacktestResult]) -> None:
    """Filterable trade log showing every simulated entry and exit."""
    all_trades = []
    for result in results:
        for t in result.trade_log:
            all_trades.append({
                "Strategy": result.strategy,
                "Date": t.date.date() if hasattr(t.date, "date") else t.date,
                "Symbol": t.symbol,
                "Side": t.side.upper(),
                "Qty": round(t.qty, 4),
                "Price": round(t.price, 2),
                "Value": round(t.value, 2),
            })

    if not all_trades:
        st.info("No trades in this backtest.")
        return

    df = pd.DataFrame(all_trades)

    # ── Filters ──
    f1, f2, f3 = st.columns(3)
    with f1:
        strategies = st.multiselect(
            "Strategy", df["Strategy"].unique().tolist(),
            default=df["Strategy"].unique().tolist(), key="tl_strategy",
        )
    with f2:
        sides = st.multiselect("Side", ["BUY", "SELL"], default=["BUY", "SELL"], key="tl_side")
    with f3:
        symbols = st.multiselect(
            "Symbol", sorted(df["Symbol"].unique()), key="tl_symbol",
        )

    filtered = df[df["Strategy"].isin(strategies) & df["Side"].isin(sides)]
    if symbols:
        filtered = filtered[filtered["Symbol"].isin(symbols)]

    filtered = filtered.sort_values("Date", ascending=False)

    st.caption(f"{len(filtered):,} trades shown of {len(df):,} total")
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    # ── Per-symbol P&L summary ──
    st.markdown("**Per-symbol P&L**")
    pnl_rows = []
    for result in results:
        entry_prices: dict[str, float] = {}
        entry_dates: dict[str, pd.Timestamp] = {}
        for t in sorted(result.trade_log, key=lambda x: x.date):
            if t.side == "buy":
                entry_prices[t.symbol] = t.price
                entry_dates[t.symbol] = t.date
            elif t.side == "sell" and t.symbol in entry_prices:
                entry = entry_prices.pop(t.symbol)
                pnl_pct = (t.price / entry - 1) * 100
                hold_days = (t.date - entry_dates.pop(t.symbol)).days
                pnl_rows.append({
                    "Strategy": result.strategy,
                    "Symbol": t.symbol,
                    "Entry": f"${entry:.2f}",
                    "Exit": f"${t.price:.2f}",
                    "P&L %": f"{pnl_pct:+.2f}%",
                    "Hold (days)": hold_days,
                    "Win": "✓" if pnl_pct >= 0 else "✗",
                })

    if pnl_rows:
        pnl_df = pd.DataFrame(pnl_rows)
        # Sort by strategy then P&L descending
        pnl_df["_sort"] = pnl_df["P&L %"].str.replace("%", "", regex=False).str.replace("+", "", regex=False).astype(float)
        pnl_df = pnl_df.sort_values(["Strategy", "_sort"], ascending=[True, False]).drop(columns="_sort")
        st.dataframe(pnl_df, use_container_width=True, hide_index=True)
