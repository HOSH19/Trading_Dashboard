"""Reusable Streamlit UI components — metrics tables and monthly heatmaps."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from backtesting.backtest_result import BacktestResult
from ui.trade_log import download_trade_log, trade_log_table  # noqa: F401 — re-exported

MIN_BARS_FOR_METRICS = 10

_MIN_IS_BEST = {"Max Consec. Losses"}
_NO_HIGHLIGHT = {"Avg Hold Days"}

_ALL_METRICS = [
    "Total Return", "CAGR", "Sharpe", "Sortino", "Max Drawdown", "Calmar", "Ann. Volatility",
    "Total Trades", "Win Rate", "Profit Factor", "Avg Win %", "Avg Loss %",
    "Expectancy $", "Best Trade %", "Worst Trade %", "Avg Hold Days", "Max Consec. Losses",
]
_DEFAULT_METRICS = ["Total Return", "CAGR", "Sharpe", "Max Drawdown", "Total Trades", "Win Rate"]


def highlight_best(s: pd.Series) -> list[str]:
    """Pandas Styler column function — green background on the best value per column."""
    if s.name in _NO_HIGHLIGHT:
        return [""] * len(s)
    try:
        cleaned = (
            s.astype(str)
            .str.replace("—", "nan", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.replace("+", "", regex=False)
            .astype(float)
        )
        if cleaned.isna().all():
            return [""] * len(s)
        best_idx = cleaned.idxmin() if s.name in _MIN_IS_BEST else cleaned.idxmax()
        return ["background-color: #1a3a2a" if i == best_idx else "" for i in s.index]
    except Exception:
        return [""] * len(s)


def metrics_table(
    results: list[BacktestResult],
    min_bars: int = MIN_BARS_FOR_METRICS,
    key: str = "metrics_cols",
) -> None:
    """Render a styled performance metrics table from a list of BacktestResults."""
    selected_cols = st.multiselect("Metrics to display", _ALL_METRICS, default=_DEFAULT_METRICS, key=key)

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


def metrics_table_from_curves(curves: dict[str, pd.Series], min_bars: int = MIN_BARS_FOR_METRICS) -> None:
    """Convenience wrapper for callers that only have equity curves (no trade log)."""
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
