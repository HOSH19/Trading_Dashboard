"""Reusable Streamlit UI components and data-display helpers."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from backtesting.engine import BacktestResult

MIN_BARS_FOR_METRICS = 10  # annualised stats are meaningless on < 2 weeks of data


def highlight_best(s: pd.Series) -> list[str]:
    """Pandas Styler column function — green background on the best (max) value."""
    try:
        nums = s.str.rstrip("%").str.replace("+", "", regex=False).astype(float)
        best_idx = nums.idxmax()
        return ["background-color: #1a3a2a" if i == best_idx else "" for i in s.index]
    except Exception:
        return [""] * len(s)


def metrics_table(curves: dict[str, pd.Series], min_bars: int = MIN_BARS_FOR_METRICS) -> None:
    """Render a styled performance metrics table for a set of equity curves."""
    metric_rows, skipped = [], []
    for name, series in curves.items():
        if len(series) < min_bars:
            skipped.append(f"{name} ({len(series)} days)")
            continue
        m = BacktestResult(strategy=name, equity_curve=series, trade_log=[]).metrics()
        if m:
            m["Strategy"] = name
            metric_rows.append(m)

    if skipped:
        st.caption(f"Metrics hidden for strategies with < {min_bars} trading days: {', '.join(skipped)}")

    if metric_rows:
        mdf = pd.DataFrame(metric_rows).set_index("Strategy")
        st.dataframe(mdf.style.apply(highlight_best, axis=0), use_container_width=True)
        st.caption("Green highlight = best value per metric column.")
    elif not skipped:
        st.info("Not enough data yet to compute metrics.")


def monthly_heatmaps_row(curves: dict[str, pd.Series]) -> None:
    """Render a row of monthly-return heatmaps, one per strategy."""
    from ui.charts import monthly_heatmap

    cols = st.columns(min(len(curves), 3))
    for col, (name, series) in zip(cols * 2, curves.items()):
        fig = monthly_heatmap(series, title=name)
        if fig:
            with col:
                st.plotly_chart(fig, use_container_width=True)
