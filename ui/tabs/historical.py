"""Historical Performance tab — daily equity from Alpaca portfolio history API."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config import STRATEGIES
from fetchers.alpaca_live import fetch_all_history, fetch_all_snapshots
from ui.charts import drawdown_chart, equity_chart
from ui.components import metrics_table_from_curves, monthly_heatmaps_row

_PERIOD_MAP = {"1 Month": "1M", "3 Months": "3M", "6 Months": "6M", "1 Year": "1A", "All time": "all"}


@st.cache_data(ttl=300)
def _load_history(period: str) -> dict[str, tuple[list, list]]:
    histories = fetch_all_history(STRATEGIES, period=period)
    return {k: (v.index.astype(str).tolist(), v.tolist()) for k, v in histories.items()}


def _deserialise(raw: dict[str, tuple[list, list]]) -> dict[str, pd.Series]:
    return {
        name: pd.Series(vals, index=pd.DatetimeIndex(idx), name=name, dtype=float)
        for name, (idx, vals) in raw.items()
    }


def _append_live_equity(curves: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """Overwrite today's entry with the current live equity from each account."""
    snapshots = {s.strategy: s for s in fetch_all_snapshots(STRATEGIES)}
    today = pd.Timestamp.now(tz="UTC").normalize()
    result = {}
    for name, series in curves.items():
        snap = snapshots.get(name)
        if snap and not snap.error and snap.equity > 0:
            series = series.copy()
            series[today] = snap.equity
        result[name] = series
    return result


def render() -> None:
    st.subheader("Live Account Performance")
    st.caption("Daily equity pulled from each strategy's Alpaca paper account.")

    period_label = st.radio("Period", list(_PERIOD_MAP.keys()), index=4, horizontal=True, key="hist_period")
    if st.button("Refresh", key="refresh_hist"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner("Fetching account history from Alpaca…"):
        curves = _append_live_equity(_deserialise(_load_history(_PERIOD_MAP[period_label])))

    if not curves:
        st.info("No portfolio history available yet. Accounts may not have started trading.")
        return

    st.plotly_chart(equity_chart(curves, "Portfolio Value Over Time"), use_container_width=True)
    st.plotly_chart(drawdown_chart(curves), use_container_width=True)

    st.subheader("Performance Metrics")
    metrics_table_from_curves(curves)

    st.subheader("Monthly Returns")
    monthly_heatmaps_row(curves)
