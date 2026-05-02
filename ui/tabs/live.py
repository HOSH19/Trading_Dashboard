"""Live Portfolio tab — real-time equity, positions, and P&L from each Alpaca account."""

from __future__ import annotations

import streamlit as st

from config import STRATEGIES
from fetchers.alpaca_live import fetch_all_snapshots, positions_to_df
from ui.charts import allocation_donut, equity_chart, drawdown_chart


@st.cache_data(ttl=60)
def _load_snapshots():
    return fetch_all_snapshots(STRATEGIES)


def _metric_row(snapshots) -> None:
    cols = st.columns(3)
    for col, snap, strat in zip(cols, snapshots, STRATEGIES):
        with col:
            if snap.error:
                st.error(f"**{strat.name}**\n\n{snap.error}")
                continue
            total_pl_pct = (snap.equity / 100_000 - 1) * 100
            today_sign = "+" if snap.today_pl >= 0 else ""
            st.metric(
                label=f"**{strat.name}**",
                value=f"${snap.equity:,.2f}",
                delta=f"{total_pl_pct:+.2f}% vs $100k start",
                delta_color="normal",
            )
            st.markdown(
                f"<small style='color:#888'>"
                f"Today: {today_sign}{snap.today_pl:,.2f} ({today_sign}{snap.today_pl_pct:.2f}%)"
                f"</small><br>"
                f"<small style='color:#888'>"
                f"Cash: {snap.cash:,.0f} &nbsp;·&nbsp; Invested: {snap.equity - snap.cash:,.0f}"
                f"</small>",
                unsafe_allow_html=True,
            )


def _positions_row(snapshots) -> None:
    cols = st.columns(3)
    for col, snap, strat in zip(cols, snapshots, STRATEGIES):
        with col:
            if not snap.error and snap.equity > 0:
                st.plotly_chart(
                    allocation_donut(snap.equity - snap.cash, snap.cash, strat.color),
                    use_container_width=True,
                )
            st.subheader(strat.name)
            st.caption(strat.description)
            if snap.error:
                st.warning("No data")
                continue
            df = positions_to_df(snap.positions)
            if df.empty:
                st.info("No open positions")
            else:
                st.dataframe(df, use_container_width=True, hide_index=True)


def render() -> None:
    if st.button("Refresh live data", key="refresh_live"):
        st.cache_data.clear()

    with st.spinner("Fetching live portfolio data…"):
        snapshots = _load_snapshots()

    _metric_row(snapshots)
    st.divider()
    _positions_row(snapshots)
