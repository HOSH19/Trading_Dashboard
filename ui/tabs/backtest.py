"""Backtest Comparison tab — runs all 3 real agent implementations on historical OHLCV data."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from backtesting.strategies import _ETF_OPTIONS
from backtesting.agents.rl.rl_trader import rl_date_range
from ui.charts import drawdown_chart, equity_chart
from ui.components import metrics_table, monthly_heatmaps_row
from ui.tabs.backtest_runner import STRATEGY_EXPLAINER as _STRATEGY_EXPLAINER, run_selected, trade_summary
from ui.trade_log import download_trade_log, trade_log_table


def _render_controls(rl_min, rl_max):
    """Render date pickers and strategy checkboxes; return all selected inputs."""
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Start date",
            value=max(pd.Timestamp("2022-01-01").date(), rl_min.date()),
            min_value=rl_min.date(), max_value=rl_max.date())
    with c2:
        end_date = st.date_input("End date",
            value=min(pd.Timestamp("2024-12-31").date(), rl_max.date()),
            min_value=rl_min.date(), max_value=rl_max.date())

    st.markdown("**Strategies to run**")
    sc1, sc2, sc3 = st.columns(3)
    with sc1: run_rl = st.checkbox("RL Trader", value=True)
    with sc2: run_regime = st.checkbox("Regime Trader", value=True)
    with sc3: run_claude = st.checkbox("Claudebot (API $$)", value=False)

    etf_benchmarks = st.multiselect("ETF benchmarks", _ETF_OPTIONS, default=["SPY"],
        help="Buy-and-hold equity curves added to the chart for comparison")

    if run_claude:
        st.info("Claudebot makes real Claude API calls (~$3–8 per year of data). "
                "Results are cached after the first run — reruns are free.", icon="💰")

    return start_date, end_date, run_rl, run_regime, run_claude, etf_benchmarks


def _render_results(results: list) -> None:
    """Render charts, metrics table, monthly heatmaps, and trade log for completed results."""
    all_curves = {r.strategy: r.equity_curve for r in results if not r.equity_curve.empty}
    visible = st.multiselect("Strategies to display on charts",
        list(all_curves.keys()), default=list(all_curves.keys()), key="bt_visible_strategies")
    curves = {k: v for k, v in all_curves.items() if k in visible}

    if curves:
        st.plotly_chart(equity_chart(curves, "Backtest Equity Curves"), use_container_width=True)
        st.plotly_chart(drawdown_chart(curves), use_container_width=True)

    st.subheader("Performance Metrics")
    metrics_table(results, min_bars=2)
    trade_summary(results)

    st.subheader("Monthly Returns")
    monthly_heatmaps_row(curves)

    st.divider()
    st.subheader("Trade Log")
    st.caption("Inspect every simulated entry and exit.")
    download_trade_log(results)
    with st.expander("Show trade log", expanded=False):
        trade_log_table(results)


def render() -> None:
    st.subheader("Historical Strategy Backtest")
    st.markdown("Runs all 3 live agents against the **same** historical period using their "
                "**real production code** — same inference path as the GitHub Actions workflows.")
    with st.expander("How each agent works", expanded=False):
        st.markdown(_STRATEGY_EXPLAINER)

    rl_min, rl_max = rl_date_range()
    start_date, end_date, run_rl, run_regime, run_claude, etf_benchmarks = _render_controls(rl_min, rl_max)

    if st.button("Run Backtest", type="primary", use_container_width=True):
        if start_date >= end_date:
            st.error("End date must be after start date.")
        elif not any([run_rl, run_regime, run_claude, etf_benchmarks]):
            st.error("Select at least one strategy or ETF benchmark to run.")
        else:
            spinner_msg = ("Running backtests — Claudebot may take several minutes on first run…"
                           if run_claude else "Running backtests…")
            with st.spinner(spinner_msg):
                try:
                    results = run_selected(
                        start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"),
                        run_rl=run_rl, run_regime=run_regime, run_claude=run_claude,
                        etf_benchmarks=etf_benchmarks,
                    )
                    st.session_state["bt_results"] = results
                except Exception as exc:
                    st.error(f"Backtest failed: {exc}")
                    st.stop()

    if "bt_results" in st.session_state:
        _render_results(st.session_state["bt_results"])
