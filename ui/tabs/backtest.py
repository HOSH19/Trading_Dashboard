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
    sc1, sc2 = st.columns(2)
    with sc1: run_rl = st.checkbox("RL Trader", value=True)
    with sc2: run_regime = st.checkbox("Regime Trader", value=True)
    etf_benchmarks = st.multiselect("ETF benchmarks", _ETF_OPTIONS, default=["SPY"],
        help="Buy-and-hold equity curves added to the chart for comparison")
    return start_date, end_date, run_rl, run_regime, etf_benchmarks


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


def _run_backtest_button(start_date, end_date, run_rl, run_regime, etf_benchmarks) -> None:
    """Handle Run Backtest button: validate inputs, run, and store results in session state."""
    if not st.button("Run Backtest", type="primary", use_container_width=True):
        return
    if start_date >= end_date:
        st.error("End date must be after start date.")
        return
    if not any([run_rl, run_regime, etf_benchmarks]):
        st.error("Select at least one strategy or ETF benchmark to run.")
        return
    with st.spinner("Running backtests…"):
        try:
            st.session_state["bt_results"] = run_selected(
                start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"),
                run_rl=run_rl, run_regime=run_regime,
                etf_benchmarks=etf_benchmarks,
            )
        except Exception as exc:
            st.error(f"Backtest failed: {exc}")
            st.stop()


def render() -> None:
    st.subheader("Historical Strategy Backtest")
    st.markdown("Runs inference on 3 live agents against the **same** historical period using their "
                "**Github Actionsproduction code**. Claudebot is excluded due to API costs.")
    with st.expander("How each agent works", expanded=False):
        st.markdown(_STRATEGY_EXPLAINER)
    rl_min, rl_max = rl_date_range()
    controls = _render_controls(rl_min, rl_max)
    _run_backtest_button(*controls)
    if "bt_results" in st.session_state:
        _render_results(st.session_state["bt_results"])
