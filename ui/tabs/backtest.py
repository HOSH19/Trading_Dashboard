"""Backtest Comparison tab — run all 3 strategy proxies on historical OHLCV data."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from backtesting.strategies import run_claudebot, run_rl_trader, run_regime_trader, run_spy_benchmark
from ui.charts import drawdown_chart, equity_chart
from ui.components import metrics_table, monthly_heatmaps_row

_PROXY_EXPLAINER = """
| Strategy | Proxy Logic |
|---|---|
| **RL Trader** | Ranks universe by 60-day rolling Sharpe. Picks top 10 with RSI 35–72. Equal-weight at 8% cap. Monthly rebalance. |
| **Regime Trader** | SPY 20-day realized vol → Low/Mid/High regime → 90%/70%/50% allocation to momentum or defensive names. ATR trailing stops. Weekly rebalance. |
| **Claudebot** | Scores each stock 0–10 on: 5-day momentum vs peers, YTD sector rank, distance from 20-SMA, volume vs avg, ATR-based R:R. Enters if ≥7. Max 3 new/week, max 10 positions, 10% trailing stop. |
"""


def _run_all(start: str, end: str, include_spy: bool):
    progress = st.progress(0, text="Downloading market data…")
    progress.progress(10, "Running RL Trader…")
    results = [run_rl_trader(start, end)]

    progress.progress(40, "Running Regime Trader…")
    results.append(run_regime_trader(start, end))

    progress.progress(70, "Running Claudebot…")
    results.append(run_claudebot(start, end))

    if include_spy:
        progress.progress(90, "Running SPY benchmark…")
        results.append(run_spy_benchmark(start, end))

    progress.progress(100, "Done!")
    return results


def _trade_summary(results) -> None:
    st.subheader("Trade Summary")
    cols = st.columns(len(results))
    for col, result in zip(cols, results):
        with col:
            buys = sum(1 for t in result.trade_log if t.side == "buy")
            sells = sum(1 for t in result.trade_log if t.side == "sell")
            st.metric(result.strategy, f"{buys} / {sells}", help="buy / sell executions")


def render() -> None:
    st.subheader("Historical Strategy Backtest")
    st.markdown(
        "Run all 3 strategy proxies against the **same** historical period to compare "
        "relative performance **without waiting months** for live results."
    )
    with st.expander("How each proxy works", expanded=False):
        st.markdown(_PROXY_EXPLAINER)

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        start_date = st.date_input("Start date", value=pd.Timestamp("2022-01-01").date())
    with c2:
        end_date = st.date_input("End date", value=pd.Timestamp("2024-12-31").date())
    with c3:
        include_spy = st.checkbox("Include SPY benchmark", value=True)

    if st.button("Run Backtest", type="primary", use_container_width=True):
        if start_date >= end_date:
            st.error("End date must be after start date.")
        else:
            with st.spinner("Running backtests — this takes 30–90 seconds…"):
                try:
                    results = _run_all(
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d"),
                        include_spy,
                    )
                    st.session_state["bt_results"] = results
                except Exception as exc:
                    st.error(f"Backtest failed: {exc}")
                    st.stop()

    if "bt_results" not in st.session_state:
        return

    results = st.session_state["bt_results"]
    curves = {r.strategy: r.equity_curve for r in results if not r.equity_curve.empty}

    if curves:
        st.plotly_chart(equity_chart(curves, "Backtest Equity Curves"), use_container_width=True)
        st.plotly_chart(drawdown_chart(curves), use_container_width=True)

    st.subheader("Performance Metrics")
    metrics_table(curves, min_bars=2)

    _trade_summary(results)

    st.subheader("Monthly Returns")
    monthly_heatmaps_row(curves)
