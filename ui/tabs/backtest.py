"""Backtest Comparison tab — runs all 3 real agent implementations on historical OHLCV data."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from backtesting.strategies import run_claudebot, run_rl_trader, run_regime_trader, run_spy_benchmark, run_etf_benchmark, _ETF_OPTIONS
from backtesting.strategies_real_rl import rl_date_range
from ui.charts import drawdown_chart, equity_chart
from ui.components import download_trade_log, metrics_table, monthly_heatmaps_row, trade_log_table

_STRATEGY_EXPLAINER = """
| Strategy | Implementation |
|---|---|
| **RL Trader** | A2C neural network. Observes 16 market features per symbol + portfolio state. Outputs softmax portfolio weights daily. |
| **Regime Trader** | Pre-trained HMM (Student-t emissions, SPY/QQQ/IWM/DIA features). Rebalances weekly on signal change. |
| **Claudebot** | Claude Haiku scores all 20 symbols weekly using the 5-factor rubric from TRADING-STRATEGY.md. Alpaca News API provides headlines for the catalyst factor. Responses are cached. |
"""


def _run_selected(start: str, end: str, *, run_rl: bool, run_regime: bool, run_claude: bool, etf_benchmarks: list[str]):
    steps = sum([run_rl, run_regime, run_claude, bool(etf_benchmarks)])
    done = 0
    results = []
    progress = st.progress(0, text="Starting…")

    if run_rl:
        progress.progress(int(done / steps * 90), "Running RL Trader…")
        results.append(run_rl_trader(start, end))
        done += 1

    if run_regime:
        progress.progress(int(done / steps * 90), "Running Regime Trader…")
        results.append(run_regime_trader(start, end))
        done += 1

    if run_claude:
        progress.progress(int(done / steps * 90), "Running Claudebot (API)…")
        results.append(run_claudebot(start, end))
        done += 1

    for ticker in etf_benchmarks:
        progress.progress(int(done / steps * 90), f"Loading {ticker}…")
        results.append(run_etf_benchmark(ticker, start, end))

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
        "Runs all 3 live agents against the **same** historical period using their **real production code** — "
        "same inference path as the GitHub Actions workflows."
    )
    with st.expander("How each agent works", expanded=False):
        st.markdown(_STRATEGY_EXPLAINER)

    rl_min, rl_max = rl_date_range()

    c1, c2 = st.columns([2, 2])
    with c1:
        start_date = st.date_input(
            "Start date",
            value=max(pd.Timestamp("2022-01-01").date(), rl_min.date()),
            min_value=rl_min.date(),
            max_value=rl_max.date(),
        )
    with c2:
        end_date = st.date_input(
            "End date",
            value=min(pd.Timestamp("2024-12-31").date(), rl_max.date()),
            min_value=rl_min.date(),
            max_value=rl_max.date(),
        )

    st.markdown("**Strategies to run**")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        run_rl = st.checkbox("RL Trader", value=True)
    with sc2:
        run_regime = st.checkbox("Regime Trader", value=True)
    with sc3:
        run_claude = st.checkbox("Claudebot (API $$)", value=False)

    etf_benchmarks = st.multiselect(
        "ETF benchmarks",
        _ETF_OPTIONS,
        default=["SPY"],
        help="Buy-and-hold equity curves added to the chart for comparison",
    )

    if run_claude:
        st.info(
            "Claudebot makes real Claude API calls (~$3–8 per year of data). "
            "Results are cached after the first run — reruns are free.",
            icon="💰",
        )

    if st.button("Run Backtest", type="primary", use_container_width=True):
        if start_date >= end_date:
            st.error("End date must be after start date.")
        elif not any([run_rl, run_regime, run_claude, etf_benchmarks]):
            st.error("Select at least one strategy or ETF benchmark to run.")
        else:
            spinner_msg = "Running backtests…" if not run_claude else "Running backtests — Claudebot may take several minutes on first run…"
            with st.spinner(spinner_msg):
                try:
                    results = _run_selected(
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d"),
                        run_rl=run_rl,
                        run_regime=run_regime,
                        run_claude=run_claude,
                        etf_benchmarks=etf_benchmarks,
                    )
                    st.session_state["bt_results"] = results
                except Exception as exc:
                    st.error(f"Backtest failed: {exc}")
                    st.stop()

    if "bt_results" not in st.session_state:
        return

    results = st.session_state["bt_results"]
    all_curves = {r.strategy: r.equity_curve for r in results if not r.equity_curve.empty}

    if all_curves:
        visible = st.multiselect(
            "Strategies to display on charts",
            list(all_curves.keys()),
            default=list(all_curves.keys()),
            key="bt_visible_strategies",
        )
        curves = {k: v for k, v in all_curves.items() if k in visible}
    else:
        curves = {}

    if curves:
        st.plotly_chart(equity_chart(curves, "Backtest Equity Curves"), use_container_width=True)
        st.plotly_chart(drawdown_chart(curves), use_container_width=True)

    st.subheader("Performance Metrics")
    metrics_table(results, min_bars=2)

    _trade_summary(results)

    st.subheader("Monthly Returns")
    monthly_heatmaps_row(curves)

    st.divider()
    st.subheader("Trade Log")
    st.caption("Inspect every simulated entry and exit.")
    download_trade_log(results)
    with st.expander("Show trade log", expanded=False):
        trade_log_table(results)
