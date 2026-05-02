"""
Algo Trading Strategy Comparison Dashboard

Tabs:
  1. Live Portfolio  — current equity, positions, today's P&L per account
  2. Historical      — equity curves + metrics from state.db / TRADE-LOG.md
  3. Backtest        — run all 3 strategy proxies on historical data, no waiting required
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from config import STRATEGIES
from fetchers.alpaca_live import fetch_all_snapshots, positions_to_df

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Algo Strategy Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  .metric-card { background: #1e1e2e; border-radius: 8px; padding: 16px; }
  .stTabs [data-baseweb="tab-list"] { gap: 12px; }
  .stTabs [data-baseweb="tab"] { padding: 8px 20px; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _color(val: float) -> str:
    return "#26a69a" if val >= 0 else "#ef5350"


def _fmt_delta(val: float, pct: float) -> str:
    sign = "+" if val >= 0 else ""
    return f"{sign}${val:,.2f} ({sign}{pct:.2f}%)"


def _equity_chart(curves: dict[str, pd.Series], title: str = "") -> go.Figure:
    fig = go.Figure()
    colors = {s.name: s.color for s in STRATEGIES}
    for name, series in curves.items():
        if series.empty:
            continue
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values,
            name=name, line=dict(color=colors.get(name, "#aaa"), width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>" + name + "</extra>",
        ))
    fig.update_layout(
        title=title, height=380,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title=None, yaxis_title="Portfolio Value ($)",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(gridcolor="#2a2a3a"),
        xaxis=dict(gridcolor="#2a2a3a"),
    )
    return fig


def _drawdown_chart(curves: dict[str, pd.Series]) -> go.Figure:
    fig = go.Figure()
    colors = {s.name: s.color for s in STRATEGIES}
    for name, series in curves.items():
        if series.empty:
            continue
        roll_max = series.cummax()
        dd = (series - roll_max) / (roll_max + 1e-9) * 100
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values,
            name=name, fill="tozeroy",
            line=dict(color=colors.get(name, "#aaa"), width=1.5),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%<extra>" + name + "</extra>",
        ))
    fig.update_layout(
        title="Drawdown (%)", height=220,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title=None, yaxis_title="Drawdown (%)",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(gridcolor="#2a2a3a"),
        xaxis=dict(gridcolor="#2a2a3a"),
    )
    return fig


# ── Header ───────────────────────────────────────────────────────────────────

st.title("Algo Strategy Comparison")
st.caption("RL Trader · Regime Trader · Claudebot — separate Alpaca paper accounts")

tab_live, tab_hist, tab_bt = st.tabs(["📡 Live Portfolio", "📊 Historical Performance", "🔬 Backtest Comparison"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE PORTFOLIO
# ════════════════════════════════════════════════════════════════════════════

with tab_live:
    if st.button("Refresh live data", key="refresh_live"):
        st.cache_data.clear()

    @st.cache_data(ttl=60)
    def _load_live():
        return fetch_all_snapshots(STRATEGIES)

    with st.spinner("Fetching live portfolio data…"):
        snapshots = _load_live()

    # ── Summary metrics row ──
    cols = st.columns(3)
    for col, snap, strat in zip(cols, snapshots, STRATEGIES):
        with col:
            if snap.error:
                st.error(f"**{strat.name}**\n\n{snap.error}")
                continue
            total_pl = snap.equity - 100_000
            total_pl_pct = (snap.equity / 100_000 - 1) * 100
            today_sign = "+" if snap.today_pl >= 0 else ""
            st.metric(
                label=f"**{strat.name}**",
                value=f"${snap.equity:,.2f}",
                delta=f"{total_pl_pct:+.2f}% vs $100k start",
                delta_color="normal",
            )
            # Use HTML to avoid Streamlit treating $ as LaTeX delimiters
            st.markdown(
                f"<small style='color:#888'>"
                f"Today: {today_sign}{snap.today_pl:,.2f} ({today_sign}{snap.today_pl_pct:.2f}%)"
                f"</small><br>"
                f"<small style='color:#888'>"
                f"Cash: {snap.cash:,.0f} &nbsp;·&nbsp; Invested: {snap.equity - snap.cash:,.0f}"
                f"</small>",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Positions tables ──
    col_a, col_b, col_c = st.columns(3)
    for col, snap, strat in zip([col_a, col_b, col_c], snapshots, STRATEGIES):
        with col:
            # Cash donut — above the title
            if not snap.error and snap.equity > 0:
                invested = snap.equity - snap.cash
                fig = go.Figure(go.Pie(
                    values=[max(invested, 0), max(snap.cash, 0)],
                    labels=["Invested", "Cash"],
                    hole=0.65,
                    marker_colors=[strat.color, "#2a2a3a"],
                    textinfo="percent",
                    textposition="inside",
                    insidetextorientation="horizontal",
                    textfont=dict(size=13),
                ))
                fig.update_layout(
                    height=200, margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)

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


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — HISTORICAL PERFORMANCE (from Alpaca portfolio history API)
# ════════════════════════════════════════════════════════════════════════════

with tab_hist:
    from fetchers.alpaca_live import fetch_all_history
    from backtesting.engine import BacktestResult

    st.subheader("Live Account Performance")
    st.caption("Daily equity pulled from each strategy's Alpaca paper account.")

    period_map = {"1 Month": "1M", "3 Months": "3M", "6 Months": "6M", "1 Year": "1A", "All time": "all"}
    selected_period_label = st.radio(
        "Period", list(period_map.keys()), index=4, horizontal=True, key="hist_period"
    )
    selected_period = period_map[selected_period_label]

    if st.button("Refresh", key="refresh_hist"):
        st.cache_data.clear()

    @st.cache_data(ttl=300)
    def _load_alpaca_history(period: str) -> dict[str, list]:
        histories = fetch_all_history(STRATEGIES, period=period)
        # Serialise to plain lists for cache compatibility
        return {k: (v.index.astype(str).tolist(), v.tolist()) for k, v in histories.items()}

    with st.spinner("Fetching account history from Alpaca…"):
        raw_history = _load_alpaca_history(selected_period)

    # Reconstruct Series from cached lists
    curves: dict[str, pd.Series] = {}
    for name, (idx, vals) in raw_history.items():
        s = pd.Series(vals, index=pd.DatetimeIndex(idx), name=name, dtype=float)
        curves[name] = s

    if not curves:
        st.info("No portfolio history available yet. Accounts may not have started trading.")
    else:
        # ── Equity curves ──
        st.plotly_chart(_equity_chart(curves, "Portfolio Value Over Time"), use_container_width=True)
        st.plotly_chart(_drawdown_chart(curves), use_container_width=True)

        # ── Performance metrics table (same style as Backtest tab) ──
        st.subheader("Performance Metrics")
        MIN_BARS_FOR_METRICS = 10  # annualised stats are meaningless on < 2 weeks of data
        metric_rows = []
        skipped = []
        for name, series in curves.items():
            if len(series) < MIN_BARS_FOR_METRICS:
                skipped.append(f"{name} ({len(series)} days)")
                continue
            result = BacktestResult(strategy=name, equity_curve=series, trade_log=[])
            m = result.metrics()
            if m:
                m["Strategy"] = name
                metric_rows.append(m)

        if skipped:
            st.caption(f"Metrics hidden for strategies with < {MIN_BARS_FOR_METRICS} trading days: {', '.join(skipped)}")

        if metric_rows:
            mdf = pd.DataFrame(metric_rows).set_index("Strategy")

            def _highlight_best(s: pd.Series) -> list[str]:
                try:
                    nums = s.str.rstrip("%").str.replace("+", "", regex=False).astype(float)
                    best_idx = nums.idxmax()
                    return ["background-color: #1a3a2a" if i == best_idx else "" for i in s.index]
                except Exception:
                    return [""] * len(s)

            styled = mdf.style.apply(_highlight_best, axis=0)
            st.dataframe(styled, use_container_width=True)
            st.caption("Green highlight = best value per metric. Based on daily closes from Alpaca.")
        elif not skipped:
            st.info("Not enough data yet to compute metrics.")

        # ── Monthly returns heatmaps ──
        st.subheader("Monthly Returns")
        heat_cols = st.columns(min(len(curves), 3))
        colors_map = {s.name: s.color for s in STRATEGIES}
        for col, (name, series) in zip(heat_cols * 2, curves.items()):
            monthly = series.resample("ME").last().pct_change().dropna() * 100
            if monthly.empty:
                continue
            pivot = monthly.to_frame("ret")
            pivot["year"] = pivot.index.year
            pivot["month"] = pivot.index.strftime("%b")
            heatmap_df = pivot.pivot(index="year", columns="month", values="ret")
            month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            heatmap_df = heatmap_df.reindex(columns=[m for m in month_order if m in heatmap_df.columns])
            fig = go.Figure(go.Heatmap(
                z=heatmap_df.values,
                x=heatmap_df.columns.tolist(),
                y=[str(y) for y in heatmap_df.index.tolist()],
                colorscale=[[0, "#c62828"], [0.5, "#1a1a2e"], [1, "#1b5e20"]],
                zmid=0,
                text=[[f"{v:.1f}%" if not pd.isna(v) else "" for v in row] for row in heatmap_df.values],
                texttemplate="%{text}",
                showscale=False,
            ))
            fig.update_layout(
                title=name, height=180,
                margin=dict(l=0, r=0, t=30, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            with col:
                st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — BACKTEST COMPARISON
# ════════════════════════════════════════════════════════════════════════════

with tab_bt:
    st.subheader("Historical Strategy Backtest")
    st.markdown(
        "Run all 3 strategy proxies against the **same** historical period to compare "
        "relative performance **without waiting months** for live results. "
        "Each strategy uses its documented logic approximated from OHLCV data."
    )

    with st.expander("How each proxy works", expanded=False):
        st.markdown("""
| Strategy | Proxy Logic |
|---|---|
| **RL Trader** | Ranks universe by 60-day rolling Sharpe ratio. Picks top 10 with RSI 35–72. Equal-weight at 8% cap. Monthly rebalance. |
| **Regime Trader** | SPY 20-day realized volatility → Low/Mid/High regime. Allocates 90%/70%/50% to momentum or defensive names. ATR trailing stops. Weekly rebalance. |
| **Claudebot** | Scores each stock 0–10 on: 5-day momentum vs peers, YTD sector rank, distance from 20-SMA, volume vs avg, ATR-based R:R. Enters if ≥7. Max 3 new/week, max 10 positions, 10% trailing stop, -7% hard stop. |
        """)

    # ── Controls ──
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        start_date = st.date_input("Start date", value=pd.Timestamp("2022-01-01").date())
    with c2:
        end_date = st.date_input("End date", value=pd.Timestamp("2024-12-31").date())
    with c3:
        include_spy = st.checkbox("Include SPY benchmark", value=True)

    run_bt = st.button("Run Backtest", type="primary", use_container_width=True)

    if run_bt:
        if start_date >= end_date:
            st.error("End date must be after start date.")
        else:
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            from backtesting.strategies import (
                run_claudebot,
                run_rl_trader,
                run_regime_trader,
                run_spy_benchmark,
            )

            progress = st.progress(0, text="Downloading market data…")

            with st.spinner("Running backtests — this takes 30–90 seconds…"):
                try:
                    progress.progress(10, "Running RL Trader…")
                    rl_result = run_rl_trader(start_str, end_str)

                    progress.progress(40, "Running Regime Trader…")
                    regime_result = run_regime_trader(start_str, end_str)

                    progress.progress(70, "Running Claudebot…")
                    claude_result = run_claudebot(start_str, end_str)

                    results = [rl_result, regime_result, claude_result]

                    if include_spy:
                        progress.progress(90, "Running SPY benchmark…")
                        spy_result = run_spy_benchmark(start_str, end_str)
                        results.append(spy_result)

                    progress.progress(100, "Done!")

                    # ── Store in session state ──
                    st.session_state["bt_results"] = results

                except Exception as e:
                    st.error(f"Backtest failed: {e}")
                    st.stop()

    # ── Display results ──
    if "bt_results" in st.session_state:
        results = st.session_state["bt_results"]
        colors_map = {s.name: s.color for s in STRATEGIES}
        colors_map["SPY B&H"] = "#888"

        # Equity curves
        bt_curves = {r.strategy: r.equity_curve for r in results if not r.equity_curve.empty}
        if bt_curves:
            st.plotly_chart(_equity_chart(bt_curves, "Backtest Equity Curves"), use_container_width=True)
            st.plotly_chart(_drawdown_chart(bt_curves), use_container_width=True)

        # Metrics table
        st.subheader("Performance Metrics")
        metric_rows = []
        for result in results:
            m = result.metrics()
            if m:
                m["Strategy"] = result.strategy
                metric_rows.append(m)

        if metric_rows:
            mdf = pd.DataFrame(metric_rows).set_index("Strategy")

            # Highlight best value per numeric column
            def _highlight_best(s: pd.Series) -> list[str]:
                try:
                    nums = s.str.rstrip("%").str.replace("+", "", regex=False).astype(float)
                    best_idx = nums.idxmax()
                    return ["background-color: #1a3a2a" if i == best_idx else "" for i in s.index]
                except Exception:
                    return [""] * len(s)

            styled = mdf.style.apply(_highlight_best, axis=0)
            st.dataframe(styled, use_container_width=True)

            st.caption("Green highlight = best value per metric column.")

        # ── Trade summary ──
        st.subheader("Trade Summary")
        t_cols = st.columns(len(results))
        for col, result in zip(t_cols, results):
            with col:
                st.markdown(f"**{result.strategy}**")
                buys = sum(1 for t in result.trade_log if t.side == "buy")
                sells = sum(1 for t in result.trade_log if t.side == "sell")
                st.metric("Trades (buy/sell)", f"{buys} / {sells}")

        # ── Monthly returns heatmap ──
        st.subheader("Monthly Returns")
        heat_cols = st.columns(min(len(results), 3))
        for col, result in zip(heat_cols * 2, results):
            if result.equity_curve.empty:
                continue
            monthly = result.equity_curve.resample("ME").last().pct_change().dropna() * 100
            if monthly.empty:
                continue
            pivot = monthly.to_frame("ret")
            pivot["year"] = pivot.index.year
            pivot["month"] = pivot.index.strftime("%b")
            heatmap_df = pivot.pivot(index="year", columns="month", values="ret")

            month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            heatmap_df = heatmap_df.reindex(columns=[m for m in month_order if m in heatmap_df.columns])

            fig = go.Figure(go.Heatmap(
                z=heatmap_df.values,
                x=heatmap_df.columns.tolist(),
                y=[str(y) for y in heatmap_df.index.tolist()],
                colorscale=[[0, "#c62828"], [0.5, "#1a1a2e"], [1, "#1b5e20"]],
                zmid=0,
                text=[[f"{v:.1f}%" if not pd.isna(v) else "" for v in row] for row in heatmap_df.values],
                texttemplate="%{text}",
                showscale=False,
            ))
            fig.update_layout(
                title=result.strategy, height=180,
                margin=dict(l=0, r=0, t=30, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            with col:
                st.plotly_chart(fig, use_container_width=True)
