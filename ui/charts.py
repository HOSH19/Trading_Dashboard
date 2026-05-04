"""Plotly chart builders — no Streamlit calls, return figures only."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from ui.chart_config import CHART_LAYOUT, HEATMAP_MONTH_ORDER, trace_style


def equity_chart(curves: dict[str, pd.Series], title: str = "") -> go.Figure:
    """Line chart of portfolio value over time, one trace per strategy."""
    fig = go.Figure()
    etf_idx = 0
    for name, series in curves.items():
        if series.empty:
            continue
        line, _, etf_idx = trace_style(name, etf_idx, equity=True)
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values, name=name, line=line,
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>" + name + "</extra>",
        ))
    fig.update_layout(title=title, height=380, yaxis_title="Portfolio Value ($)", **CHART_LAYOUT)
    return fig


def drawdown_chart(curves: dict[str, pd.Series]) -> go.Figure:
    """Area chart of drawdown (%) from peak, one trace per strategy."""
    fig = go.Figure()
    etf_idx = 0
    for name, series in curves.items():
        if series.empty:
            continue
        roll_max = series.cummax()
        dd = (series - roll_max) / (roll_max + 1e-9) * 100
        line, fill, etf_idx = trace_style(name, etf_idx, equity=False)
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values, name=name, fill=fill, line=line,
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%<extra>" + name + "</extra>",
        ))
    fig.update_layout(title="Drawdown (%)", height=220, yaxis_title="Drawdown (%)", **CHART_LAYOUT)
    return fig


def monthly_heatmap(series: pd.Series, title: str) -> go.Figure | None:
    """Heatmap of monthly returns (%) by year × month. Returns None if no data."""
    monthly = series.resample("ME").last().pct_change().dropna() * 100
    if monthly.empty:
        return None
    pivot = monthly.to_frame("ret")
    pivot["year"] = pivot.index.year
    pivot["month"] = pivot.index.strftime("%b")
    heatmap_df = pivot.pivot(index="year", columns="month", values="ret")
    heatmap_df = heatmap_df.reindex(columns=[m for m in HEATMAP_MONTH_ORDER if m in heatmap_df.columns])
    fig = go.Figure(go.Heatmap(
        z=heatmap_df.values,
        x=heatmap_df.columns.tolist(),
        y=[str(y) for y in heatmap_df.index.tolist()],
        colorscale=[[0, "#c62828"], [0.5, "#1a1a2e"], [1, "#1b5e20"]],
        zmid=0,
        text=[[f"{v:.1f}%" if not pd.isna(v) else "" for v in row] for row in heatmap_df.values],
        texttemplate="%{text}", showscale=False,
    ))
    fig.update_layout(
        title=title, height=180, margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def allocation_donut(invested: float, cash: float, color: str) -> go.Figure:
    """Donut chart showing invested vs cash allocation."""
    fig = go.Figure(go.Pie(
        values=[max(invested, 0), max(cash, 0)],
        labels=["Invested", "Cash"],
        hole=0.65, marker_colors=[color, "#2a2a3a"],
        textinfo="percent", textposition="inside",
        insidetextorientation="horizontal", textfont=dict(size=13),
    ))
    fig.update_layout(
        height=200, margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig
