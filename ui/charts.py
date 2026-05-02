"""Reusable Plotly chart builders. No Streamlit calls — return figures only."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from config import STRATEGIES

_STRATEGY_COLORS = {s.name: s.color for s in STRATEGIES}
_CHART_LAYOUT = dict(
    margin=dict(l=0, r=0, t=30, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(gridcolor="#2a2a3a", title=None),
    yaxis=dict(gridcolor="#2a2a3a"),
)

_HEATMAP_MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def equity_chart(curves: dict[str, pd.Series], title: str = "") -> go.Figure:
    fig = go.Figure()
    for name, series in curves.items():
        if series.empty:
            continue
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values,
            name=name,
            line=dict(color=_STRATEGY_COLORS.get(name, "#aaa"), width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>" + name + "</extra>",
        ))
    fig.update_layout(title=title, height=380, yaxis_title="Portfolio Value ($)", **_CHART_LAYOUT)
    return fig


def drawdown_chart(curves: dict[str, pd.Series]) -> go.Figure:
    fig = go.Figure()
    for name, series in curves.items():
        if series.empty:
            continue
        roll_max = series.cummax()
        dd = (series - roll_max) / (roll_max + 1e-9) * 100
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values,
            name=name, fill="tozeroy",
            line=dict(color=_STRATEGY_COLORS.get(name, "#aaa"), width=1.5),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%<extra>" + name + "</extra>",
        ))
    fig.update_layout(title="Drawdown (%)", height=220, yaxis_title="Drawdown (%)", **_CHART_LAYOUT)
    return fig


def monthly_heatmap(series: pd.Series, title: str) -> go.Figure | None:
    monthly = series.resample("ME").last().pct_change().dropna() * 100
    if monthly.empty:
        return None

    pivot = monthly.to_frame("ret")
    pivot["year"] = pivot.index.year
    pivot["month"] = pivot.index.strftime("%b")
    heatmap_df = pivot.pivot(index="year", columns="month", values="ret")
    heatmap_df = heatmap_df.reindex(columns=[m for m in _HEATMAP_MONTH_ORDER if m in heatmap_df.columns])

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
        title=title, height=180,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def allocation_donut(invested: float, cash: float, color: str) -> go.Figure:
    fig = go.Figure(go.Pie(
        values=[max(invested, 0), max(cash, 0)],
        labels=["Invested", "Cash"],
        hole=0.65,
        marker_colors=[color, "#2a2a3a"],
        textinfo="percent",
        textposition="inside",
        insidetextorientation="horizontal",
        textfont=dict(size=13),
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig
