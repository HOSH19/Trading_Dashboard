"""Shared chart styling: colors, layout defaults, and trace-style helpers."""

from __future__ import annotations

from config import STRATEGIES

STRATEGY_COLORS = {s.name: s.color for s in STRATEGIES}

CHART_LAYOUT = dict(
    margin=dict(l=0, r=0, t=30, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(gridcolor="#2a2a3a", title=None),
    yaxis=dict(gridcolor="#2a2a3a"),
)

HEATMAP_MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

ETF_COLORS: dict = {
    "SPY": "#aaaaaa",
    "QQQ": "#e377c2",
    "IWM": "#17becf",
    "DIA": "#bcbd22",
    "GLD": "#ffd700",
    "TLT": "#9467bd",
    "BTC-USD": "#f5a623",
    "_default": ["#cccccc", "#888888", "#bbbbbb", "#999999"],
}


def etf_color(name: str, idx: int) -> str:
    ticker = name.replace(" B&H", "")
    if ticker in ETF_COLORS:
        return ETF_COLORS[ticker]
    return ETF_COLORS["_default"][idx % len(ETF_COLORS["_default"])]


def trace_style(name: str, etf_color_idx: int, *, equity: bool) -> tuple[dict, str | None, int]:
    """Return (line_dict, fill, updated_etf_color_idx) for a named equity or drawdown curve."""
    is_etf = name not in STRATEGY_COLORS
    if is_etf:
        color = etf_color(name, etf_color_idx)
        line = dict(color=color, width=1.5 if equity else 1, dash="dash")
        return line, None, etf_color_idx + 1
    line = dict(color=STRATEGY_COLORS[name], width=2 if equity else 1.5)
    fill = None if equity else "tozeroy"
    return line, fill, etf_color_idx
