"""Trade log display and download components."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from backtesting.backtest_result import BacktestResult


def _trade_row(strategy: str, t) -> dict:
    return {
        "Strategy": strategy,
        "Date": t.date.date() if hasattr(t.date, "date") else t.date,
        "Symbol": t.symbol, "Side": t.side.upper(),
        "Qty": round(t.qty, 4), "Price": round(t.price, 2), "Value": round(t.value, 2),
    }


def _build_trades_df(results: list[BacktestResult]) -> pd.DataFrame:
    return pd.DataFrame([_trade_row(r.strategy, t) for r in results for t in r.trade_log])


def _compute_pnl_rows(results: list[BacktestResult]) -> list[dict]:
    rows = []
    for result in results:
        entry_prices: dict[str, float] = {}
        entry_dates: dict[str, pd.Timestamp] = {}
        for t in sorted(result.trade_log, key=lambda x: x.date):
            if t.side == "buy":
                entry_prices[t.symbol] = t.price
                entry_dates[t.symbol] = t.date
            elif t.side == "sell" and t.symbol in entry_prices:
                entry = entry_prices.pop(t.symbol)
                pnl_pct = (t.price / entry - 1) * 100
                hold_days = (t.date - entry_dates.pop(t.symbol)).days
                rows.append({
                    "Strategy": result.strategy,
                    "Symbol": t.symbol,
                    "Entry": f"${entry:.2f}",
                    "Exit": f"${t.price:.2f}",
                    "P&L %": f"{pnl_pct:+.2f}%",
                    "Hold (days)": hold_days,
                    "Win": "✓" if pnl_pct >= 0 else "✗",
                })
    return rows


def download_trade_log(results: list[BacktestResult]) -> None:
    """Render a CSV download button for the full combined trade log."""
    df = _build_trades_df(results)
    if df.empty: return
    st.download_button(
        label="Download trade log (CSV)",
        data=df.to_csv(index=False).encode(),
        file_name="backtest_trade_log.csv",
        mime="text/csv",
    )


def _render_filters(df: pd.DataFrame):
    f1, f2, f3 = st.columns(3)
    with f1:
        strategies = st.multiselect(
            "Strategy", df["Strategy"].unique().tolist(),
            default=df["Strategy"].unique().tolist(), key="tl_strategy",
        )
    with f2:
        sides = st.multiselect("Side", ["BUY", "SELL"], default=["BUY", "SELL"], key="tl_side")
    with f3:
        symbols = st.multiselect("Symbol", sorted(df["Symbol"].unique()), key="tl_symbol")
    return strategies, sides, symbols


def trade_log_table(results: list[BacktestResult]) -> None:
    """Filterable trade log table with per-symbol P&L summary below."""
    df = _build_trades_df(results)
    if df.empty:
        st.info("No trades in this backtest.")
        return

    strategies, sides, symbols = _render_filters(df)
    filtered = df[df["Strategy"].isin(strategies) & df["Side"].isin(sides)]
    if symbols:
        filtered = filtered[filtered["Symbol"].isin(symbols)]
    filtered = filtered.sort_values("Date", ascending=False)

    st.caption(f"{len(filtered):,} trades shown of {len(df):,} total")
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    st.markdown("**Per-symbol P&L**")
    pnl_rows = _compute_pnl_rows(results)
    if pnl_rows:
        pnl_df = pd.DataFrame(pnl_rows)
        pnl_df["_sort"] = pnl_df["P&L %"].str.replace("%", "", regex=False).str.replace("+", "", regex=False).astype(float)
        pnl_df = pnl_df.sort_values(["Strategy", "_sort"], ascending=[True, False]).drop(columns="_sort")
        st.dataframe(pnl_df, use_container_width=True, hide_index=True)
