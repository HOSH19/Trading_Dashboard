"""Backtest execution, trade summary helpers, and strategy explainer text."""

from __future__ import annotations

import streamlit as st

from backtesting.strategies import run_claudebot, run_rl_trader, run_regime_trader, run_etf_benchmark, _ETF_OPTIONS  # noqa: F401

STRATEGY_EXPLAINER = """
| Strategy | Implementation |
|---|---|
| **RL Trader** | A2C neural network. Observes 16 market features per symbol + portfolio state. Outputs softmax portfolio weights daily. |
| **Regime Trader** | Pre-trained HMM (Student-t emissions, SPY/QQQ/IWM/DIA features). Rebalances weekly on signal change. |
| **Claudebot** | Claude Haiku scores all 20 symbols weekly using the 5-factor rubric from TRADING-STRATEGY.md. Alpaca News API provides headlines for the catalyst factor. Responses are cached. |
"""


def run_selected(
    start: str,
    end: str,
    *,
    run_rl: bool,
    run_regime: bool,
    run_claude: bool,
    etf_benchmarks: list[str],
) -> list:
    """Execute each selected strategy in order, updating a progress bar."""
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


def trade_summary(results: list) -> None:
    """Render a row of buy/sell execution counts, one metric per strategy."""
    st.subheader("Trade Summary")
    cols = st.columns(len(results))
    for col, result in zip(cols, results):
        with col:
            buys = sum(1 for t in result.trade_log if t.side == "buy")
            sells = sum(1 for t in result.trade_log if t.side == "sell")
            st.metric(result.strategy, f"{buys} / {sells}", help="buy / sell executions")
