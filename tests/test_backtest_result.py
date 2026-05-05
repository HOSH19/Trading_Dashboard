"""Tests for BacktestResult metrics, trim_and_scale, and empty constructors."""

from __future__ import annotations

import pandas as pd
import pytest

from backtesting.backtest_result import BacktestResult, _max_drawdown, _return_stats, _risk_stats
from backtesting.trade_record import TradeRecord


def _eq(values: list[float], start="2024-01-01") -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq="B")
    return pd.Series(values, index=idx)


class TestReturnStats:
    def test_flat_equity(self):
        eq = _eq([100.0] * 252)
        total, cagr = _return_stats(eq)
        assert total == pytest.approx(0.0)
        assert cagr == pytest.approx(0.0, abs=1e-4)

    def test_doubling(self):
        eq = _eq([100.0, 200.0])
        total, cagr = _return_stats(eq)
        assert total == pytest.approx(1.0)


class TestMaxDrawdown:
    def test_no_drawdown(self):
        eq = _eq([100.0, 110.0, 120.0])
        assert _max_drawdown(eq) == pytest.approx(0.0, abs=1e-6)

    def test_50_pct_drawdown(self):
        eq = _eq([100.0, 50.0, 60.0])
        assert _max_drawdown(eq) == pytest.approx(-0.5, abs=1e-4)


class TestBacktestResultEmpty:
    def test_empty_has_empty_equity(self):
        r = BacktestResult.empty("Test")
        assert r.equity_curve.empty
        assert r.trade_log == []

    def test_metrics_returns_empty_when_too_few_bars(self):
        r = BacktestResult.empty("Test")
        assert r.metrics() == {}


class TestTrimAndScale:
    def test_normalises_to_100k(self):
        eq = _eq([50_000.0, 60_000.0, 70_000.0], start="2024-01-01")
        r = BacktestResult.trim_and_scale("X", eq, [], "2024-01-01")
        assert r.equity_curve.iloc[0] == pytest.approx(100_000.0)

    def test_trims_before_start_date(self):
        eq = _eq([100.0, 110.0, 120.0, 130.0], start="2024-01-01")
        r = BacktestResult.trim_and_scale("X", eq, [], "2024-01-03")
        assert len(r.equity_curve) < 4

    def test_empty_result_when_start_after_equity(self):
        eq = _eq([100.0, 110.0], start="2024-01-01")
        r = BacktestResult.trim_and_scale("X", eq, [], "2025-01-01")
        assert r.equity_curve.empty

    def test_filters_trades_before_start(self):
        eq = _eq([100.0, 110.0, 120.0], start="2024-01-01")
        trades = [
            TradeRecord(pd.Timestamp("2023-12-01"), "AAPL", "buy", 1.0, 100.0, 100.0),
            TradeRecord(pd.Timestamp("2024-01-02"), "AAPL", "sell", 1.0, 110.0, 110.0),
        ]
        r = BacktestResult.trim_and_scale("X", eq, trades, "2024-01-01")
        assert all(t.date >= pd.Timestamp("2024-01-01") for t in r.trade_log)


class TestMetrics:
    def test_returns_expected_keys(self):
        eq = _eq([100_000.0 * (1.0005 ** i) for i in range(252)])
        r = BacktestResult("Test", eq, [])
        m = r.metrics()
        for key in ("Total Return", "CAGR", "Sharpe", "Sortino", "Max Drawdown", "Calmar"):
            assert key in m

    def test_positive_sharpe_for_trending_equity(self):
        eq = _eq([100_000.0 * (1.001 ** i) for i in range(252)])
        r = BacktestResult("Test", eq, [])
        sharpe = float(r.metrics()["Sharpe"])
        assert sharpe > 0
