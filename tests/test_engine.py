"""Tests for the backtest simulation engine (run_simulation)."""

from __future__ import annotations

import pandas as pd
import pytest

from backtesting.engine import run_simulation


def _ohlcv(prices: list[float], symbol: str = "AAPL") -> dict:
    idx = pd.date_range("2024-01-01", periods=len(prices), freq="B")
    df = pd.DataFrame({
        "open": prices, "high": prices, "low": prices, "close": prices, "volume": 1_000_000.0,
    }, index=idx)
    return {symbol: df}


def _hold_signal(sym):
    """Signal that always targets 50% in sym."""
    def signal(date, ohlcv, state):
        return {sym: 0.50}
    return signal


def _no_trade_signal(date, ohlcv, state):
    return {}


class TestRunSimulation:
    def test_equity_starts_at_initial_capital(self):
        ohlcv = _ohlcv([100.0] * 20)
        result = run_simulation(ohlcv, _no_trade_signal, "Test")
        assert result.equity_curve.iloc[0] == pytest.approx(100_000.0)

    def test_equity_grows_with_rising_prices(self):
        prices = [100.0 + i for i in range(50)]
        ohlcv = _ohlcv(prices)
        result = run_simulation(ohlcv, _hold_signal("AAPL"), "Test", rebalance_every=1)
        assert result.equity_curve.iloc[-1] > result.equity_curve.iloc[0]

    def test_no_trades_when_signal_is_empty(self):
        ohlcv = _ohlcv([100.0] * 20)
        result = run_simulation(ohlcv, _no_trade_signal, "Test")
        assert result.trade_log == []

    def test_equity_curve_length_matches_trading_days(self):
        n = 30
        ohlcv = _ohlcv([100.0] * n)
        result = run_simulation(ohlcv, _no_trade_signal, "Test")
        assert len(result.equity_curve) == n

    def test_strategy_name_preserved(self):
        ohlcv = _ohlcv([100.0] * 5)
        result = run_simulation(ohlcv, _no_trade_signal, "MyStrategy")
        assert result.strategy == "MyStrategy"

    def test_buys_happen_on_rebalance_bars(self):
        ohlcv = _ohlcv([100.0] * 20)
        result = run_simulation(ohlcv, _hold_signal("AAPL"), "Test", rebalance_every=5)
        buys = [t for t in result.trade_log if t.side == "buy"]
        assert len(buys) >= 1

    def test_stop_loss_triggers_sell(self):
        prices = [100.0] * 5 + [80.0] * 5
        idx = pd.date_range("2024-01-01", periods=10, freq="B")
        df = pd.DataFrame({
            "open": prices, "high": prices, "low": prices, "close": prices, "volume": 1_000_000.0,
        }, index=idx)
        ohlcv = {"AAPL": df}

        def signal_with_stop(date, ohlcv, state):
            state["trail_AAPL"] = 0.10
            return {"AAPL": 0.50}

        result = run_simulation(ohlcv, signal_with_stop, "Test", rebalance_every=1)
        sells = [t for t in result.trade_log if t.side == "sell"]
        assert len(sells) >= 1

    def test_commission_reduces_final_equity(self):
        prices = [100.0] * 20
        ohlcv = _ohlcv(prices)
        result_free = run_simulation(ohlcv, _hold_signal("AAPL"), "Test", commission=0.0, rebalance_every=1)
        result_paid = run_simulation(ohlcv, _hold_signal("AAPL"), "Test", commission=0.001, rebalance_every=1)
        assert result_paid.equity_curve.iloc[-1] < result_free.equity_curve.iloc[-1]

    def test_multi_symbol_rebalance(self):
        ohlcv = {
            "AAPL": _ohlcv([100.0] * 20)["AAPL"],
            "MSFT": _ohlcv([200.0] * 20, symbol="MSFT")["MSFT"],
        }

        def two_asset_signal(date, ohlcv, state):
            return {"AAPL": 0.30, "MSFT": 0.30}

        result = run_simulation(ohlcv, two_asset_signal, "Test", rebalance_every=5)
        symbols_bought = {t.symbol for t in result.trade_log if t.side == "buy"}
        assert "AAPL" in symbols_bought
        assert "MSFT" in symbols_bought
