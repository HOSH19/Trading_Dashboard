"""Tests for stop processing, trailing stops, and rebalance order helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from backtesting.engine.orders import process_stops, update_trailing_stops
from backtesting.engine.rebalance import rebalance
from backtesting.trade_record import TradeRecord

DATE = pd.Timestamp("2024-01-15")


def _holdings(**kwargs):
    return dict(kwargs)


class TestProcessStops:
    def test_exit_when_low_touches_stop(self):
        holdings = {"AAPL": 10.0}
        stop_prices = {"AAPL": 150.0}
        prices = {"AAPL": 155.0}
        lows = {"AAPL": 149.0}
        trades = []
        cash = process_stops(DATE, holdings, stop_prices, prices, lows, trades, 0.0, 0.0)
        assert "AAPL" not in holdings
        assert len(trades) == 1
        assert trades[0].side == "sell"
        assert cash == pytest.approx(1500.0)

    def test_no_exit_when_low_above_stop(self):
        holdings = {"AAPL": 10.0}
        stop_prices = {"AAPL": 140.0}
        prices = {"AAPL": 155.0}
        lows = {"AAPL": 152.0}
        trades = []
        cash = process_stops(DATE, holdings, stop_prices, prices, lows, trades, 0.0, 0.0)
        assert "AAPL" in holdings
        assert trades == []
        assert cash == 0.0

    def test_exit_price_floored_at_stop_not_below(self):
        holdings = {"AAPL": 10.0}
        stop_prices = {"AAPL": 150.0}
        prices = {"AAPL": 155.0}
        lows = {"AAPL": 145.0}
        trades = []
        process_stops(DATE, holdings, stop_prices, prices, lows, trades, 0.0, 0.0)
        assert trades[0].price == 150.0

    def test_commission_deducted(self):
        holdings = {"AAPL": 10.0}
        stop_prices = {"AAPL": 100.0}
        prices = {"AAPL": 100.0}
        lows = {"AAPL": 100.0}
        trades = []
        cash = process_stops(DATE, holdings, stop_prices, prices, lows, trades, 0.0, 0.001)
        assert cash == pytest.approx(10.0 * 100.0 * (1 - 0.001))


class TestUpdateTrailingStops:
    def test_ratchet_up_when_price_rises(self):
        holdings = {"AAPL": 10.0}
        stop_prices = {"AAPL": 90.0}
        prices = {"AAPL": 110.0}
        state = {"trail_AAPL": 0.10}
        update_trailing_stops(holdings, stop_prices, prices, state)
        assert stop_prices["AAPL"] == pytest.approx(110.0 * 0.90)

    def test_does_not_lower_stop(self):
        holdings = {"AAPL": 10.0}
        stop_prices = {"AAPL": 95.0}
        prices = {"AAPL": 100.0}
        state = {"trail_AAPL": 0.10}
        update_trailing_stops(holdings, stop_prices, prices, state)
        assert stop_prices["AAPL"] == pytest.approx(95.0)

    def test_skips_symbol_with_no_trail(self):
        holdings = {"AAPL": 10.0}
        stop_prices = {}
        prices = {"AAPL": 100.0}
        update_trailing_stops(holdings, stop_prices, prices, {})
        assert stop_prices == {}


class TestRebalance:
    def _base_state(self):
        return {"holdings": {}, "stop_prices": {}, "trades": [], "state": {}}

    def test_buys_into_new_position(self):
        holdings = {}
        stop_prices = {}
        trades = []
        cash = rebalance(DATE, {"AAPL": 0.10}, holdings, stop_prices,
                         {"AAPL": 100.0}, trades, 10_000.0, 10_000.0, 0.0, {})
        assert "AAPL" in holdings
        assert any(t.side == "buy" for t in trades)
        assert cash < 10_000.0

    def test_exits_position_not_in_target(self):
        holdings = {"AAPL": 10.0}
        stop_prices = {}
        trades = []
        cash = rebalance(DATE, {}, holdings, stop_prices,
                         {"AAPL": 100.0}, trades, 0.0, 10_000.0, 0.0, {})
        assert "AAPL" not in holdings
        assert trades[-1].side == "sell"
        assert cash == pytest.approx(1000.0)

    def test_skips_tiny_delta(self):
        holdings = {"AAPL": 9.6}
        stop_prices = {}
        trades = []
        rebalance(DATE, {"AAPL": 0.10}, holdings, stop_prices,
                  {"AAPL": 100.0}, trades, 0.0, 10_000.0, 0.0, {})
        assert trades == []

    def test_sets_trailing_stop_on_entry(self):
        holdings = {}
        stop_prices = {}
        trades = []
        rebalance(DATE, {"AAPL": 0.10}, holdings, stop_prices,
                  {"AAPL": 100.0}, trades, 10_000.0, 10_000.0, 0.0, {"trail_AAPL": 0.10})
        assert stop_prices.get("AAPL") == pytest.approx(90.0)

    def test_partial_sell_to_reduce_position(self):
        holdings = {"AAPL": 20.0}
        stop_prices = {}
        trades = []
        cash = rebalance(DATE, {"AAPL": 0.05}, holdings, stop_prices,
                         {"AAPL": 100.0}, trades, 0.0, 10_000.0, 0.0, {})
        assert holdings["AAPL"] < 20.0
        assert any(t.side == "sell" for t in trades)
