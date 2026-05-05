"""Tests for matched-trade statistics."""

from __future__ import annotations

import pandas as pd
import pytest

from backtesting.trade_record import TradeRecord
from backtesting.trade_stats import matched_trades, trade_stats

D = lambda s: pd.Timestamp(s)  # noqa: E731


def _log(*sides_prices):
    """Build a trade log from alternating (sym, side, price) tuples on sequential dates."""
    log = []
    for i, (sym, side, price) in enumerate(sides_prices):
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)
        log.append(TradeRecord(date, sym, side, 1.0, price, price))
    return log


class TestMatchedTrades:
    def test_single_round_trip(self):
        log = _log(("AAPL", "buy", 100.0), ("AAPL", "sell", 110.0))
        trips = list(matched_trades(log))
        assert len(trips) == 1
        ep, xp, *_ = trips[0]
        assert ep == pytest.approx(100.0)
        assert xp == pytest.approx(110.0)

    def test_no_match_without_sell(self):
        log = _log(("AAPL", "buy", 100.0))
        assert list(matched_trades(log)) == []

    def test_fifo_matching(self):
        log = _log(("AAPL", "buy", 100.0), ("AAPL", "buy", 120.0), ("AAPL", "sell", 130.0))
        trips = list(matched_trades(log))
        assert trips[0][0] == pytest.approx(100.0)


class TestTradeStats:
    def test_empty_log_returns_nones(self):
        stats = trade_stats([], 100_000.0)
        assert all(v is None for v in stats.values())

    def test_win_rate_all_wins(self):
        log = _log(("AAPL", "buy", 100.0), ("AAPL", "sell", 120.0),
                   ("MSFT", "buy", 200.0), ("MSFT", "sell", 240.0))
        stats = trade_stats(log, 100_000.0)
        assert stats["win_rate"] == pytest.approx(1.0)

    def test_win_rate_all_losses(self):
        log = _log(("AAPL", "buy", 100.0), ("AAPL", "sell", 80.0))
        stats = trade_stats(log, 100_000.0)
        assert stats["win_rate"] == pytest.approx(0.0)

    def test_profit_factor_none_when_no_losses(self):
        log = _log(("AAPL", "buy", 100.0), ("AAPL", "sell", 120.0))
        stats = trade_stats(log, 100_000.0)
        assert stats["profit_factor"] is None

    def test_max_consec_losses(self):
        log = _log(
            ("A", "buy", 100.0), ("A", "sell", 90.0),
            ("B", "buy", 100.0), ("B", "sell", 90.0),
            ("C", "buy", 100.0), ("C", "sell", 110.0),
        )
        stats = trade_stats(log, 100_000.0)
        assert stats["max_consec_losses"] == 2

    def test_avg_hold_days(self):
        log = [
            TradeRecord(D("2024-01-01"), "AAPL", "buy", 1.0, 100.0, 100.0),
            TradeRecord(D("2024-01-11"), "AAPL", "sell", 1.0, 110.0, 110.0),
        ]
        stats = trade_stats(log, 100_000.0)
        assert stats["avg_hold"] == pytest.approx(10.0)
