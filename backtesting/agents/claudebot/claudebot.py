"""Claudebot entry point — wires cache and OHLCV data into ClaudebotSignal."""

from __future__ import annotations

import json
from pathlib import Path

from backtesting.agents.claudebot.claudebot_signal import ClaudebotSignal
from backtesting.backtest_result import BacktestResult
from backtesting.engine import run_simulation
from fetchers.market_data import load_ohlcv

CACHE_DIR = Path(__file__).parent.parent.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)


def _load_cache(cache_path: Path) -> dict:
    try:
        return json.loads(cache_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _flush_cache(cache: dict, cache_path: Path) -> None:
    try:
        cache_path.write_text(json.dumps(cache))
    except OSError:
        pass


def run_claudebot_api(start: str, end: str, symbols=None) -> BacktestResult:
    cache_path = CACHE_DIR / f"claudebot_{start}_{end}.json"
    cache = _load_cache(cache_path)

    ohlcv = load_ohlcv(start, end, symbols)
    if not ohlcv:
        return BacktestResult.empty("Claudebot")

    try:
        return run_simulation(ohlcv, ClaudebotSignal(cache, cache_path), "Claudebot", rebalance_every=1)
    finally:
        _flush_cache(cache, cache_path)
