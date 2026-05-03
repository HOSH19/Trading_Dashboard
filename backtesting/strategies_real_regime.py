"""Regime Trader — real HMM + SignalGenerator, mirroring run_daily.py."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import yaml

VENDOR = Path(__file__).parents[1] / "vendors" / "regime_trader"
sys.path.insert(0, str(VENDOR))

from backtesting.engine import run_simulation
from backtesting.metrics import BacktestResult
from fetchers.market_data import load_ohlcv

_hmm = None
_signal_gen = None


def _init() -> None:
    global _hmm, _signal_gen
    if _signal_gen is not None:
        return

    with open(VENDOR / "config" / "settings.yaml") as f:
        config = yaml.safe_load(f)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from core.hmm import HMMEngine
        from core.signal_generator import SignalGenerator
        from core.strategies import StrategyOrchestrator

    _hmm = HMMEngine(config.get("hmm", {}))
    _hmm.load(str(VENDOR / "hmm_model.pkl"))

    orchestrator = StrategyOrchestrator(config.get("strategy", {}), _hmm.regime_infos)
    _signal_gen = SignalGenerator(_hmm, orchestrator, config)


def _regime_signal(date, ohlcv, state):
    _init()
    try:
        signals, _ = _signal_gen.generate(
            symbols=list(ohlcv.keys()),
            bars_by_symbol=ohlcv,
            current_allocations=state.get("weights", {}),
        )
    except Exception:
        return {}

    result = {
        s.symbol: s.position_size_pct
        for s in signals
        if s.direction == "long" and s.position_size_pct > 0.001
    }
    state["weights"] = result
    return result


def run_regime_trader(start: str, end: str, symbols=None) -> BacktestResult:
    _init()
    # HMM requires SPY/QQQ/IWM/DIA for multi-symbol feature extraction
    hmm_syms = ["IWM", "DIA"]
    extra = [s for s in hmm_syms if s not in (symbols or [])]
    ohlcv = load_ohlcv(start, end, (symbols or []) + extra)
    if not ohlcv:
        return BacktestResult("Regime Trader", __import__("pandas").Series(dtype=float), [])
    return run_simulation(ohlcv, _regime_signal, "Regime Trader", rebalance_every=5)
