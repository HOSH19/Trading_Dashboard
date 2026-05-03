"""Regime Trader — real HMM + SignalGenerator, mirroring run_daily.py."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import yaml

VENDOR = Path(__file__).parents[1] / "vendors" / "regime_trader"
RL_VENDOR = Path(__file__).parents[1] / "vendors" / "rl_trader"


def _ensure_vendor_path() -> None:
    """Keep regime_trader vendor at the front so its `data` package wins over rl_trader's."""
    vendor_str = str(VENDOR)
    rl_str = str(RL_VENDOR)
    # Remove both then re-insert in correct order: regime first, rl second
    for p in (vendor_str, rl_str):
        while p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, rl_str)
    sys.path.insert(0, vendor_str)

from backtesting.engine import run_simulation
from backtesting.metrics import BacktestResult
from fetchers.market_data import load_ohlcv

_hmm = None
_signal_gen = None


def _init() -> None:
    global _hmm, _signal_gen
    if _signal_gen is not None:
        return

    _ensure_vendor_path()

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
    _ensure_vendor_path()
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
