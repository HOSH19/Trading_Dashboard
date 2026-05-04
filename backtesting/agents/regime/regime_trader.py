"""Regime Trader — lazy HMM loader and backtest entry point."""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import yaml

from backtesting.backtest_result import BacktestResult
from backtesting.engine import run_simulation
from backtesting.vendor_path import setup_regime_paths
from config import UNIVERSE
from fetchers.market_data import load_ohlcv

_VENDOR = Path(__file__).parent
_HMM_ONLY = {"IWM", "DIA"}
MAX_SINGLE_POSITION = 0.08
MAX_TOTAL_EXPOSURE = 0.80

_signal_gen = None


def _init() -> None:
    """Load HMM model and SignalGenerator on first call; no-op on subsequent calls."""
    global _signal_gen
    if _signal_gen is not None:
        return
    setup_regime_paths()
    with open(_VENDOR / "config" / "settings.yaml") as f:
        config = yaml.safe_load(f)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from core.hmm import HMMEngine
        from core.signal_generator import SignalGenerator
        from core.strategies import StrategyOrchestrator
    hmm = HMMEngine(config.get("hmm", {}))
    hmm.load(str(_VENDOR / "hmm_model.pkl"))
    orchestrator = StrategyOrchestrator(config.get("strategy", {}), hmm.regime_infos)
    _signal_gen = SignalGenerator(hmm, orchestrator, config)


def _apply_risk_caps(raw: dict[str, float]) -> dict[str, float]:
    """Mirror the live RiskManager caps: 8% per position, 80% total exposure."""
    result = {sym: min(w, MAX_SINGLE_POSITION) for sym, w in raw.items()}
    total = sum(result.values())
    if total > MAX_TOTAL_EXPOSURE:
        scale = MAX_TOTAL_EXPOSURE / total
        result = {sym: w * scale for sym, w in result.items()}
    return result


def _regime_signal(date, ohlcv, state) -> dict[str, float]:
    setup_regime_paths()
    trading_syms = [s for s in ohlcv if s not in _HMM_ONLY]
    try:
        signals, _ = _signal_gen.generate(
            symbols=trading_syms,
            bars_by_symbol=ohlcv,
            current_allocations=state.get("weights", {}),
        )
    except Exception:
        return {}
    raw = {
        s.symbol: s.position_size_pct * s.leverage
        for s in signals
        if s.direction.upper() == "LONG" and s.position_size_pct > 0.001
    }
    result = _apply_risk_caps(raw)
    state["weights"] = result
    return result


def run_regime_trader(start: str, end: str, symbols=None) -> BacktestResult:
    """Run the Regime Trader backtest.

    Loads 25 months of warmup data so the HMM has valid feature rows
    (SMA200=200 + zscore_window=252 requires ~452 raw bars before first valid row).
    """
    _init()
    warmup_start = (pd.Timestamp(start) - pd.DateOffset(months=25)).strftime("%Y-%m-%d")
    all_syms = list({*(symbols or UNIVERSE), "IWM", "DIA"})
    ohlcv = load_ohlcv(warmup_start, end, all_syms)
    if not ohlcv:
        return BacktestResult.empty("Regime Trader")
    result = run_simulation(ohlcv, _regime_signal, "Regime Trader", rebalance_every=5)
    return BacktestResult.trim_and_scale("Regime Trader", result.equity_curve, result.trade_log, start)
