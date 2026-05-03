"""RL Trader — real A2C inference, mirroring scripts/trade.py."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import yaml

VENDOR = Path(__file__).parents[1] / "vendors" / "rl_trader"
sys.path.insert(0, str(VENDOR))

from backtesting.engine import run_simulation
from backtesting.metrics import BacktestResult
from fetchers.market_data import load_ohlcv

_agent = None
_config = None


def _init() -> None:
    global _agent, _config
    if _agent is not None:
        return

    with open(VENDOR / "config" / "settings.yaml") as f:
        _config = yaml.safe_load(f)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from agents.a2c_agent import A2CAgent

    a = A2CAgent(_config)
    a.load(str(VENDOR / "checkpoints" / "best_model.zip"))
    _agent = a


def _rl_signal(date, ohlcv, state):
    _init()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from data.features import compute_features, feature_columns
        from env.observation import ObservationBuilder
        from env.portfolio import PortfolioState

    feat_cols = feature_columns(use_macro=False)
    n_features = len(feat_cols)
    symbols = _config.get("data", {}).get("symbols", list(ohlcv.keys()))
    symbols = [s for s in symbols if s in ohlcv]

    feature_rows: dict[str, np.ndarray] = {}
    for sym in symbols:
        df = ohlcv.get(sym)
        if df is None or len(df) < 252:
            feature_rows[sym] = np.zeros(n_features, dtype=np.float32)
            continue
        feats = compute_features(df)
        if feats.empty:
            feature_rows[sym] = np.zeros(n_features, dtype=np.float32)
        else:
            feature_rows[sym] = feats.iloc[-1][feat_cols].fillna(0).values.astype(np.float32)

    if not feature_rows:
        return {}

    equity = state.get("__equity__", 100_000.0)
    holdings = state.get("__holdings__", {})
    cash = state.get("__cash__", equity)
    peak = state.get("rl_peak", equity)
    peak = max(peak, equity)
    state["rl_peak"] = peak
    drawdown = (equity - peak) / peak if peak > 0 else 0.0

    prices = {s: ohlcv[s]["close"].iloc[-1] for s in symbols if s in ohlcv}
    current_weights = {
        s: holdings.get(s, 0) * prices.get(s, 0) / equity if equity > 0 else 0.0
        for s in symbols
    }
    unrealized_pnl = {s: 0.0 for s in symbols}

    pstate = PortfolioState(
        equity=equity,
        cash_fraction=cash / equity if equity > 0 else 1.0,
        weights=current_weights,
        unrealized_pnl_pct=unrealized_pnl,
        drawdown_from_peak=drawdown,
        days_since_rebalance=state.get("rl_days", 0),
        step=state.get("rl_step", 0),
    )
    state["rl_step"] = state.get("rl_step", 0) + 1
    state["rl_days"] = state.get("rl_days", 0) + 1

    obs_builder = ObservationBuilder(symbols, n_features)
    episode_len = _config.get("environment", {}).get("episode_len", 252)
    obs = obs_builder.build(feature_rows=feature_rows, portfolio=pstate, episode_len=episode_len)

    raw_action = _agent.predict(obs, deterministic=True)
    # raw_action may be a tuple (action, state) from SB3
    if isinstance(raw_action, tuple):
        raw_action = raw_action[0]

    risk_cfg = _config.get("risk", {})
    max_pos = risk_cfg.get("max_single_position", 0.08)
    min_cash = risk_cfg.get("min_cash", 0.20)

    equity_weights = np.clip(raw_action[:-1], 0.0, max_pos)
    if equity_weights.sum() > 1.0 - min_cash:
        equity_weights = equity_weights * (1.0 - min_cash) / equity_weights.sum()

    return {
        symbols[i]: float(equity_weights[i])
        for i in range(len(symbols))
        if equity_weights[i] > 0.001
    }


def run_rl_trader(start: str, end: str, symbols=None) -> BacktestResult:
    _init()
    rl_symbols = _config.get("data", {}).get("symbols") if _config else symbols
    ohlcv = load_ohlcv(start, end, rl_symbols or symbols)
    if not ohlcv:
        return BacktestResult("RL Trader", __import__("pandas").Series(dtype=float), [])
    return run_simulation(ohlcv, _rl_signal, "RL Trader", rebalance_every=1)
