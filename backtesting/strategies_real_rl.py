"""RL Trader — per-fold inference using pre-trained walk-forward checkpoints.

For each fold produced by build_folds() we load fold_N/best_model.zip and run
OOS inference on that fold's test window only, then stitch the equity curves.
No training happens here — the checkpoints must already exist.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dataclasses import dataclass
from typing import Dict


@dataclass
class _Fold:
    fold_idx: int
    test_bars: Dict[str, pd.DataFrame]
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _build_folds(
    bars_by_symbol: dict,
    train_window: int,
    test_window: int,
    step_size: int,
) -> list[_Fold]:
    """Replicate build_folds() from RL_Trader/data/dataset.py without importing it."""
    idx = None
    for df in bars_by_symbol.values():
        idx = df.index if idx is None else idx.intersection(df.index)
    common = idx.sort_values()
    total = len(common)

    folds, fold_idx, start = [], 0, 0
    while start + train_window + test_window <= total:
        train_end = start + train_window
        test_end = min(train_end + test_window, total)
        test_idx = common[train_end:test_end]
        folds.append(_Fold(
            fold_idx=fold_idx,
            test_bars={sym: df.loc[test_idx] for sym, df in bars_by_symbol.items()},
            test_start=test_idx[0],
            test_end=test_idx[-1],
        ))
        fold_idx += 1
        start += step_size
    return folds

VENDOR = Path(__file__).parents[1] / "vendors" / "rl_trader"
REGIME_VENDOR = Path(__file__).parents[1] / "vendors" / "regime_trader"

# Pre-trained fold checkpoints — vendored inside the dashboard repo.
# Override via RL_CHECKPOINT_DIR env var if needed.
_DEFAULT_CHECKPOINT_DIR = VENDOR / "checkpoints" / "a2c_20sym"


def _ensure_vendor_path() -> None:
    rl_str = str(VENDOR)
    regime_str = str(REGIME_VENDOR)
    for p in (rl_str, regime_str):
        while p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, regime_str)
    sys.path.insert(0, rl_str)
    for key in list(sys.modules.keys()):
        if key == "data" or key.startswith("data."):
            del sys.modules[key]


from backtesting.metrics import BacktestResult
from fetchers.market_data import load_ohlcv

_date_range_cache: tuple[pd.Timestamp, pd.Timestamp] | None = None


def rl_date_range() -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (min_date, max_date) covered by the vendored fold checkpoints.

    Computes fold boundaries with the same arithmetic as build_folds(), but
    without importing from the vendor — avoiding sys.path/sys.modules conflicts.
    Result is cached for the session lifetime.
    """
    global _date_range_cache
    if _date_range_cache is not None:
        return _date_range_cache

    import os

    with open(VENDOR / "config" / "settings.yaml") as f:
        config = yaml.safe_load(f)

    checkpoint_dir = Path(os.environ.get("RL_CHECKPOINT_DIR", str(_DEFAULT_CHECKPOINT_DIR)))
    available_folds = sorted(
        int(p.name.split("_")[1])
        for p in checkpoint_dir.iterdir()
        if p.is_dir() and p.name.startswith("fold_") and (p / "best_model.zip").exists()
    )
    if not available_folds:
        raise RuntimeError(f"No fold checkpoints found in {checkpoint_dir}")

    train_cfg = config.get("training", {})
    train_window = train_cfg.get("train_window", 252)
    test_window = train_cfg.get("test_window", 126)
    step_size = train_cfg.get("step_size", 126)

    # Load one symbol's trading-day index to reconstruct fold boundaries
    symbols = config.get("data", {}).get("symbols", ["SPY"])
    anchor = symbols[0]
    total_bars_needed = train_window + (max(available_folds) + 1) * step_size + test_window + 252
    warmup_months = int(total_bars_needed / 21) + 3
    earliest = (pd.Timestamp.today() - pd.DateOffset(months=warmup_months)).strftime("%Y-%m-%d")
    ohlcv = load_ohlcv(earliest, None, [anchor])
    if not ohlcv:
        raise RuntimeError("Could not fetch data to determine RL date range")

    common_index = ohlcv[anchor].index.sort_values()
    total = len(common_index)

    # Replicate build_folds() boundary arithmetic — no vendor import needed
    test_starts, test_ends = [], []
    fold_idx = 0
    start = 0
    while start + train_window + test_window <= total:
        train_end_pos = start + train_window
        test_end_pos = min(train_end_pos + test_window, total)
        if fold_idx in available_folds:
            test_starts.append(common_index[train_end_pos])
            test_ends.append(common_index[test_end_pos - 1])
        fold_idx += 1
        start += step_size

    if not test_starts:
        raise RuntimeError("Fold date reconstruction returned no matching folds")

    result = (test_starts[0], test_ends[-1])
    _date_range_cache = result
    return result


def _simulate_oos(agent, fold, config: dict, start_equity: float):
    """Run one OOS fold with a loaded agent (no gradient updates)."""
    _ensure_vendor_path()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from env.trading_env import TradingEnv

    cfg = dict(config)
    env_cfg = dict(cfg.get("environment", {}))
    env_cfg["initial_capital"] = start_equity
    cfg["environment"] = env_cfg

    env = TradingEnv(fold.test_bars, cfg, macro=None, noise_sigma=0.0, equity_jitter=0.0)
    obs, _ = env.reset(seed=42)

    equity_by_date: dict[pd.Timestamp, float] = {}
    common_dates = list(env._common_index)
    done = False

    while not done:
        action = agent.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step = env._step_idx
        if step > 0 and step - 1 < len(common_dates):
            date = common_dates[env._start_idx + step - 1]
            equity_by_date[date] = info["equity"]

    return pd.Series(equity_by_date).sort_index()


def run_rl_trader(start: str, end: str, symbols=None) -> BacktestResult:
    import os

    _ensure_vendor_path()

    with open(VENDOR / "config" / "settings.yaml") as f:
        config = yaml.safe_load(f)

    checkpoint_dir = Path(os.environ.get("RL_CHECKPOINT_DIR", str(_DEFAULT_CHECKPOINT_DIR)))
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"RL checkpoint directory not found: {checkpoint_dir}\n"
            "Set RL_CHECKPOINT_DIR env var to the folder containing fold_0/, fold_1/, …"
        )

    rl_symbols = symbols or config.get("data", {}).get("symbols")
    train_cfg = config.get("training", {})
    train_window = train_cfg.get("train_window", 252)
    warmup_months = max(24, int((train_window + 252) / 21) + 2)
    warmup_start = (pd.Timestamp(start) - pd.DateOffset(months=warmup_months)).strftime("%Y-%m-%d")

    ohlcv = load_ohlcv(warmup_start, end, rl_symbols)
    if not ohlcv:
        return BacktestResult("RL Trader", pd.Series(dtype=float), [])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from agents.a2c_agent import A2CAgent

    folds = _build_folds(
        ohlcv,
        train_window=train_window,
        test_window=train_cfg.get("test_window", 126),
        step_size=train_cfg.get("step_size", 126),
    )
    if not folds:
        return BacktestResult("RL Trader", pd.Series(dtype=float), [])

    initial_capital = config.get("environment", {}).get("initial_capital", 100_000.0)
    all_equity: dict[pd.Timestamp, float] = {}
    carry_equity = initial_capital

    start_ts = pd.Timestamp(start)

    for fold in folds:
        # Skip folds whose OOS window ends before the requested start
        if fold.test_end < start_ts:
            continue

        fold_ckpt = checkpoint_dir / f"fold_{fold.fold_idx}" / "best_model.zip"
        if not fold_ckpt.exists():
            print(f"[RL Trader] missing checkpoint {fold_ckpt}, skipping fold {fold.fold_idx}")
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            agent = A2CAgent(config)
            agent.load(str(fold_ckpt))

        oos_eq = _simulate_oos(agent, fold, config, carry_equity)

        if not oos_eq.empty:
            carry_equity = float(oos_eq.iloc[-1])
            for date, val in oos_eq.items():
                all_equity[date] = val

    if not all_equity:
        return BacktestResult("RL Trader", pd.Series(dtype=float), [])

    eq = pd.Series(all_equity).sort_index()
    if eq.index.tz is not None:
        start_ts = start_ts.tz_localize(eq.index.tz)
    eq_trimmed = eq[eq.index >= start_ts]
    if eq_trimmed.empty:
        return BacktestResult("RL Trader", pd.Series(dtype=float), [])

    scale = 100_000.0 / eq_trimmed.iloc[0]
    return BacktestResult("RL Trader", eq_trimmed * scale, [])
