"""RL Trader — per-fold inference using pre-trained walk-forward checkpoints.

fold_dates.json in the checkpoint directory is the single source of truth for
which fold index maps to which test date window. This avoids rebuilding fold
boundaries from data (fragile due to symbol-intersection differences).
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml

VENDOR = Path(__file__).parents[1] / "vendors" / "rl_trader"
REGIME_VENDOR = Path(__file__).parents[1] / "vendors" / "regime_trader"

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


from backtesting.metrics import BacktestResult, TradeRecord
from fetchers.market_data import load_ohlcv


def _checkpoint_dir() -> Path:
    return Path(os.environ.get("RL_CHECKPOINT_DIR", str(_DEFAULT_CHECKPOINT_DIR)))


def _load_fold_dates() -> dict[int, tuple[pd.Timestamp, pd.Timestamp]]:
    """Load fold index → (test_start, test_end) from fold_dates.json."""
    path = _checkpoint_dir() / "fold_dates.json"
    if not path.exists():
        raise FileNotFoundError(
            f"fold_dates.json not found at {path}. "
            "Re-run the checkpoint export script to regenerate it."
        )
    raw = json.loads(path.read_text())
    return {
        int(k): (pd.Timestamp(v["test_start"]), pd.Timestamp(v["test_end"]))
        for k, v in raw.items()
        if (_checkpoint_dir() / f"fold_{k}" / "best_model.zip").exists()
    }


_date_range_cache: tuple[pd.Timestamp, pd.Timestamp] | None = None


def rl_date_range() -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (min_date, max_date) covered by the vendored fold checkpoints."""
    global _date_range_cache
    if _date_range_cache is not None:
        return _date_range_cache

    fold_dates = _load_fold_dates()
    if not fold_dates:
        raise RuntimeError("No fold checkpoints found")

    min_fold = min(fold_dates)
    max_fold = max(fold_dates)
    result = (fold_dates[min_fold][0], fold_dates[max_fold][1])
    _date_range_cache = result
    return result


def _make_dummy_macro(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Zero-filled macro so TradingEnv uses use_macro=True → 425-dim obs."""
    return pd.DataFrame(
        {"vix": 0.0, "yield_spread": 0.0, "credit_proxy": 0.0},
        index=index,
    )


def _simulate_oos(agent, test_bars: dict, config: dict, start_equity: float) -> pd.Series:
    """Run OOS inference on test_bars with a loaded agent (no gradient updates)."""
    _ensure_vendor_path()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from env.trading_env import TradingEnv

    cfg = dict(config)
    env_cfg = dict(cfg.get("environment", {}))
    env_cfg["initial_capital"] = start_equity
    cfg["environment"] = env_cfg

    first_sym = next(iter(test_bars))
    dummy_macro = _make_dummy_macro(test_bars[first_sym].index)
    env = TradingEnv(test_bars, cfg, macro=dummy_macro, noise_sigma=0.0, equity_jitter=0.0)
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

    eq = pd.Series(equity_by_date).sort_index()

    # Convert RL Trade objects (rebalance events) to dashboard TradeRecords
    trades: list[TradeRecord] = []
    for t in env._portfolio.trade_log:
        side = "buy" if t.new_weight > t.old_weight else "sell"
        qty = abs(t.turnover) / t.price if t.price > 0 else 0.0
        trades.append(TradeRecord(
            date=t.timestamp,
            symbol=t.symbol,
            side=side,
            qty=qty,
            price=t.price,
            value=abs(t.turnover),
        ))

    return eq, trades


def run_rl_trader(start: str, end: str, symbols=None) -> BacktestResult:
    _ensure_vendor_path()

    with open(VENDOR / "config" / "settings.yaml") as f:
        config = yaml.safe_load(f)

    fold_dates = _load_fold_dates()
    if not fold_dates:
        return BacktestResult("RL Trader", pd.Series(dtype=float), [])

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    # Select only folds whose test window overlaps [start, end]
    relevant = {
        idx: (ts, te)
        for idx, (ts, te) in fold_dates.items()
        if te >= start_ts and ts <= end_ts
    }
    if not relevant:
        return BacktestResult("RL Trader", pd.Series(dtype=float), [])

    rl_symbols = symbols or config.get("data", {}).get("symbols")

    # Load only the date range needed: earliest test_start to end
    earliest = min(ts for ts, _ in relevant.values()).strftime("%Y-%m-%d")
    ohlcv = load_ohlcv(earliest, end, rl_symbols)
    if not ohlcv:
        return BacktestResult("RL Trader", pd.Series(dtype=float), [])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from agents.a2c_agent import A2CAgent

    ckpt_dir = _checkpoint_dir()
    initial_capital = config.get("environment", {}).get("initial_capital", 100_000.0)
    all_equity: dict[pd.Timestamp, float] = {}
    all_trades: list[TradeRecord] = []
    carry_equity = initial_capital

    for fold_idx in sorted(relevant):
        ts, te = relevant[fold_idx]
        fold_ckpt = ckpt_dir / f"fold_{fold_idx}" / "best_model.zip"

        # Slice test bars for this fold from the loaded data
        test_bars = {
            sym: df.loc[(df.index >= ts) & (df.index <= te)]
            for sym, df in ohlcv.items()
            if not df.loc[(df.index >= ts) & (df.index <= te)].empty
        }
        if not test_bars:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            agent = A2CAgent(config)
            agent.load(str(fold_ckpt))

        oos_eq, oos_trades = _simulate_oos(agent, test_bars, config, carry_equity)

        if not oos_eq.empty:
            carry_equity = float(oos_eq.iloc[-1])
            for date, val in oos_eq.items():
                all_equity[date] = val
            all_trades.extend(oos_trades)

    if not all_equity:
        return BacktestResult("RL Trader", pd.Series(dtype=float), [])

    eq = pd.Series(all_equity).sort_index()
    eq_trimmed = eq[eq.index >= start_ts]
    if eq_trimmed.empty:
        return BacktestResult("RL Trader", pd.Series(dtype=float), [])

    scale = 100_000.0 / eq_trimmed.iloc[0]
    trades_trimmed = [t for t in all_trades if t.date >= start_ts]
    return BacktestResult("RL Trader", eq_trimmed * scale, trades_trimmed)
