"""RL Trader — per-fold inference using pre-trained walk-forward checkpoints."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from backtesting.agents.rl.rl_env import _ensure_vendor_path, load_agent, simulate_oos
from backtesting.agents.rl.rl_fold_loader import checkpoint_dir, load_fold_dates, relevant_folds, rl_date_range  # noqa: F401
from backtesting.backtest_result import BacktestResult
from backtesting.trade_record import TradeRecord
from fetchers.market_data import load_ohlcv

VENDOR = Path(__file__).parent


def _load_config() -> dict:
    with open(VENDOR / "config" / "settings.yaml") as f:
        return yaml.safe_load(f)


def _slice_fold(ohlcv: dict, ts: pd.Timestamp, te: pd.Timestamp) -> dict:
    return {
        sym: sliced
        for sym, df in ohlcv.items()
        if not (sliced := df.loc[(df.index >= ts) & (df.index <= te)]).empty
    }


def _run_all_folds(
    relevant: dict, ohlcv: dict, ckpt_dir: Path, config: dict, initial_capital: float
) -> tuple[dict, list[TradeRecord]]:
    """Iterate over relevant folds in order, chaining equity from one fold to the next."""
    all_equity: dict[pd.Timestamp, float] = {}
    all_trades: list[TradeRecord] = []
    carry = initial_capital
    for fold_idx in sorted(relevant):
        ts, te = relevant[fold_idx]
        test_bars = _slice_fold(ohlcv, ts, te)
        if not test_bars:
            continue
        agent = load_agent(ckpt_dir / f"fold_{fold_idx}" / "best_model.zip", config)
        eq, trades = simulate_oos(agent, test_bars, config, carry)
        if not eq.empty:
            carry = float(eq.iloc[-1])
            all_equity.update(eq.to_dict())
            all_trades.extend(trades)
    return all_equity, all_trades




def run_rl_trader(start: str, end: str, symbols=None) -> BacktestResult:
    _ensure_vendor_path()
    config = _load_config()
    fold_dates = load_fold_dates()

    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    folds = relevant_folds(fold_dates, start_ts, end_ts)
    if not folds:
        return BacktestResult.empty("RL Trader")

    rl_symbols = symbols or config.get("data", {}).get("symbols")
    earliest = min(ts for ts, _ in folds.values()).strftime("%Y-%m-%d")
    ohlcv = load_ohlcv(earliest, end, rl_symbols)
    if not ohlcv:
        return BacktestResult.empty("RL Trader")

    initial_capital = config.get("environment", {}).get("initial_capital", 100_000.0)
    all_equity, all_trades = _run_all_folds(folds, ohlcv, checkpoint_dir(), config, initial_capital)
    if not all_equity:
        return BacktestResult.empty("RL Trader")
    return BacktestResult.trim_and_scale("RL Trader", pd.Series(all_equity).sort_index(), all_trades, start)
