"""RL trading environment setup and inference helpers."""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from backtesting.trade_record import TradeRecord
from backtesting.vendor_path import setup_rl_paths

VENDOR = Path(__file__).parent


def _ensure_vendor_path() -> None:
    setup_rl_paths()


def _make_dummy_macro(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Zero-filled macro DataFrame so TradingEnv sees use_macro=True (→ 425-dim obs)."""
    return pd.DataFrame({"vix": 0.0, "yield_spread": 0.0, "credit_proxy": 0.0}, index=index)


def make_env(test_bars: dict, config: dict, start_equity: float):
    """Construct a TradingEnv for OOS inference with no noise or equity jitter."""
    _ensure_vendor_path()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from env.trading_env import TradingEnv
    cfg = {**config, "environment": {**config.get("environment", {}), "initial_capital": start_equity}}
    macro = _make_dummy_macro(test_bars[next(iter(test_bars))].index)
    return TradingEnv(test_bars, cfg, macro=macro, noise_sigma=0.0, equity_jitter=0.0)


def run_agent_loop(agent, env) -> dict[pd.Timestamp, float]:
    """Step agent through env to completion; return date → equity mapping."""
    common_dates = list(env._common_index)
    equity_by_date: dict[pd.Timestamp, float] = {}
    obs, _ = env.reset(seed=42)
    done = False
    while not done:
        obs, _r, terminated, truncated, info = env.step(agent.predict(obs, deterministic=True))
        done = terminated or truncated
        step = env._step_idx
        if step > 0 and step - 1 < len(common_dates):
            equity_by_date[common_dates[env._start_idx + step - 1]] = info["equity"]
    return equity_by_date


def convert_rl_trades(trade_log) -> list[TradeRecord]:
    """Convert RL portfolio rebalance events to dashboard TradeRecords."""
    return [
        TradeRecord(
            date=t.timestamp,
            symbol=t.symbol,
            side="buy" if t.new_weight > t.old_weight else "sell",
            qty=abs(t.turnover) / t.price if t.price > 0 else 0.0,
            price=t.price,
            value=abs(t.turnover),
        )
        for t in trade_log
    ]


def simulate_oos(
    agent, test_bars: dict, config: dict, start_equity: float
) -> tuple[pd.Series, list[TradeRecord]]:
    """Run OOS inference on test_bars with a loaded agent (no gradient updates)."""
    env = make_env(test_bars, config, start_equity)
    equity_by_date = run_agent_loop(agent, env)
    trades = convert_rl_trades(env._portfolio.trade_log)
    return pd.Series(equity_by_date).sort_index(), trades


def load_agent(fold_ckpt: Path, config: dict):
    """Load an A2CAgent from a checkpoint zip, suppressing deprecation warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from agents.a2c_agent import A2CAgent
        agent = A2CAgent(config)
        agent.load(str(fold_ckpt))
    return agent
