"""
Strategy proxy implementations for backtesting.

Public API:  run_rl_trader | run_regime_trader | run_claudebot | run_spy_benchmark
Each strategy has one _signal function + focused helpers. No cross-strategy coupling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting.engine import BacktestResult, atr, rsi, rolling_sharpe, run_simulation, sma
from config import DEFENSIVE, SECTOR_MAP
from fetchers.market_data import load_ohlcv


# ── Shared utilities ──────────────────────────────────────────────────────────

def _empty(name: str) -> BacktestResult:
    return BacktestResult(name, pd.Series(dtype=float), [])


def _run(name: str, signal_fn, start: str, end: str, symbols=None, rebalance_every: int = 5) -> BacktestResult:
    ohlcv = load_ohlcv(start, end, symbols)
    return run_simulation(ohlcv, signal_fn, name, rebalance_every=rebalance_every) if ohlcv else _empty(name)


# ── RL Trader ─────────────────────────────────────────────────────────────────
# Monthly momentum portfolio: rank universe by 60-day Sharpe, RSI-filtered,
# equal-weight top 10 at 8% cap.

def _sharpe_score(df: pd.DataFrame) -> float | None:
    """Return 60-day rolling Sharpe for a symbol, or None if RSI filter rejects it."""
    if len(df) < 65:
        return None
    closes = df["close"]
    r = rsi(closes).iloc[-1]
    if pd.isna(r) or not (35 <= r <= 72):
        return None
    s = rolling_sharpe(closes.pct_change().dropna()).iloc[-1]
    return s if pd.notna(s) and s > 0 else None


def _rl_signal(date, ohlcv, state):
    scores = {sym: s for sym, df in ohlcv.items() if (s := _sharpe_score(df)) is not None}
    if not scores:
        return {}
    top = sorted(scores, key=scores.__getitem__, reverse=True)[:10]
    weight = min(0.08, 1.0 / len(top))
    return {sym: weight for sym in top}


def run_rl_trader(start: str, end: str, symbols=None) -> BacktestResult:
    return _run("RL Trader", _rl_signal, start, end, symbols, rebalance_every=21)


# ── Regime Trader ─────────────────────────────────────────────────────────────
# SPY volatility → regime tier → allocation % and candidate pool.
# ATR trailing stops set per position each bar.

_REGIME_TIERS: dict[str, tuple[float, int]] = {
    "low":  (0.90, 10),  # LowVolBull
    "mid":  (0.70,  8),  # MidVolCautious
    "high": (0.50,  5),  # HighVolDefensive
}


def _detect_regime(spy_closes: pd.Series) -> str:
    vol = spy_closes.pct_change().rolling(20).std().iloc[-1]
    if vol < 0.012:
        return "low"
    if vol < 0.020:
        return "mid"
    return "high"


def _momentum_candidates(ohlcv: dict, max_pos: int) -> list[str]:
    ranked = []
    for sym, df in ohlcv.items():
        if len(df) < 25:
            continue
        mom = df["close"].pct_change(20).iloc[-1]
        r = rsi(df["close"]).iloc[-1]
        if pd.notna(mom) and mom > 0 and pd.notna(r) and 30 < r < 75:
            ranked.append((sym, mom))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in ranked[:max_pos]]


def _atr_trail(df: pd.DataFrame) -> float:
    a = atr(df["high"], df["low"], df["close"]).iloc[-1]
    return float(np.clip(1.5 * a / df["close"].iloc[-1], 0.05, 0.20))


def _regime_signal(date, ohlcv, state):
    if "SPY" not in ohlcv or len(ohlcv["SPY"]) < 25:
        return {}

    regime = _detect_regime(ohlcv["SPY"]["close"])
    alloc, max_pos = _REGIME_TIERS[regime]
    state["regime"] = regime

    if regime == "high":
        candidates = [s for s in DEFENSIVE if s in ohlcv and len(ohlcv[s]) >= 25]
    else:
        candidates = _momentum_candidates(ohlcv, max_pos)
        if regime == "mid":
            defensive = [s for s in DEFENSIVE if s in ohlcv][:3]
            candidates = list(dict.fromkeys(candidates[:4] + defensive))

    if not candidates:
        return {}

    for sym in candidates:
        if sym in ohlcv and len(ohlcv[sym]) >= 15:
            state[f"trail_{sym}"] = _atr_trail(ohlcv[sym])

    weight = min(alloc / len(candidates), 0.08)
    return {sym: weight for sym in candidates}


def run_regime_trader(start: str, end: str, symbols=None) -> BacktestResult:
    return _run("Regime Trader", _regime_signal, start, end, symbols, rebalance_every=5)


# ── Claudebot ─────────────────────────────────────────────────────────────────
# 5-factor scoring per TRADING-STRATEGY.md. Each factor → 0/1/2 points.
# Enter if total ≥ 7. Max 3 new/week, max 10 positions, 10% trailing stop.

def _ytd_returns(ohlcv: dict, date: pd.Timestamp) -> dict[str, float]:
    year_start = pd.Timestamp(date.year, 1, 1, tz=date.tz)
    result = {}
    for sym, df in ohlcv.items():
        sub = df[df.index >= year_start]
        if len(sub) >= 2:
            result[sym] = sub["close"].iloc[-1] / sub["close"].iloc[0] - 1
    return result


def _factor_catalyst(closes: pd.Series, ytd: dict) -> int:
    ret = closes.pct_change(5).iloc[-1]
    if pd.isna(ret):
        return 0
    median = np.nanmedian(list(ytd.values())) / 252 * 5
    return 2 if ret > median * 1.5 else (1 if ret > 0 else 0)


def _factor_sector_rank(sym: str, ytd: dict) -> int:
    vals = sorted(ytd.values())
    val = ytd.get(sym, 0)
    rank = vals.index(val) / max(len(vals) - 1, 1) if val in vals else 0.5
    return 2 if rank >= 0.67 else (1 if rank >= 0.33 else 0)


def _factor_technical(closes: pd.Series) -> int:
    s20, price = sma(closes, 20).iloc[-1], closes.iloc[-1]
    if pd.isna(s20) or s20 == 0:
        return 0
    dist = (price - s20) / s20
    return 2 if dist <= 0.02 else (1 if dist <= 0.08 else 0)


def _factor_volume(df: pd.DataFrame) -> int:
    vols = df.get("volume", pd.Series(dtype=float))
    if len(vols) < 21 or vols.empty:
        return 1  # neutral when data is missing
    ratio = vols.iloc[-1] / (vols.rolling(20).mean().iloc[-1] + 1e-9)
    return 2 if ratio >= 1.5 else (1 if ratio >= 1.2 else 0)


def _factor_rr(df: pd.DataFrame, closes: pd.Series) -> int:
    a = atr(df["high"], df["low"], closes).iloc[-1] if "high" in df.columns else closes.std()
    if pd.isna(a) or a == 0:
        return 0
    ret = closes.pct_change(5).iloc[-1]
    rr = max(ret * 2, 0) / (1.5 * a / closes.iloc[-1] + 1e-9)
    return 2 if rr >= 2.0 else (1 if rr >= 1.5 else 0)


def _score(sym: str, df: pd.DataFrame, ytd: dict) -> int:
    if len(df) < 25:
        return 0
    closes = df["close"]
    return (
        _factor_catalyst(closes, ytd)
        + _factor_sector_rank(sym, ytd)
        + _factor_technical(closes)
        + _factor_volume(df)
        + _factor_rr(df, closes)
    )


def _claudebot_signal(date, ohlcv, state):
    week_key = date.isocalendar()[:2]
    if state.get("week") != week_key:
        state["week"] = week_key
        state["new_this_week"] = 0

    positions: set = state.get("positions", set())
    sector_losses: dict = state.get("sector_losses", {})
    ytd = _ytd_returns(ohlcv, date)

    # Score once per symbol, filter ≥ 7
    scored = {sym: s for sym, df in ohlcv.items()
              if sym not in positions and (s := _score(sym, df, ytd)) >= 7}

    target = {sym: 0.08 for sym in positions if sym in ohlcv}
    max_new = min(3 - state.get("new_this_week", 0), 10 - len(target))

    for sym in sorted(scored, key=scored.__getitem__, reverse=True)[:max_new]:
        if sector_losses.get(SECTOR_MAP.get(sym, "Unknown"), 0) >= 2:
            continue
        target[sym] = 0.08
        state["new_this_week"] = state.get("new_this_week", 0) + 1
        positions.add(sym)

    for sym in target:
        state[f"trail_{sym}"] = 0.10

    state["positions"] = set(target)
    return target


def run_claudebot(start: str, end: str, symbols=None) -> BacktestResult:
    return _run("Claudebot", _claudebot_signal, start, end, symbols, rebalance_every=1)


# ── SPY benchmark ─────────────────────────────────────────────────────────────

def run_spy_benchmark(start: str, end: str) -> BacktestResult:
    return _run("SPY B&H", lambda d, o, s: {"SPY": 0.99}, start, end, ["SPY"], rebalance_every=999)
