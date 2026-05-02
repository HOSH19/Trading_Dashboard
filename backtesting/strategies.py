"""
Strategy proxy implementations for backtesting.

Each strategy approximates its live counterpart's logic using only
historical OHLCV data — no trained models, no external calls.

RL Trader     → momentum-weighted allocation (Sharpe-ranked, RSI-filtered)
Regime Trader → SPY-volatility regime detection with tier-based allocation
Claudebot     → 5-factor scoring (per TRADING-STRATEGY.md), swing-trade entries
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting.engine import atr, rsi, rolling_sharpe, run_simulation, sma, BacktestResult
from config import DEFENSIVE
from fetchers.market_data import load_ohlcv


# ── RL Trader proxy: Sharpe-weighted momentum portfolio ─────────────────────
#
# Logic: each month, rank all universe stocks by their 60-day rolling Sharpe.
# Pick top N with RSI in [35, 72] (not overbought/oversold).
# Allocate equal weight capped at 8% per position. Rebalance monthly.

def _rl_signal(date: pd.Timestamp, ohlcv: dict[str, pd.DataFrame], state: dict) -> dict[str, float]:
    scores: dict[str, float] = {}
    for sym, df in ohlcv.items():
        if len(df) < 65:
            continue
        closes = df["close"]
        rets = closes.pct_change().dropna()
        r = rsi(closes).iloc[-1]
        if pd.isna(r) or not (35 <= r <= 72):
            continue
        sharpe = rolling_sharpe(rets).iloc[-1]
        if pd.isna(sharpe) or sharpe <= 0:
            continue
        scores[sym] = sharpe

    if not scores:
        return {}

    ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:10]
    weight = min(0.08, 1.0 / len(ranked))
    return {sym: weight for sym in ranked}


def run_rl_trader(start: str, end: str, symbols: list[str] | None = None) -> BacktestResult:
    ohlcv = load_ohlcv(start, end, symbols)
    if not ohlcv:
        from backtesting.engine import BacktestResult
        return BacktestResult("RL Trader", pd.Series(dtype=float), [])

    # Rebalance every ~21 trading days (monthly)
    return run_simulation(ohlcv, _rl_signal, "RL Trader", rebalance_every=21)


# ── Regime Trader proxy: volatility-based regime allocation ─────────────────
#
# Regime detection via SPY 20-day realized volatility:
#   LowVolBull  (vol < 0.012)  → 90% equity, top momentum stocks, 1× leverage
#   MidVolCautious (0.012–0.02) → 70% equity, mixed momentum + defensive
#   HighVolDefensive (> 0.02)   → 50% equity, defensive names only
#
# ATR trailing stop: 1.5× ATR per position, updated nightly.

def _regime_signal(date: pd.Timestamp, ohlcv: dict[str, pd.DataFrame], state: dict) -> dict[str, float]:
    if "SPY" not in ohlcv or len(ohlcv["SPY"]) < 25:
        return {}

    spy = ohlcv["SPY"]["close"]
    vol = spy.pct_change().rolling(20).std().iloc[-1]

    if vol < 0.012:
        regime, alloc, max_pos = "low", 0.90, 10
    elif vol < 0.020:
        regime, alloc, max_pos = "mid", 0.70, 8
    else:
        regime, alloc, max_pos = "high", 0.50, 5

    state["regime"] = regime

    if regime == "high":
        candidates = [s for s in DEFENSIVE if s in ohlcv and len(ohlcv[s]) >= 25]
    else:
        # Rank by 20-day momentum, RSI filter
        candidates = []
        for sym, df in ohlcv.items():
            if len(df) < 25:
                continue
            mom = df["close"].pct_change(20).iloc[-1]
            r = rsi(df["close"]).iloc[-1]
            if pd.notna(mom) and pd.notna(r) and mom > 0 and 30 < r < 75:
                candidates.append((sym, mom))
        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = [s for s, _ in candidates[:max_pos]]
        if regime == "mid":
            # Blend: half momentum, half defensive
            defensive = [s for s in DEFENSIVE if s in ohlcv][:3]
            candidates = list(dict.fromkeys(candidates[:4] + defensive))

    if not candidates:
        return {}

    # Set per-symbol ATR trailing stops
    for sym in candidates:
        if sym in ohlcv and len(ohlcv[sym]) >= 15:
            df = ohlcv[sym]
            a = atr(df["high"], df["low"], df["close"]).iloc[-1]
            p = df["close"].iloc[-1]
            trail = min(max(1.5 * a / p, 0.05), 0.20)  # clamp 5%–20%
            state[f"trail_{sym}"] = trail

    weight = min(alloc / len(candidates), 0.08)
    return {sym: weight for sym in candidates}


def run_regime_trader(start: str, end: str, symbols: list[str] | None = None) -> BacktestResult:
    ohlcv = load_ohlcv(start, end, symbols)
    if not ohlcv:
        return BacktestResult("Regime Trader", pd.Series(dtype=float), [])

    # Rebalance every ~5 trading days (weekly)
    return run_simulation(ohlcv, _regime_signal, "Regime Trader", rebalance_every=5)


# ── Claudebot proxy: 5-factor scoring → swing trade entries ─────────────────
#
# Factors (per TRADING-STRATEGY.md):
#   1. Catalyst strength   → proxy: 5-day return vs universe median
#   2. Sector rank YTD     → proxy: YTD return percentile rank in universe
#   3. Technical setup     → price relative to 20-day SMA
#   4. Volume confirmation → volume vs 20-day avg volume
#   5. R:R ratio           → ATR-based: momentum / (1.5 × ATR / price)
#
# Rules:
#   - Min score 7/10 to enter
#   - Max 10 positions, max 8% each
#   - Max 3 new positions per week (tracked via state)
#   - 10% trailing stop (tightens to 7% at +15%, 5% at +20%)
#   - -7% hard stop (stop-loss exit)
#   - Exit sector after 2 consecutive losses in that sector

def _score_candidate(sym: str, df: pd.DataFrame, universe_ytd: dict[str, float]) -> int:
    if len(df) < 25:
        return 0

    closes = df["close"]
    volumes = df["volume"] if "volume" in df.columns else pd.Series(dtype=float)

    # Factor 1: Catalyst proxy — 5-day return vs median
    ret_5d = closes.pct_change(5).iloc[-1]
    median_5d = np.nanmedian(list(universe_ytd.values())) / 252 * 5  # rough scaling
    if pd.isna(ret_5d):
        f1 = 0
    elif ret_5d > median_5d * 1.5:
        f1 = 2
    elif ret_5d > 0:
        f1 = 1
    else:
        f1 = 0

    # Factor 2: Sector rank YTD
    ytd = universe_ytd.get(sym, 0)
    all_ytd = sorted(universe_ytd.values())
    n = len(all_ytd)
    rank_pct = all_ytd.index(ytd) / max(n - 1, 1) if ytd in all_ytd else 0.5
    if rank_pct >= 0.67:
        f2 = 2
    elif rank_pct >= 0.33:
        f2 = 1
    else:
        f2 = 0

    # Factor 3: Technical setup (distance from 20d SMA)
    s20 = sma(closes, 20).iloc[-1]
    price = closes.iloc[-1]
    if pd.isna(s20) or s20 == 0:
        f3 = 0
    else:
        dist = (price - s20) / s20
        if dist <= 0.02:
            f3 = 2  # at or below SMA (potential breakout)
        elif dist <= 0.08:
            f3 = 1  # slightly above
        else:
            f3 = 0  # too extended

    # Factor 4: Volume confirmation
    if len(volumes) >= 21 and not volumes.empty:
        avg_vol = volumes.rolling(20).mean().iloc[-1]
        today_vol = volumes.iloc[-1]
        ratio = today_vol / (avg_vol + 1e-9)
        if ratio >= 1.5:
            f4 = 2
        elif ratio >= 1.2:
            f4 = 1
        else:
            f4 = 0
    else:
        f4 = 1  # assume neutral if volume data missing

    # Factor 5: R:R ratio
    a = atr(df["high"], df["low"], closes).iloc[-1] if "high" in df.columns else closes.std()
    if pd.isna(a) or a == 0:
        f5 = 0
    else:
        risk_pct = (1.5 * a) / price
        target_pct = max(ret_5d * 2, 0)  # extrapolate momentum
        rr = target_pct / (risk_pct + 1e-9)
        if rr >= 2.0:
            f5 = 2
        elif rr >= 1.5:
            f5 = 1
        else:
            f5 = 0

    return f1 + f2 + f3 + f4 + f5


def _claudebot_signal(date: pd.Timestamp, ohlcv: dict[str, pd.DataFrame], state: dict) -> dict[str, float]:
    # Track trades per week
    week_key = date.isocalendar()[:2]
    if state.get("week") != week_key:
        state["week"] = week_key
        state["new_this_week"] = 0

    current_positions = state.get("positions", set())
    sector_losses = state.get("sector_losses", {})  # sector -> consecutive losses

    # Compute YTD returns for all symbols
    ytd_returns: dict[str, float] = {}
    year_start = pd.Timestamp(date.year, 1, 1, tz=date.tz)
    for sym, df in ohlcv.items():
        after_start = df[df.index >= year_start]
        if len(after_start) < 2:
            continue
        ytd_returns[sym] = (after_start["close"].iloc[-1] / after_start["close"].iloc[0]) - 1

    # Score all candidates
    scored = {}
    for sym, df in ohlcv.items():
        if sym in current_positions:
            continue  # already holding
        score = _score_candidate(sym, df, ytd_returns)
        if score >= 7:
            scored[sym] = score

    # Current positions continue (managed by trailing stops in engine)
    target = {sym: 0.08 for sym in current_positions if sym in ohlcv}

    # Add new entries up to limits
    max_new = min(3 - state.get("new_this_week", 0), 10 - len(target))
    new_entries = sorted(scored, key=scored.__getitem__, reverse=True)[:max_new]

    from config import SECTOR_MAP
    for sym in new_entries:
        sector = SECTOR_MAP.get(sym, "Unknown")
        # Skip sectors with 2+ consecutive losses
        if sector_losses.get(sector, 0) >= 2:
            continue
        target[sym] = 0.08
        state["new_this_week"] = state.get("new_this_week", 0) + 1
        state.setdefault("positions", set()).add(sym)

    # Claudebot uses 10% trailing stop as default, tightens as gains accumulate
    # The engine handles trailing stops; we set per-symbol trail in state
    for sym in list(target):
        if sym in ohlcv:
            state[f"trail_{sym}"] = 0.10  # engine updates per gains

    state["positions"] = set(target.keys())
    return target


def run_claudebot(start: str, end: str, symbols: list[str] | None = None) -> BacktestResult:
    ohlcv = load_ohlcv(start, end, symbols)
    if not ohlcv:
        return BacktestResult("Claudebot", pd.Series(dtype=float), [])

    # Daily evaluation (claudebot runs multiple sessions per day)
    return run_simulation(ohlcv, _claudebot_signal, "Claudebot", rebalance_every=1)


# ── SPY Buy-and-Hold benchmark ───────────────────────────────────────────────

def run_spy_benchmark(start: str, end: str) -> BacktestResult:
    ohlcv = load_ohlcv(start, end, ["SPY"])
    if not ohlcv:
        return BacktestResult("SPY B&H", pd.Series(dtype=float), [])

    def _spy_signal(date, ohlcv, state):
        return {"SPY": 0.99}

    return run_simulation(ohlcv, _spy_signal, "SPY B&H", rebalance_every=999)
