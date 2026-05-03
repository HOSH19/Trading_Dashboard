"""Claudebot — Claude API scoring with Alpaca News, approximating the pre-market + market-open routines."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from backtesting.engine import run_simulation
from backtesting.indicators import atr, rsi, sma
from backtesting.metrics import BacktestResult, TradeRecord
from backtesting.news import fetch_headlines
from config import SECTOR_MAP, UNIVERSE
from fetchers.market_data import load_ohlcv

CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

TRADING_STRATEGY_RUBRIC = """
Score each stock 0-10 using exactly these five factors (0, 1, or 2 points each):

1. Catalyst strength
   0 = no news / vague rumor
   1 = scheduled event (earnings, macro release, product launch scheduled)
   2 = confirmed catalyst (earnings beat, FDA approval, analyst upgrade, major contract)

2. Sector rank YTD
   0 = bottom third of the 20-symbol universe by YTD return
   1 = middle third
   2 = top third

3. Technical setup (distance from 20-day SMA)
   0 = more than 10% above 20-day SMA (extended, risky entry)
   1 = 5-10% above 20-day SMA
   2 = at or below 20-day SMA (ideal entry point)

4. Volume confirmation (today vs 20-day average volume)
   0 = below average volume
   1 = 1.0x - 1.5x average volume
   2 = more than 1.5x average volume

5. Risk/reward ratio
   0 = R:R < 1.5
   1 = R:R 1.5 - 2.0
   2 = R:R > 2.0

Only enter trades with a score >= 7. Return ONLY a valid JSON object mapping ticker to integer score.
"""


def _compute_candidate_data(date: pd.Timestamp, ohlcv: dict) -> dict[str, dict]:
    """Compute per-symbol factor data from OHLCV slices."""
    year_start = pd.Timestamp(date.year, 1, 1, tz=date.tz)
    ytd: dict[str, float] = {}
    for sym, df in ohlcv.items():
        sub = df[df.index >= year_start]
        if len(sub) >= 2:
            ytd[sym] = sub["close"].iloc[-1] / sub["close"].iloc[0] - 1

    candidates = {}
    for sym in UNIVERSE:
        df = ohlcv.get(sym)
        if df is None or len(df) < 25:
            continue
        closes = df["close"]
        price = float(closes.iloc[-1])

        sma20 = float(sma(closes, 20).iloc[-1])
        sma_dist = (price - sma20) / sma20 if sma20 else 0.0

        mom5 = float(closes.pct_change(5).iloc[-1]) if len(closes) >= 6 else 0.0

        vols = df.get("volume", pd.Series(dtype=float))
        vol_ratio = 1.0
        if len(vols) >= 21 and not vols.empty:
            avg_vol = float(vols.rolling(20).mean().iloc[-1])
            vol_ratio = float(vols.iloc[-1]) / (avg_vol + 1e-9)

        a = float(atr(df["high"], df["low"], closes).iloc[-1]) if "high" in df.columns else float(closes.std())
        rr = max(mom5 * 2, 0) / (1.5 * a / price + 1e-9) if price > 0 else 0.0

        ytd_val = ytd.get(sym, 0.0)
        ytd_vals = sorted(ytd.values())
        ytd_rank = ytd_vals.index(ytd_val) / max(len(ytd_vals) - 1, 1) if ytd_val in ytd_vals else 0.5

        candidates[sym] = {
            "price": round(price, 2),
            "sma20_dist_pct": round(sma_dist * 100, 2),
            "momentum_5d_pct": round(mom5 * 100, 2),
            "volume_ratio": round(vol_ratio, 2),
            "rr_estimate": round(rr, 2),
            "ytd_rank_pct": round(ytd_rank * 100, 1),
        }
    return candidates


def _build_prompt(date: str, candidates: dict[str, dict], headlines: dict[str, list[str]]) -> str:
    lines = [TRADING_STRATEGY_RUBRIC, f"\nDate: {date}\n\nCandidate data:\n"]
    for sym, data in candidates.items():
        news = headlines.get(sym, [])
        news_str = "; ".join(news[:3]) if news else "no news found"
        lines.append(
            f"{sym}: price=${data['price']}, SMA20_dist={data['sma20_dist_pct']:+.1f}%, "
            f"5d_momentum={data['momentum_5d_pct']:+.1f}%, vol_ratio={data['volume_ratio']:.2f}x, "
            f"R:R={data['rr_estimate']:.2f}, YTD_rank={data['ytd_rank_pct']:.0f}th_pct | "
            f"News: {news_str}"
        )
    lines.append('\nRespond with only a JSON object like {"AAPL": 7, "MSFT": 4, ...} for all symbols listed.')
    return "\n".join(lines)


def _parse_scores(text: str) -> dict[str, int]:
    match = re.search(r"\{[^}]+\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        raw = json.loads(match.group())
        return {k: int(v) for k, v in raw.items() if isinstance(v, (int, float))}
    except Exception:
        return {}


def _score_batch(date: str, candidates: dict[str, dict], headlines: dict[str, list[str]]) -> dict[str, int]:
    from anthropic import Anthropic
    client = Anthropic()
    prompt = _build_prompt(date, candidates, headlines)
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return _parse_scores(resp.content[0].text)


def _make_claudebot_signal(cache: dict, cache_path: Path):
    def _signal(date, ohlcv, state):
        week_key = date.isocalendar()[:2]
        if state.get("week") != week_key:
            state["week"] = week_key
            state["new_this_week"] = 0

        # Score weekly
        if state.get("last_scored_week") != week_key:
            state["last_scored_week"] = week_key
            date_str = date.strftime("%Y-%m-%d")
            candidates = _compute_candidate_data(date, ohlcv)

            scores: dict[str, int] = {}
            uncached = [sym for sym in candidates if f"{date_str}_{sym}" not in cache]

            if uncached:
                # Fetch news only for uncached symbols
                headlines = {sym: fetch_headlines(sym, date_str) for sym in uncached}
                uncached_candidates = {sym: candidates[sym] for sym in uncached}
                try:
                    new_scores = _score_batch(date_str, uncached_candidates, headlines)
                except Exception:
                    new_scores = {}
                for sym, score in new_scores.items():
                    cache[f"{date_str}_{sym}"] = score
                try:
                    cache_path.write_text(json.dumps(cache))
                except Exception:
                    pass

            for sym in candidates:
                key = f"{date_str}_{sym}"
                if key in cache:
                    scores[sym] = cache[key]

            state["scores"] = scores

        scores = state.get("scores", {})
        positions: set = state.get("positions", set())
        sector_losses: dict = state.get("sector_losses", {})

        qualified = {sym: s for sym, s in scores.items() if s >= 7 and sym not in positions}
        target = {sym: 0.08 for sym in positions if sym in ohlcv}
        max_new = min(3 - state.get("new_this_week", 0), 10 - len(target))

        for sym in sorted(qualified, key=qualified.__getitem__, reverse=True)[:max(max_new, 0)]:
            if sector_losses.get(SECTOR_MAP.get(sym, "Unknown"), 0) >= 2:
                continue
            target[sym] = 0.08
            state["new_this_week"] = state.get("new_this_week", 0) + 1
            positions.add(sym)

        for sym in target:
            state[f"trail_{sym}"] = 0.10

        state["positions"] = set(target)
        return target

    return _signal


def run_claudebot_api(start: str, end: str, symbols=None) -> BacktestResult:
    cache_path = CACHE_DIR / f"claudebot_{start}_{end}.json"
    cache: dict = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:
            pass

    ohlcv = load_ohlcv(start, end, symbols)
    if not ohlcv:
        return BacktestResult("Claudebot", pd.Series(dtype=float), [])

    signal_fn = _make_claudebot_signal(cache, cache_path)
    try:
        return run_simulation(ohlcv, signal_fn, "Claudebot", rebalance_every=1)
    finally:
        try:
            cache_path.write_text(json.dumps(cache))
        except Exception:
            pass
