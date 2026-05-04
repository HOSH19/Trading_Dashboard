"""Per-symbol technical feature extraction for Claudebot candidate scoring."""

from __future__ import annotations

import pandas as pd

from backtesting.indicators import atr, sma
from config import UNIVERSE


def _ytd_returns(ohlcv: dict, year_start: pd.Timestamp) -> dict[str, float]:
    result = {}
    for sym, df in ohlcv.items():
        sub = df[df.index >= year_start]
        if len(sub) >= 2:
            result[sym] = sub["close"].iloc[-1] / sub["close"].iloc[0] - 1
    return result


def _symbol_features(df: pd.DataFrame, ytd_rank: float) -> dict:
    """Compute the six factor values used by the Claude scoring rubric for one symbol."""
    closes = df["close"]
    price = float(closes.iloc[-1])
    sma20 = float(sma(closes, 20).iloc[-1])
    sma_dist = (price - sma20) / sma20 if sma20 else 0.0
    mom5 = float(closes.pct_change(5).iloc[-1]) if len(closes) >= 6 else 0.0

    vols = df.get("volume", pd.Series(dtype=float))
    avg_vol = float(vols.rolling(20).mean().iloc[-1]) if len(vols) >= 21 and not vols.empty else 1.0
    vol_ratio = float(vols.iloc[-1]) / (avg_vol + 1e-9) if not vols.empty else 1.0

    a = float(atr(df["high"], df["low"], closes).iloc[-1]) if "high" in df.columns else float(closes.std())
    rr = max(mom5 * 2, 0) / (1.5 * a / price + 1e-9) if price > 0 else 0.0

    return {
        "price": round(price, 2),
        "sma20_dist_pct": round(sma_dist * 100, 2),
        "momentum_5d_pct": round(mom5 * 100, 2),
        "volume_ratio": round(vol_ratio, 2),
        "rr_estimate": round(rr, 2),
        "ytd_rank_pct": round(ytd_rank * 100, 1),
    }


def compute_candidates(date: pd.Timestamp, ohlcv: dict) -> dict[str, dict]:
    """Build the full candidate feature dict for all UNIVERSE symbols on a given date."""
    year_start = pd.Timestamp(date.year, 1, 1, tz=date.tz)
    ytd = _ytd_returns(ohlcv, year_start)
    ytd_vals = sorted(ytd.values())

    candidates = {}
    for sym in UNIVERSE:
        df = ohlcv.get(sym)
        if df is None or len(df) < 25:
            continue
        ytd_val = ytd.get(sym, 0.0)
        ytd_rank = ytd_vals.index(ytd_val) / max(len(ytd_vals) - 1, 1) if ytd_val in ytd_vals else 0.5
        candidates[sym] = _symbol_features(df, ytd_rank)
    return candidates
