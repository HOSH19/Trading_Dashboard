"""ClaudebotSignal — weekly Claude-scored signal callable for the backtest engine."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from backtesting.agents.claudebot.claudebot_data import fetch_scores
from backtesting.agents.claudebot.claudebot_features import compute_candidates
from config import SECTOR_MAP

SCORE_THRESHOLD = 7
MAX_NEW_PER_WEEK = 3
MAX_POSITIONS = 10
POSITION_WEIGHT = 0.08
TRAIL_PCT = 0.10


class ClaudebotSignal:
    """Callable signal function for the Claudebot strategy, carrying its own cache."""

    def __init__(self, cache: dict, cache_path: Path) -> None:
        self._cache = cache
        self._cache_path = cache_path

    def _fetch_and_cache(self, date_str: str, candidates: dict) -> None:
        uncached = {sym: candidates[sym] for sym in candidates if f"{date_str}_{sym}" not in self._cache}
        if not uncached:
            return
        try:
            scores = fetch_scores(date_str, uncached)
        except Exception:
            scores = {}
        self._cache.update({f"{date_str}_{sym}": s for sym, s in scores.items()})
        try:
            self._cache_path.write_text(json.dumps(self._cache))
        except OSError:
            pass

    def _refresh_scores(self, date: pd.Timestamp, ohlcv: dict, state: dict) -> None:
        date_str = date.strftime("%Y-%m-%d")
        candidates = compute_candidates(date, ohlcv)
        self._fetch_and_cache(date_str, candidates)
        state["scores"] = {
            sym: self._cache[f"{date_str}_{sym}"]
            for sym in candidates
            if f"{date_str}_{sym}" in self._cache
        }

    def _select_entries(self, scores: dict, positions: set, sector_losses: dict, budget: int) -> list[str]:
        qualified = {sym: s for sym, s in scores.items() if s >= SCORE_THRESHOLD and sym not in positions}
        return [
            sym
            for sym in sorted(qualified, key=qualified.__getitem__, reverse=True)[:max(budget, 0)]
            if sector_losses.get(SECTOR_MAP.get(sym, "Unknown"), 0) < 2
        ]

    def __call__(self, date: pd.Timestamp, ohlcv: dict, state: dict) -> dict[str, float]:
        week_key = date.isocalendar()[:2]
        if state.get("week") != week_key:
            state["week"] = week_key
            state["new_this_week"] = 0

        if state.get("last_scored_week") != week_key:
            state["last_scored_week"] = week_key
            self._refresh_scores(date, ohlcv, state)

        scores = state.get("scores", {})
        positions: set = state.get("positions", set())
        budget = min(MAX_NEW_PER_WEEK - state.get("new_this_week", 0), MAX_POSITIONS - len(positions))

        target = {sym: POSITION_WEIGHT for sym in positions if sym in ohlcv}
        for sym in self._select_entries(scores, positions, state.get("sector_losses", {}), budget):
            target[sym] = POSITION_WEIGHT
            state["new_this_week"] = state.get("new_this_week", 0) + 1
            positions.add(sym)

        for sym in target:
            state[f"trail_{sym}"] = TRAIL_PCT
        state["positions"] = set(target)
        return target
