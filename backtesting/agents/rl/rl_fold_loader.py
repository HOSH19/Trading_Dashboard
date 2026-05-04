"""RL fold checkpoint discovery and date-range utilities."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

VENDOR = Path(__file__).parent
_DEFAULT_CHECKPOINT_DIR = VENDOR / "checkpoints" / "a2c_20sym"
_date_range_cache: tuple[pd.Timestamp, pd.Timestamp] | None = None


def checkpoint_dir() -> Path:
    return Path(os.environ.get("RL_CHECKPOINT_DIR", str(_DEFAULT_CHECKPOINT_DIR)))


def load_fold_dates() -> dict[int, tuple[pd.Timestamp, pd.Timestamp]]:
    """Load fold index → (test_start, test_end) from fold_dates.json.

    fold_dates.json is the single source of truth for which fold index maps to
    which test window, avoiding fragile date reconstruction from data.
    """
    path = checkpoint_dir() / "fold_dates.json"
    if not path.exists():
        raise FileNotFoundError(
            f"fold_dates.json not found at {path}. "
            "Re-run the checkpoint export script to regenerate it."
        )
    raw = json.loads(path.read_text())
    return {
        int(k): (pd.Timestamp(v["test_start"]), pd.Timestamp(v["test_end"]))
        for k, v in raw.items()
        if (checkpoint_dir() / f"fold_{k}" / "best_model.zip").exists()
    }


def rl_date_range() -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (min_date, max_date) covered by the vendored fold checkpoints."""
    global _date_range_cache
    if _date_range_cache is not None:
        return _date_range_cache
    fold_dates = load_fold_dates()
    if not fold_dates:
        raise RuntimeError("No fold checkpoints found")
    _date_range_cache = (fold_dates[min(fold_dates)][0], fold_dates[max(fold_dates)][1])
    return _date_range_cache


def relevant_folds(
    fold_dates: dict, start_ts: pd.Timestamp, end_ts: pd.Timestamp
) -> dict[int, tuple[pd.Timestamp, pd.Timestamp]]:
    """Filter fold_dates to only folds whose test window overlaps [start_ts, end_ts]."""
    return {
        idx: (ts, te)
        for idx, (ts, te) in fold_dates.items()
        if te >= start_ts and ts <= end_ts
    }
