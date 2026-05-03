"""Walk-forward episode window builder."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class WalkForwardFold:
    """One train/test split for walk-forward evaluation."""
    fold_idx: int
    train_bars: Dict[str, pd.DataFrame]
    test_bars: Dict[str, pd.DataFrame]
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def build_folds(
    bars_by_symbol: Dict[str, pd.DataFrame],
    train_window: int = 252,
    test_window: int = 126,
    step_size: int = 126,
) -> List[WalkForwardFold]:
    """
    Build walk-forward train/test folds over the common date range.

    Train and test windows never overlap. All symbols are sliced on the
    same date index (intersection of available dates).

    Args:
        bars_by_symbol: {symbol: OHLCV DataFrame}
        train_window:   Number of bars in each training window.
        test_window:    Number of bars in each test window.
        step_size:      Bars to advance between folds.

    Returns:
        List of WalkForwardFold (train then test, no overlap).
    """
    common_index = _common_index(bars_by_symbol)
    total = len(common_index)
    folds = []
    fold_idx = 0
    start = 0

    while start + train_window + test_window <= total:
        train_end_pos = start + train_window
        test_end_pos = min(train_end_pos + test_window, total)

        train_idx = common_index[start:train_end_pos]
        test_idx = common_index[train_end_pos:test_end_pos]

        train_bars = {sym: df.loc[train_idx] for sym, df in bars_by_symbol.items()}
        test_bars = {sym: df.loc[test_idx] for sym, df in bars_by_symbol.items()}

        folds.append(WalkForwardFold(
            fold_idx=fold_idx,
            train_bars=train_bars,
            test_bars=test_bars,
            train_start=train_idx[0],
            train_end=train_idx[-1],
            test_start=test_idx[0],
            test_end=test_idx[-1],
        ))
        fold_idx += 1
        start += step_size

    return folds


def _common_index(bars_by_symbol: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """Intersection of all symbol date indices."""
    idx = None
    for df in bars_by_symbol.values():
        if idx is None:
            idx = df.index
        else:
            idx = idx.intersection(df.index)
    return idx.sort_values()
