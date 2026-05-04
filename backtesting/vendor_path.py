"""Shared vendor sys.path management for strategies that load vendored packages."""

from __future__ import annotations

import sys
from pathlib import Path

_BACKTESTING = Path(__file__).parent
REGIME_PATH = _BACKTESTING / "agents" / "regime"
RL_PATH = _BACKTESTING / "agents" / "rl"


def setup_vendor_paths(*paths: Path) -> None:
    """Remove then re-insert vendor paths at front of sys.path, evict cached 'data' modules."""
    str_paths = [str(p) for p in paths]
    for p in str_paths:
        while p in sys.path:
            sys.path.remove(p)
    for p in reversed(str_paths):
        sys.path.insert(0, p)
    for key in list(sys.modules.keys()):
        if key == "data" or key.startswith("data."):
            del sys.modules[key]


def setup_regime_paths() -> None:
    """Set sys.path so regime/ is first — its 'data' package must resolve before rl/data/."""
    setup_vendor_paths(REGIME_PATH, RL_PATH)


def setup_rl_paths() -> None:
    """Set sys.path so rl/ is first — its 'data' package must resolve before regime/data/."""
    setup_vendor_paths(RL_PATH, REGIME_PATH)
