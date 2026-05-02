import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Streamlit Community Cloud injects secrets into st.secrets, not os.environ.
# Pull them into os.environ once so the rest of the code is environment-agnostic.
try:
    import streamlit as st
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:
    pass

BASE = Path(__file__).parent

UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "AVGO", "TSM",
    "JPM", "GS", "V", "MA",
    "XOM", "CVX",
    "UNH", "JNJ",
    "SPY", "QQQ",
]

SECTOR_MAP = {
    "AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "AMZN": "Tech",
    "NVDA": "Tech", "META": "Tech", "TSLA": "Tech", "AMD": "Tech",
    "AVGO": "Tech", "TSM": "Tech",
    "JPM": "Finance", "GS": "Finance", "V": "Finance", "MA": "Finance",
    "XOM": "Energy", "CVX": "Energy",
    "UNH": "Healthcare", "JNJ": "Healthcare",
    "SPY": "ETF", "QQQ": "ETF",
}

DEFENSIVE = ["XOM", "CVX", "UNH", "JNJ", "SPY"]


@dataclass
class StrategyConfig:
    name: str
    short_name: str
    color: str
    env_prefix: str
    db_path: Path | None
    trade_log_path: Path | None
    description: str


STRATEGIES: list[StrategyConfig] = [
    StrategyConfig(
        name="RL Trader",
        short_name="rl",
        color="#4C72B0",
        env_prefix="RL",
        db_path=BASE / "../RL_Trader/state.db",
        trade_log_path=None,
        description="Deep RL (PPO) — momentum-weighted multi-asset allocation, weekly rebalance",
    ),
    StrategyConfig(
        name="Regime Trader",
        short_name="regime",
        color="#DD8452",
        env_prefix="REGIME",
        db_path=BASE / "../regime-trader/state.db",
        trade_log_path=None,
        description="HMM regime detection — allocation shifts between bull/cautious/defensive tiers",
    ),
    StrategyConfig(
        name="Claudebot",
        short_name="claude",
        color="#55A868",
        env_prefix="CLAUDE",
        db_path=None,
        trade_log_path=BASE / "../claudebot/memory/TRADE-LOG.md",
        description="AI swing trader — 5-factor scoring, trailing stops, max 3 trades/week",
    ),
]


def alpaca_keys(prefix: str) -> tuple[str, str]:
    key = os.getenv(f"{prefix}_ALPACA_API_KEY", "")
    secret = os.getenv(f"{prefix}_ALPACA_SECRET_KEY", "")
    return key, secret
