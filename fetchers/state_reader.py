"""Read historical equity curves from SQLite state stores and Claudebot's markdown log."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd


# ── SQLite readers (RL Trader & Regime Trader) ──────────────────────────────

def read_equity_curve(db_path: Path | str) -> pd.DataFrame:
    """Returns DataFrame with columns: timestamp (datetime), equity, cash."""
    db_path = Path(db_path)
    if not db_path.exists():
        return pd.DataFrame(columns=["timestamp", "equity", "cash"])
    try:
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        # Both RL_Trader and Regime-Trader use equity_curve with similar schemas
        # RL_Trader: (timestamp, equity, cash)
        # Regime-Trader: (ts, equity, cash)
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if "equity_curve" not in tables:
            return pd.DataFrame(columns=["timestamp", "equity", "cash"])

        cols = [r[1] for r in conn.execute("PRAGMA table_info(equity_curve)").fetchall()]
        ts_col = "ts" if "ts" in cols else "timestamp"

        df = pd.read_sql(f"SELECT {ts_col} AS timestamp, equity, cash FROM equity_curve ORDER BY {ts_col}", conn)
        conn.close()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["equity"] = pd.to_numeric(df["equity"])
        df["cash"] = pd.to_numeric(df["cash"])
        return df
    except Exception:
        return pd.DataFrame(columns=["timestamp", "equity", "cash"])


def read_trade_log(db_path: Path | str) -> pd.DataFrame:
    """Returns unified trade log DataFrame."""
    db_path = Path(db_path)
    if not db_path.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if "trade_log" not in tables:
            return pd.DataFrame()

        cols = [r[1] for r in conn.execute("PRAGMA table_info(trade_log)").fetchall()]
        ts_col = "ts" if "ts" in cols else "timestamp"

        df = pd.read_sql(f"SELECT * FROM trade_log ORDER BY {ts_col}", conn)
        conn.close()
        if ts_col != "timestamp":
            df = df.rename(columns={ts_col: "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df
    except Exception:
        return pd.DataFrame()


# ── Claudebot markdown reader ────────────────────────────────────────────────

_SNAPSHOT_RE = re.compile(
    r"###\s+(\w+ \d+)\s+—\s+EOD Snapshot.*?\n"
    r"\*\*Portfolio:\*\*\s+\$([\d,]+\.?\d*)\s*\|.*?"
    r"\*\*Cash:\*\*\s+\$([\d,]+\.?\d*)\s*\(",
    re.DOTALL,
)

_TRADE_RE = re.compile(
    r"###\s+(\d{4}-\d{2}-\d{2})\s+—\s+(BUY|SELL)\s+(\w+).*?"
    r"(?:Entry|Exit):\s+\$([\d.]+).*?"
    r"(?:Shares|shares):\s+([\d.]+)",
    re.DOTALL | re.IGNORECASE,
)


def _parse_month_day(raw: str, year: int) -> str:
    """Convert 'Apr 28' to '2026-04-28' using current year as fallback."""
    try:
        dt = pd.to_datetime(f"{raw} {year}", format="%b %d %Y")
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return ""


def read_claudebot_equity(trade_log_path: Path | str) -> pd.DataFrame:
    trade_log_path = Path(trade_log_path)
    if not trade_log_path.exists():
        return pd.DataFrame(columns=["timestamp", "equity", "cash"])

    text = trade_log_path.read_text()
    rows = []
    # Infer year from file modification time as best guess
    year = trade_log_path.stat().st_mtime
    import datetime
    year = datetime.datetime.fromtimestamp(year).year

    for m in _SNAPSHOT_RE.finditer(text):
        date_str = _parse_month_day(m.group(1), year)
        if not date_str:
            continue
        equity = float(m.group(2).replace(",", ""))
        cash = float(m.group(3).replace(",", ""))
        rows.append({"timestamp": pd.Timestamp(date_str, tz="UTC"), "equity": equity, "cash": cash})

    if not rows:
        return pd.DataFrame(columns=["timestamp", "equity", "cash"])
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def read_claudebot_trades(trade_log_path: Path | str) -> pd.DataFrame:
    trade_log_path = Path(trade_log_path)
    if not trade_log_path.exists():
        return pd.DataFrame()

    text = trade_log_path.read_text()
    rows = []
    for m in _TRADE_RE.finditer(text):
        rows.append({
            "timestamp": pd.Timestamp(m.group(1), tz="UTC"),
            "side": m.group(2).lower(),
            "symbol": m.group(3),
            "price": float(m.group(4)),
            "qty": float(m.group(5)),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()
