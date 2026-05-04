"""
Feature engineering for RL observation space.

All features are:
- Causal: computed only from information available at or before each bar.
- Z-score normalized: rolling 252-bar window, scale-invariant across regimes.
- NaN-safe: rows with insufficient history are dropped before training.
"""

import numpy as np
import pandas as pd


_ZSCORE_WINDOW = 252


def compute_features(bars: pd.DataFrame, macro: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Build the full feature matrix for one symbol.

    Args:
        bars:  OHLCV DataFrame (lowercase columns) indexed by date.
        macro: Optional macro DataFrame with columns [vix, yield_spread, credit_proxy].

    Returns:
        DataFrame of z-scored features, same index as bars (NaN rows at start dropped).
    """
    df = pd.DataFrame(index=bars.index)

    close = bars["close"]
    high = bars["high"]
    low = bars["low"]
    volume = bars["volume"]

    log_close = np.log(close)

    # --- Returns ---
    df["log_ret_1"] = log_close.diff(1)
    df["log_ret_5"] = log_close.diff(5)
    df["log_ret_20"] = log_close.diff(20)

    # --- Realized volatility ---
    df["realized_vol_20"] = df["log_ret_1"].rolling(20).std() * np.sqrt(252)
    df["vol_ratio_5_20"] = df["log_ret_1"].rolling(5).std() / (
        df["log_ret_1"].rolling(20).std() + 1e-8
    )

    # --- Momentum ---
    df["rsi_14"] = _rsi(close, 14)
    df["roc_10"] = close.pct_change(10)
    df["roc_20"] = close.pct_change(20)
    df["macd_hist"] = _macd_hist(close)

    # --- Trend ---
    df["adx_14"] = _adx(high, low, close, 14)
    sma50 = close.rolling(50).mean()
    df["sma50_slope"] = sma50.diff(5) / (sma50 + 1e-8)
    sma200 = close.rolling(200).mean()
    df["dist_sma200"] = (close - sma200) / (sma200 + 1e-8)

    # --- Volume ---
    vol_sma50 = volume.rolling(50).mean()
    df["vol_norm"] = (volume - vol_sma50) / (vol_sma50 + 1e-8)
    df["vol_trend"] = vol_sma50.diff(5) / (vol_sma50 + 1e-8)

    # --- Risk ---
    atr = _atr(high, low, close, 14)
    df["norm_atr"] = atr / (close + 1e-8)
    df["overnight_gap"] = (bars["open"] - close.shift(1)) / (close.shift(1) + 1e-8)

    # --- Macro (optional) ---
    if macro is not None:
        macro_aligned = macro.reindex(bars.index, method="ffill")
        df["vix_raw"] = macro_aligned["vix"]
        df["yield_spread_raw"] = macro_aligned["yield_spread"]
        df["credit_proxy_raw"] = macro_aligned["credit_proxy"]
        for col in ["vix_raw", "yield_spread_raw", "credit_proxy_raw"]:
            df[col.replace("_raw", "_zscore")] = _zscore(df[col])
        df.drop(columns=["vix_raw", "yield_spread_raw", "credit_proxy_raw"], inplace=True)

    # --- Z-score normalize all features ---
    feat_cols = [c for c in df.columns if not c.endswith("_zscore")]
    for col in feat_cols:
        df[col] = _zscore(df[col])

    return df.fillna(0.0)


def feature_columns(use_macro: bool = True) -> list:
    """Return the ordered list of feature column names."""
    cols = [
        "log_ret_1", "log_ret_5", "log_ret_20",
        "realized_vol_20", "vol_ratio_5_20",
        "rsi_14", "roc_10", "roc_20", "macd_hist",
        "adx_14", "sma50_slope", "dist_sma200",
        "vol_norm", "vol_trend",
        "norm_atr", "overnight_gap",
    ]
    if use_macro:
        cols += ["vix_zscore", "yield_spread_zscore", "credit_proxy_zscore"]
    return cols


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------

def _zscore(series: pd.Series, window: int = _ZSCORE_WINDOW) -> pd.Series:
    mean = series.rolling(window, min_periods=window // 2).mean()
    std = series.rolling(window, min_periods=window // 2).std()
    return (series - mean) / (std + 1e-8)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-8)
    return 100 - 100 / (1 + rs)


def _macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    atr = _atr(high, low, close, period)
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-8)
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-8)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)
    return dx.ewm(span=period, adjust=False).mean()
