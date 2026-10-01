"""Momentum and Trend-Deviation Derived Features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .registry import register_feature


@register_feature(
    name="price_momentum",
    description="Rolling percentage return over specified lookback window.",
    required_columns=("close",),
)
def compute_price_momentum(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """Simple rate of change momentum."""
    return df["close"].pct_change(window).fillna(0.0)


@register_feature(
    name="rsi",
    description="Relative Strength Index (RSI) between [0, 100].",
    required_columns=("close",),
)
def compute_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Pure pandas/numpy RSI calculation without external TA-Lib dependencies."""
    delta = df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()

    # Wilder's smoothing
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


@register_feature(
    name="zscore_price",
    description="Price distance from rolling mean in units of rolling standard deviation.",
    required_columns=("close",),
)
def compute_zscore_price(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Bollinger-band style normalized price deviation."""
    roll_mean = df["close"].rolling(window).mean()
    roll_std = df["close"].rolling(window).std().replace(0, np.nan)
    return ((df["close"] - roll_mean) / roll_std).fillna(0.0)


@register_feature(
    name="vwap_distance",
    description="Percentage deviation of close price from rolling Volume Weighted Average Price.",
    required_columns=("close", "buy_volume", "sell_volume"),
)
def compute_vwap_distance(df: pd.DataFrame, window: int = 60) -> pd.Series:
    """Relative basis points distance from rolling VWAP."""
    total_volume = df["buy_volume"] + df["sell_volume"]
    cum_pv = (df["close"] * total_volume).rolling(window).sum()
    cum_vol = total_volume.rolling(window).sum()
    vwap = cum_pv / cum_vol.replace(0, np.nan)
    dist = (df["close"] - vwap) / vwap.replace(0, np.nan)
    return dist.fillna(0.0)
