"""Volatility and Dispersion Derived Features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .registry import register_feature


@register_feature(
    name="realized_vol",
    description="Rolling annualized standard deviation of log returns.",
    required_columns=("close",),
)
def compute_realized_volatility(df: pd.DataFrame, window: int = 20, periods_per_year: int = 252 * 1440) -> pd.Series:
    """Realized volatility of close price returns."""
    log_returns = np.log(df["close"] / df["close"].shift(1))
    rolling_std = log_returns.rolling(window).std()
    return (rolling_std * np.sqrt(periods_per_year)).fillna(0.0)


@register_feature(
    name="parkinson_vol",
    description="Parkinson extreme-value volatility estimator using High and Low.",
    required_columns=("high", "low"),
)
def compute_parkinson_volatility(df: pd.DataFrame, window: int = 20, periods_per_year: int = 252 * 1440) -> pd.Series:
    """Parkinson (1980) volatility estimator: 5x more efficient than close-to-close."""
    log_hl = np.log(df["high"] / df["low"]) ** 2
    factor = 1.0 / (4.0 * np.log(2.0))
    rolling_var = (factor * log_hl).rolling(window).mean()
    return (np.sqrt(rolling_var * periods_per_year)).fillna(0.0)


@register_feature(
    name="garman_klass_vol",
    description="Garman-Klass volatility estimator using Open, High, Low, and Close.",
    required_columns=("open", "high", "low", "close"),
)
def compute_garman_klass_volatility(
    df: pd.DataFrame, window: int = 20, periods_per_year: int = 252 * 1440
) -> pd.Series:
    """Garman-Klass (1980) volatility estimator."""
    log_hl = np.log(df["high"] / df["low"]) ** 2
    log_co = np.log(df["close"] / df["open"]) ** 2
    gk = 0.5 * log_hl - (2.0 * np.log(2.0) - 1.0) * log_co
    rolling_var = gk.rolling(window).mean()
    return (np.sqrt(rolling_var.clip(lower=0.0) * periods_per_year)).fillna(0.0)


@register_feature(
    name="atr_pct",
    description="Average True Range (ATR) as percentage of current close price.",
    required_columns=("high", "low", "close"),
)
def compute_atr_pct(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Normalized True Range indicator."""
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window).mean()
    return (atr / df["close"]).fillna(0.0)
