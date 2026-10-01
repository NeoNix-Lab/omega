"""Order Flow and Microstructural Derived Features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .registry import register_feature


@register_feature(
    name="cvd_zscore",
    description="Cumulative Volume Delta (CVD) normalized Z-score over rolling window.",
    required_columns=("delta",),
)
def compute_cvd_zscore(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Rolling sum of delta divided by its rolling standard deviation."""
    roll_sum = df["delta"].rolling(window).sum()
    roll_std = df["delta"].rolling(window).std().replace(0, np.nan)
    return (roll_sum / roll_std).fillna(0.0)


@register_feature(
    name="imbalance_ratio",
    description="Volume Imbalance Ratio: (BuyVol - SellVol) / (BuyVol + SellVol).",
    required_columns=("buy_volume", "sell_volume"),
)
def compute_imbalance_ratio(df: pd.DataFrame, window: int = 1) -> pd.Series:
    """Normalized aggression ratio between [-1, +1]."""
    buy = df["buy_volume"].rolling(window).sum()
    sell = df["sell_volume"].rolling(window).sum()
    total = buy + sell
    ratio = (buy - sell) / total.replace(0, np.nan)
    return ratio.fillna(0.0)


@register_feature(
    name="absorption_ratio",
    description="Order flow absorption: high delta volume with small price displacement.",
    required_columns=("close", "delta"),
)
def compute_absorption_ratio(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """Ratio of absolute delta to price displacement (indicates resting limit orders)."""
    abs_delta = df["delta"].abs().rolling(window).sum()
    price_disp = df["close"].diff(window).abs().replace(0, np.nan)
    # Normalized absorption indicator
    return (abs_delta / (price_disp * 1000.0 + 1e-4)).fillna(0.0)


@register_feature(
    name="aggressor_pressure",
    description="Exponential moving average ratio of buy vs sell volume.",
    required_columns=("buy_volume", "sell_volume"),
)
def compute_aggressor_pressure(df: pd.DataFrame, span: int = 14) -> pd.Series:
    """EMA buy / EMA sell pressure metric centered at zero."""
    ema_buy = df["buy_volume"].ewm(span=span).mean()
    ema_sell = df["sell_volume"].ewm(span=span).mean()
    pressure = (ema_buy - ema_sell) / (ema_buy + ema_sell).replace(0, np.nan)
    return pressure.fillna(0.0)
