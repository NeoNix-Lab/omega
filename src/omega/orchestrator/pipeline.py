"""Alpha Training Pipeline and Model Fitting Orchestrator.

Orchestrates causal feature extraction, temporal walk-forward folds, model fitting,
out-of-sample evaluation, and artifact sealing with experiment registration.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pandas as pd

from ..tracking import ExperimentRegistry, RunRecord, get_git_provenance, save_model_artifact


def generate_synthetic_market_data(
    n_bars: int = 1500,
    base_price: float = 60000.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate reproducible order flow bars with price, buy/sell volumes, and delta."""
    np.random.seed(seed)
    returns = np.random.normal(loc=0.0001, scale=0.002, size=n_bars)
    prices = base_price * np.exp(np.cumsum(returns))

    # Volumes and aggressor imbalance
    buy_vol = np.random.exponential(scale=10.0, size=n_bars)
    # Give slight positive drift to buy volume when return is positive (synthetic microstructural correlation)
    buy_vol += np.maximum(0, returns * 5000.0)
    sell_vol = np.random.exponential(scale=10.0, size=n_bars)
    delta = buy_vol - sell_vol

    timestamps = pd.date_range("2024-01-01 00:00:00", periods=n_bars, freq="1min", tz="UTC")

    return pd.DataFrame({
        "timestamp": timestamps,
        "close": prices,
        "buy_volume": buy_vol,
        "sell_volume": sell_vol,
        "delta": delta,
    })


def compute_features(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Compute causal rolling microstructural features."""
    feat_df = df.copy()

    # 1. Rolling Cumulative Delta (Z-Score normalized)
    roll_delta = feat_df["delta"].rolling(lookback).sum()
    roll_std = feat_df["delta"].rolling(lookback).std().replace(0, np.nan)
    feat_df["feature_norm_delta"] = (roll_delta / roll_std).fillna(0.0)

    # 2. Volume Imbalance Ratio: (Buy - Sell) / (Buy + Sell)
    total_vol = feat_df["buy_volume"] + feat_df["sell_volume"]
    feat_df["feature_imbalance_ratio"] = (feat_df["delta"] / total_vol.replace(0, np.nan)).fillna(0.0)

    # 3. Short-term Price Momentum
    feat_df["feature_momentum"] = (feat_df["close"].pct_change(5)).fillna(0.0)

    return feat_df


def compute_deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    sample_length: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Compute approximate Deflated Sharpe Ratio (DSR) discounting selection bias."""
    if sample_length <= 1 or sharpe <= 0:
        return 0.0

    # Expected maximum Sharpe under null hypothesis (Euler-Mascheroni approximation)
    gamma = 0.5772156649
    if n_trials > 1:
        e_max_sharpe = (1.0 - gamma) * math.sqrt(2.0 * math.log(n_trials)) + (gamma / math.sqrt(2.0 * math.log(n_trials)))
    else:
        e_max_sharpe = 0.0

    # Variance of Sharpe estimate with non-normal returns
    var_sharpe = (1.0 - skewness * sharpe + ((kurtosis - 1.0) / 4.0) * (sharpe**2)) / sample_length
    std_sharpe = math.sqrt(max(1e-6, var_sharpe))

    dsr = (sharpe - e_max_sharpe) / std_sharpe
    # Return as an annualized-equivalent confidence score or normalized z-score
    return round(float(dsr), 4)


class LinearCentroidClassifier:
    """Simple deterministic classifier mapping features to directional probabilities."""

    def __init__(self) -> None:
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pos_mask = (y > 0)
        neg_mask = (y < 0)

        pos_center = X[pos_mask].mean(axis=0) if np.any(pos_mask) else np.zeros(X.shape[1])
        neg_center = X[neg_mask].mean(axis=0) if np.any(neg_mask) else np.zeros(X.shape[1])

        direction = pos_center - neg_center
        norm = np.linalg.norm(direction)
        self.weights = direction / norm if norm > 1e-9 else direction
        self.bias = float(-0.5 * np.dot(self.weights, pos_center + neg_center))

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = np.dot(X, self.weights) + self.bias
        preds = np.zeros(len(X), dtype=int)
        preds[scores > 0.05] = 1
        preds[scores < -0.05] = -1
        return preds

    def serialize(self) -> bytes:
        return pickle.dumps({
            "weights": self.weights.tolist() if self.weights is not None else [],
            "bias": self.bias,
        })


def execute_training_run(
    name: str = "orderflow_alpha",
    model_type: str = "centroid_classifier",
    lookback: int = 20,
    horizon: int = 5,
    folds: int = 3,
    seed: int = 42,
    registry: ExperimentRegistry | None = None,
) -> RunRecord:
    """Execute end-to-end alpha training with walk-forward CV and experiment registration."""
    if registry is None:
        registry = ExperimentRegistry()

    run_timestamp = datetime.now(timezone.utc).isoformat()
    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{name}"

    # 1. Acquire Data (Deterministic synthetic baseline)
    df = generate_synthetic_market_data(n_bars=1500, seed=seed)

    # 2. Causal Feature Engineering
    df_feat = compute_features(df, lookback=lookback)

    # 3. Label Definition: Forward return direction (+1, -1, 0)
    df_feat["fwd_ret"] = df_feat["close"].shift(-horizon) / df_feat["close"] - 1.0
    thresh = 0.0005
    df_feat["target"] = 0
    df_feat.loc[df_feat["fwd_ret"] > thresh, "target"] = 1
    df_feat.loc[df_feat["fwd_ret"] < -thresh, "target"] = -1

    # Drop warm-up and boundary rows
    valid_data = df_feat.dropna().copy()
    feature_cols = ["feature_norm_delta", "feature_imbalance_ratio", "feature_momentum"]
    X = valid_data[feature_cols].values
    y = valid_data["target"].values
    fwd_ret = valid_data["fwd_ret"].values

    # 4. Walk-Forward Cross-Validation (Purged & Embargoed)
    fold_size = len(valid_data) // (folds + 1)
    accuracies = []
    strategy_returns = []

    model = LinearCentroidClassifier()

    for i in range(folds):
        train_end = (i + 1) * fold_size
        test_start = train_end + 10  # 10-bar purge/embargo
        test_end = min(len(valid_data), test_start + fold_size)

        if test_start >= len(valid_data):
            break

        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[test_start:test_end], y[test_start:test_end]
        ret_test = fwd_ret[test_start:test_end]

        # Fit model
        model.fit(X_train, y_train)

        # Predict
        preds = model.predict(X_test)
        acc = float(np.mean(preds == y_test))
        accuracies.append(acc)

        # Economic return of directional bets
        strat_ret = preds * ret_test
        strategy_returns.extend(strat_ret.tolist())

    # 5. Compute Metrics
    mean_acc = float(np.mean(accuracies)) if accuracies else 0.0
    strat_ret_arr = np.array(strategy_returns)
    mean_ret = float(np.mean(strat_ret_arr)) if len(strat_ret_arr) > 0 else 0.0
    std_ret = float(np.std(strat_ret_arr)) if len(strat_ret_arr) > 0 and np.std(strat_ret_arr) > 1e-8 else 1.0
    sharpe = float((mean_ret / std_ret) * math.sqrt(252 * 1440))  # annualized assuming 1-min grain
    dsr = compute_deflated_sharpe_ratio(sharpe, n_trials=folds * 2, sample_length=len(strategy_returns))

    # Max Drawdown
    cum_returns = np.cumprod(1.0 + strat_ret_arr)
    peak = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - peak) / peak
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Spearman Rank IC
    sig_series = pd.Series(valid_data["feature_norm_delta"])
    ret_series = pd.Series(valid_data["fwd_ret"])
    spearman_ic = float(sig_series.rank().corr(ret_series.rank()))

    # 6. Fit final model on full set and seal artifact
    model.fit(X, y)
    payload = model.serialize()
    artifact_meta = save_model_artifact(payload, extension="bin")

    # 7. Code & Data Provenance
    git_prov = get_git_provenance()
    data_prov = {
        "dataset_id": "synthetic:orderflow:BTCUSDT:v1",
        "sample_rows": len(valid_data),
        "time_start": str(valid_data["timestamp"].iloc[0]),
        "time_end": str(valid_data["timestamp"].iloc[-1]),
    }

    parameters = {
        "name": name,
        "model_type": model_type,
        "features": feature_cols,
        "lookback": lookback,
        "horizon": horizon,
        "folds": folds,
        "seed": seed,
    }

    metrics = {
        "accuracy": round(mean_acc, 4),
        "spearman_ic": round(spearman_ic, 4),
        "sharpe_ratio": round(sharpe, 4),
        "deflated_sharpe_ratio": round(dsr, 4),
        "max_drawdown": round(max_dd, 4),
    }

    record = RunRecord(
        run_id=run_id,
        timestamp=run_timestamp,
        code_provenance=git_prov,
        data_provenance=data_prov,
        parameters=parameters,
        metrics=metrics,
        artifacts=artifact_meta,
        tags=("alpha", name, model_type),
    )

    registry.register(record)
    return record
