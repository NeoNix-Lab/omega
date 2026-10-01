"""Alpha Training Pipeline and Model Fitting Orchestrator.

Orchestrates causal feature extraction from the Feature Hub, temporal walk-forward folds,
model fitting, out-of-sample evaluation, strategy replays, and artifact sealing.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pandas as pd

from ..features import apply_features
from ..strategies import get_strategy
from ..tracking import ExperimentRegistry, RunRecord, get_git_provenance, save_model_artifact


def generate_synthetic_market_data(
    n_bars: int = 1500,
    base_price: float = 60000.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate reproducible OHLCV order flow bars with realistic volatility."""
    np.random.seed(seed)
    returns = np.random.normal(loc=0.0001, scale=0.002, size=n_bars)
    close_prices = base_price * np.exp(np.cumsum(returns))

    # Construct coherent OHLC from close prices
    intrabar_disp = np.random.exponential(scale=15.0, size=n_bars)
    high_prices = close_prices + intrabar_disp + np.random.uniform(2.0, 10.0, size=n_bars)
    low_prices = close_prices - intrabar_disp - np.random.uniform(2.0, 10.0, size=n_bars)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price

    # Order flow volume and delta
    buy_vol = np.random.exponential(scale=10.0, size=n_bars)
    # Drift buy volume with returns (microstructural correlation)
    buy_vol += np.maximum(0, returns * 5000.0)
    sell_vol = np.random.exponential(scale=10.0, size=n_bars)
    delta = buy_vol - sell_vol

    timestamps = pd.date_range("2024-01-01 00:00:00", periods=n_bars, freq="1min", tz="UTC")

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "buy_volume": buy_vol,
        "sell_volume": sell_vol,
        "delta": delta,
    })


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

    gamma = 0.5772156649
    if n_trials > 1:
        e_max_sharpe = (1.0 - gamma) * math.sqrt(2.0 * math.log(n_trials)) + (gamma / math.sqrt(2.0 * math.log(n_trials)))
    else:
        e_max_sharpe = 0.0

    var_sharpe = (1.0 - skewness * sharpe + ((kurtosis - 1.0) / 4.0) * (sharpe**2)) / sample_length
    std_sharpe = math.sqrt(max(1e-6, var_sharpe))

    dsr = (sharpe - e_max_sharpe) / std_sharpe
    return round(float(dsr), 4)


class LinearCentroidClassifier:
    """Deterministic classifier mapping features to directional trade probabilities."""

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
    feature_names: list[str] | str = "cvd_zscore,imbalance_ratio,price_momentum",
    horizon: int = 5,
    folds: int = 3,
    seed: int = 42,
    registry: ExperimentRegistry | None = None,
) -> RunRecord:
    """Execute end-to-end alpha training using pluggable features from the Feature Hub."""
    if registry is None:
        registry = ExperimentRegistry()

    if isinstance(feature_names, str):
        feature_names = [f.strip() for f in feature_names.split(",") if f.strip()]

    run_timestamp = datetime.now(timezone.utc).isoformat()
    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{name}"

    # 1. Acquire Data (Deterministic synthetic baseline)
    df = generate_synthetic_market_data(n_bars=1500, seed=seed)

    # 2. Causal Feature Engineering via Feature Hub
    df_feat, computed_feature_cols = apply_features(df, feature_names=feature_names)

    # 3. Target Label Definition: Forward return direction (+1, -1, 0)
    df_feat["fwd_ret"] = df_feat["close"].shift(-horizon) / df_feat["close"] - 1.0
    thresh = 0.0005
    df_feat["target"] = 0
    df_feat.loc[df_feat["fwd_ret"] > thresh, "target"] = 1
    df_feat.loc[df_feat["fwd_ret"] < -thresh, "target"] = -1

    # Drop warm-up rows
    valid_data = df_feat.dropna().copy()
    X = valid_data[computed_feature_cols].values
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

        strat_ret = preds * ret_test
        strategy_returns.extend(strat_ret.tolist())

    # 5. Compute Metrics
    mean_acc = float(np.mean(accuracies)) if accuracies else 0.0
    strat_ret_arr = np.array(strategy_returns)
    mean_ret = float(np.mean(strat_ret_arr)) if len(strat_ret_arr) > 0 else 0.0
    std_ret = float(np.std(strat_ret_arr)) if len(strat_ret_arr) > 0 and np.std(strat_ret_arr) > 1e-8 else 1.0
    sharpe = float((mean_ret / std_ret) * math.sqrt(252 * 1440))
    dsr = compute_deflated_sharpe_ratio(sharpe, n_trials=folds * 2, sample_length=len(strategy_returns))

    # Max Drawdown
    cum_returns = np.cumprod(1.0 + strat_ret_arr)
    peak = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - peak) / peak
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Spearman IC of the first feature
    first_feat_series = pd.Series(valid_data[computed_feature_cols[0]])
    ret_series = pd.Series(valid_data["fwd_ret"])
    spearman_ic = float(first_feat_series.rank().corr(ret_series.rank()))

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
        "features": feature_names,
        "computed_columns": computed_feature_cols,
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


def simulate_strategy_replay(
    strategy_name: str,
    feature_names: list[str] | str = "cvd_zscore,price_momentum,realized_vol",
    n_bars: int = 1500,
    initial_capital: float = 10000.0,
    seed: int = 42,
    strategy_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute a deterministic strategy replay and calculate portfolio statistics."""
    strategy_params = strategy_params or {}
    strategy = get_strategy(strategy_name, **strategy_params)

    # 1. Acquire Data
    df = generate_synthetic_market_data(n_bars=n_bars, seed=seed)

    # 2. Compute required features
    df_feat, _ = apply_features(df, feature_names=feature_names)

    # 3. Simulate Strategy Signals and Positions
    sim_df = strategy.simulate_signals(df_feat)

    # 4. Accounting and Performance Metrics
    trade_changes = sim_df["position"].diff().fillna(0.0) != 0
    trade_count = int(trade_changes.sum())

    total_return_pct = float(sim_df["equity_curve"].iloc[-1] - 1.0)
    final_equity = initial_capital * (1.0 + total_return_pct)

    strat_rets = sim_df["strategy_return"].values
    mean_ret = float(np.mean(strat_rets))
    std_ret = float(np.std(strat_rets)) if np.std(strat_rets) > 1e-8 else 1.0
    annualized_sharpe = float((mean_ret / std_ret) * math.sqrt(252 * 1440))

    # Max Drawdown
    equity = sim_df["equity_curve"].values
    peak = np.maximum.accumulate(equity)
    drawdowns = (equity - peak) / peak
    max_drawdown = float(np.min(drawdowns))

    # Win rate
    active_bars = strat_rets[sim_df["position"] != 0]
    win_rate = float(np.mean(active_bars > 0)) if len(active_bars) > 0 else 0.0

    return {
        "strategy_name": strategy.name,
        "description": strategy.description,
        "n_bars": n_bars,
        "initial_capital": initial_capital,
        "final_equity": round(final_equity, 2),
        "total_return_pct": round(total_return_pct * 100.0, 2),
        "trade_count": trade_count,
        "win_rate_pct": round(win_rate * 100.0, 2),
        "annualized_sharpe": round(annualized_sharpe, 2),
        "max_drawdown_pct": round(max_drawdown * 100.0, 2),
    }
