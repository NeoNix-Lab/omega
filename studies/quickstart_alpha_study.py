#!/usr/bin/env python3
"""Quickstart Alpha Study — Omega Research Lab.

Demonstrates how to conduct exploratory alpha research using quant-platform
as the underlying engine, respecting the DataGateway boundary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from omega.platform_link import check_compat


def compute_information_coefficient(signal: pd.Series, forward_returns: pd.Series) -> dict[str, float]:
    """Compute Spearman Rank IC and Pearson IC between signal and future return."""
    valid = signal.notna() & forward_returns.notna()
    sig = signal[valid]
    ret = forward_returns[valid]
    if len(sig) < 10:
        return {"spearman_ic": 0.0, "pearson_ic": 0.0, "sample_size": float(len(sig))}

    # Spearman rank correlation without requiring scipy
    spearman = float(sig.rank().corr(ret.rank(), method="pearson"))
    pearson = float(sig.corr(ret, method="pearson"))
    return {
        "spearman_ic": spearman,
        "pearson_ic": pearson,
        "sample_size": float(len(sig)),
    }


def main() -> int:
    print("=" * 60)
    print("OMEGA — Quickstart Alpha Research Study")
    print("=" * 60)

    # 1. Check quant_platform integration (through the single platform import surface)
    compat = check_compat()
    print(f"[{compat.status}] {compat.message}")

    # 2. Synthetic demonstration of an alpha research workflow
    print("\n[STEP 1] Generating synthetic order flow data (trades / delta)...")
    np.random.seed(42)
    n_bars = 500
    prices = 60000.0 + np.cumsum(np.random.randn(n_bars) * 15.0)
    buy_vol = np.random.exponential(scale=5.0, size=n_bars)
    sell_vol = np.random.exponential(scale=5.0, size=n_bars)
    delta = buy_vol - sell_vol

    df = pd.DataFrame({
        "close": prices,
        "delta": delta,
    })

    # Forward return (e.g. 5 bars ahead)
    horizon = 5
    df["fwd_ret"] = df["close"].shift(-horizon) / df["close"] - 1.0

    # Alpha candidate: Normalized cumulative delta (rolling 20 bars)
    rolling_delta = df["delta"].rolling(20).sum()
    rolling_std = df["delta"].rolling(20).std().replace(0, np.nan)
    alpha_signal = rolling_delta / rolling_std

    # 3. Evaluate Signal
    print(f"[STEP 2] Evaluating Alpha Signal against {horizon}-bar forward returns...")
    metrics = compute_information_coefficient(alpha_signal, df["fwd_ret"])
    print(f"  Spearman Rank IC : {metrics['spearman_ic']:.4f}")
    print(f"  Pearson IC       : {metrics['pearson_ic']:.4f}")
    print(f"  Sample Size      : {int(metrics['sample_size'])} bars")

    if abs(metrics['spearman_ic']) > 0.05:
        print("\n[RESULT] Promising statistical correlation detected (|IC| > 0.05).")
        print("  Next steps: Run purged/embargoed cross-validation and DSR/PBO checks.")
    else:
        print("\n[RESULT] Weak signal (|IC| <= 0.05). Refine feature definition or hypothesis.")

    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
