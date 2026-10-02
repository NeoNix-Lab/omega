#!/usr/bin/env python3
"""Smoke test of the quant platform through Omega: catalog data -> Omega specs -> platform replay.

Deliberately thin. Everything quantitative is the platform's (DataGateway, HistoricalReplayRuntime,
ledger, fees); Omega only contributes the declarative feature/strategy specs. Anything that breaks here
is a platform finding: comment it on the upstream-asks issue (#15) with the exact error.

    python scripts/smoke_replay.py --dsn "$OMEGA_CATALOG_DSN" --start 2024-01-15T00:00:00Z \
        --end 2024-01-16T00:00:00Z --features studies/specs/features/momentum_1m_5.yaml \
        --strategy studies/specs/strategies/volatility_breakout_long.yaml --twice
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from omega import platform_link as qp
from omega.features import load_feature_spec
from omega.strategies import compile_strategy, load_strategy_spec

DATASET = ("canonical", "trades", "bybit", "BTCUSDT", "trade-v1")


def open_gateway(dsn: str) -> Any:
    return qp.DataGateway(qp.Catalog(dsn=dsn), ordering_providers=(qp.BYBIT_ORDERING_PROVIDER,))


def run_smoke(
    gateway: Any,
    *,
    features: Sequence[Path],
    strategy: Path,
    start: str,
    end: str,
    capital: str = "10000",
    maker_fee: str | None = None,
    taker_fee: str | None = None,
    slippage_bps: str | None = None,
    batch_size: int = 65_536,
    ordering_policy: str | None = None,
) -> dict[str, Any]:
    """One tick-level replay through the platform; returns identities, outcome and timing."""
    compiled = compile_strategy(load_strategy_spec(strategy), [load_feature_spec(path) for path in features])
    fee_rates = {k: v for k, v in (("maker_fee_rate", maker_fee), ("taker_fee_rate", taker_fee)) if v is not None}
    spec = qp.ReplaySpec(
        dataset=qp.DatasetIdentity(*DATASET),
        start=start,
        end=end,
        strategy=compiled.platform_spec,
        initial_capital=capital,
        fee_schedule=qp.FeeSchedule(policy_key="omega.smoke.fees", **fee_rates),
        ordering_policy=ordering_policy or qp.BYBIT_TRADE_V1_ORDERING_POLICY,
        slippage_model=None
        if slippage_bps is None
        else qp.SyntheticSlippageModel(policy_key="omega.smoke.slippage", slippage_bps=slippage_bps),
        batch_size=batch_size,
    )
    began = time.perf_counter()
    result = qp.HistoricalReplayRuntime(gateway, compiled.make_provider()).run(spec)
    elapsed = time.perf_counter() - began
    ticks = len(result.equity_curve)
    if ticks == 0:
        raise RuntimeError("the platform returned no records for this interval")
    final_equity = Decimal(result.equity_curve[-1].equity)
    return {
        "spec_identity": spec.identity,
        "result_identity": result.identity,
        "trace_fingerprint": result.trace_fingerprint,
        "strategy_identity": compiled.platform_spec.strategy_identity,
        "ticks": ticks,
        "orders": len(result.orders),
        "fills": len(result.fills),
        "fees": str(sum((fill.fee for fill in result.fills), Decimal(0))),
        "initial_capital": capital,
        "final_equity": str(final_equity),
        "return_pct": str((final_equity / Decimal(capital) - 1) * 100),
        "fee_rates": f"maker={spec.fee_schedule.maker_fee_rate} taker={spec.fee_schedule.taker_fee_rate}",
        "elapsed_s": round(elapsed, 3),
        "ticks_per_s": round(ticks / elapsed) if elapsed else None,
        "coverage_complete": getattr(result.data_metadata, "coverage_complete", None),
        "data_row_count": getattr(result.data_metadata, "row_count", None),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dsn", default=os.environ.get("OMEGA_CATALOG_DSN"), help="catalog DSN (or OMEGA_CATALOG_DSN)")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--features", nargs="+", type=Path, required=True)
    parser.add_argument("--strategy", type=Path, required=True)
    parser.add_argument("--capital", default="10000")
    parser.add_argument("--maker-fee", help="default: platform default (source unverified)")
    parser.add_argument("--taker-fee", help="default: platform default (source unverified)")
    parser.add_argument("--slippage-bps")
    parser.add_argument("--twice", action="store_true", help="run twice and compare identities (determinism)")
    args = parser.parse_args(argv)
    if not args.dsn:
        print("error: no catalog DSN (use --dsn or OMEGA_CATALOG_DSN)", file=sys.stderr)
        return 2

    gateway = open_gateway(args.dsn)
    kwargs = {
        "features": args.features,
        "strategy": args.strategy,
        "start": args.start,
        "end": args.end,
        "capital": args.capital,
        "maker_fee": args.maker_fee,
        "taker_fee": args.taker_fee,
        "slippage_bps": args.slippage_bps,
    }
    report = run_smoke(gateway, **kwargs)
    print("SMOKE RUN (real catalog data, tick-level) - mechanics only, NOT AN ALPHA CLAIM")
    for key, value in report.items():
        print(f"  {key:18} {value}")
    if args.twice:
        again = run_smoke(gateway, **kwargs)
        same = (again["result_identity"], again["trace_fingerprint"]) == (
            report["result_identity"],
            report["trace_fingerprint"],
        )
        print(f"  determinism        {'IDENTICAL' if same else 'DIFFERENT'} (second run {again['elapsed_s']} s)")
        return 0 if same else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
