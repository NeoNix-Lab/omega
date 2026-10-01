"""Canonical Application Orchestrator for Omega.

Directly composes quant-platform application services (Wave 4 Replay, Wave 5 Supervised,
and DataGateway) without creating parallel or duplicate domain logic.
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any

from quant_platform.application import (
    CANONICAL_BTCUSDT_DATASET,
    build_golden_replay_spec,
    minimal_breakout_strategy,
    run_golden_replay_proof,
    run_wave5_golden_supervised_proof,
)

from ..tracking import ExperimentRegistry, RunRecord, get_git_provenance


def orchestrate_canonical_training(
    name: str = "supervised_orderflow_model",
    code_ref: str = "omega-cli",
    execution_id: str | None = None,
    registry: ExperimentRegistry | None = None,
) -> RunRecord:
    """Orchestrate canonical Wave 5 supervised training from quant_platform.application."""
    if registry is None:
        registry = ExperimentRegistry()

    if execution_id is None:
        execution_id = f"exec_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    run_timestamp = datetime.now(timezone.utc).isoformat()
    run_id = f"run_{execution_id}_{name}"

    # 1. Execute canonical quant-platform Wave 5 proof / training
    proof = run_wave5_golden_supervised_proof(
        code_ref=code_ref,
        execution_id=execution_id,
    )

    summary = proof.stable_dict()
    first_run = proof.first
    metrics_obj = first_run.metrics
    artifacts_map = summary.get("artifact_content_identities", {})

    # 2. Extract canonical metrics
    metrics = {
        "accuracy": float(metrics_obj.accuracy),
        "macro_precision": float(metrics_obj.macro_precision),
        "macro_recall": float(metrics_obj.macro_recall),
        "macro_f1": float(metrics_obj.macro_f1),
        "brier_score": round(float(metrics_obj.brier_score), 4),
        "row_count": float(metrics_obj.row_count),
    }

    # 3. Extract canonical provenance
    data_prov = {
        "dataset_layer": proof.dataset_identity.layer,
        "dataset_kind": proof.dataset_identity.dataset_kind,
        "venue": proof.dataset_identity.venue,
        "instrument": proof.dataset_identity.instrument,
        "schema": proof.dataset_identity.record_schema_id,
        "source_evidence_identity": proof.source_evidence_identity,
        "projection_identity": summary.get("projection_identity", "unknown"),
    }

    parameters = {
        "name": name,
        "code_ref": code_ref,
        "execution_id": execution_id,
        "proof_version": summary.get("proof_version", "1"),
        "training_policy_identity": summary.get("training_policy_identity", "unknown"),
        "selection_policy_identity": summary.get("selection_policy_identity", "unknown"),
        "normalizer_identity": summary.get("normalizer_identity", "unknown"),
        "model_identity": summary.get("model_identity", "unknown"),
        "deterministic": proof.deterministic,
    }

    artifact_meta = {
        "sha256": artifacts_map.get("model", "unknown"),
        "normalizer": artifacts_map.get("normalizer", "unknown"),
        "predictions": artifacts_map.get("predictions", "unknown"),
        "metrics_artifact": artifacts_map.get("metrics", "unknown"),
    }

    git_prov = get_git_provenance()

    record = RunRecord(
        run_id=run_id,
        timestamp=run_timestamp,
        code_provenance=git_prov,
        data_provenance=data_prov,
        parameters=parameters,
        metrics=metrics,
        artifacts=artifact_meta,
        tags=("quant_platform", "canonical", "wave5_supervised", name),
    )

    registry.register(record)
    return record


def orchestrate_canonical_replay(
    dsn: str | None = None,
    start: str = "2024-01-15T00:00:00Z",
    end: str = "2024-01-16T00:00:00Z",
    initial_capital: str = "10000",
    lookback: int = 20,
    batch_size: int = 65_536,
) -> dict[str, Any]:
    """Orchestrate canonical Wave 4 deterministic replay from quant_platform.application."""
    resolved_dsn = (
        dsn
        or os.environ.get("GOLDEN_REPLAY_E2E_DSN")
        or os.environ.get("DATA_GATEWAY_TEST_DSN")
    )

    if not resolved_dsn:
        # Build specification without executing live DSN
        spec = build_golden_replay_spec(
            start=start,
            end=end,
            initial_capital=initial_capital,
            batch_size=batch_size,
        )
        return {
            "status": "SPEC_BUILT_STANDALONE",
            "message": "ReplaySpec constructed successfully. To run against the live catalog database, pass --dsn or set GOLDEN_REPLAY_E2E_DSN.",
            "spec_identity": spec.identity,
            "dataset": str(spec.dataset),
            "start": spec.start,
            "end": spec.end,
            "initial_capital": spec.initial_capital,
            "batch_size": spec.batch_size,
            "strategy": spec.strategy.strategy_identity,
            "ordering_policy": spec.ordering_policy,
        }

    # Execute against real catalog
    proof = run_golden_replay_proof(
        dsn=resolved_dsn,
        start=start,
        end=end,
        initial_capital=initial_capital,
        lookback=lookback,
        batch_size=batch_size,
    )

    return {
        "status": "REPLAY_PROVEN_DETERMINISTIC",
        "spec_identity": proof.spec_identity,
        "deterministic": proof.deterministic,
        "trace_fingerprint": proof.first.trace_fingerprint,
        "first_run_equals_second": (proof.first.trace_fingerprint == proof.second.trace_fingerprint),
        "fills_count": len(proof.first.fills),
        "orders_count": len(proof.first.orders),
    }
