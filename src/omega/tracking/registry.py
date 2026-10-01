"""Experiment Registry and Provenance Tracker for Omega.

Tracks model training, hyperparameters, data lineage, git revisions, and
evaluations in immutable lightweight manifests, while keeping heavy model
binaries off Git.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_REGISTRY_PATH = REPO_ROOT / "experiments" / "registry.jsonl"
DEFAULT_MODELS_DIR = REPO_ROOT / "data" / "models"


def get_git_provenance() -> dict[str, Any]:
    """Capture current Git commit, branch and working tree dirty status."""
    try:
        commit_res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        commit_sha = commit_res.stdout.strip() if commit_res.returncode == 0 else "unknown"

        branch_res = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        branch_name = branch_res.stdout.strip() if branch_res.returncode == 0 else "unknown"

        dirty_res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        is_dirty = bool(dirty_res.stdout.strip()) if dirty_res.returncode == 0 else False

        return {
            "git_commit": commit_sha,
            "git_branch": branch_name,
            "is_dirty": is_dirty,
        }
    except Exception as exc:
        return {
            "git_commit": "error",
            "git_branch": "error",
            "is_dirty": True,
            "error": str(exc),
        }


def save_model_artifact(payload_bytes: bytes, extension: str = "bin") -> dict[str, str]:
    """Save model binary payload to local storage named by its SHA-256 hash.

    Keeps heavy binaries out of Git while maintaining cryptographic verification.
    """
    DEFAULT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    sha256_hash = hashlib.sha256(payload_bytes).hexdigest()
    filename = f"model_{sha256_hash[:16]}.{extension}"
    target_path = DEFAULT_MODELS_DIR / filename

    if not target_path.exists():
        target_path.write_bytes(payload_bytes)

    relative_locator = str(target_path.relative_to(REPO_ROOT)).replace("\\", "/")
    return {
        "sha256": f"sha256:{sha256_hash}",
        "storage_locator": relative_locator,
        "byte_size": str(len(payload_bytes)),
    }


@dataclass(frozen=True)
class RunRecord:
    """Immutable manifest for one training run or experiment evaluation."""

    run_id: str
    timestamp: str
    code_provenance: dict[str, Any]
    data_provenance: dict[str, Any]
    parameters: dict[str, Any]
    metrics: dict[str, float]
    artifacts: dict[str, Any]
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tags"] = list(self.tags)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunRecord:
        tags = tuple(data.get("tags", []))
        return cls(
            run_id=data["run_id"],
            timestamp=data["timestamp"],
            code_provenance=data.get("code_provenance", {}),
            data_provenance=data.get("data_provenance", {}),
            parameters=data.get("parameters", {}),
            metrics=data.get("metrics", {}),
            artifacts=data.get("artifacts", {}),
            tags=tags,
        )


class ExperimentRegistry:
    """Append-only, line-delimited JSON registry of experiment runs."""

    def __init__(self, registry_file: Path | str | None = None) -> None:
        self.path = Path(registry_file) if registry_file else DEFAULT_REGISTRY_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def register(self, record: RunRecord) -> RunRecord:
        """Append a new RunRecord to the registry log."""
        line = json.dumps(record.to_dict(), sort_keys=True) + "\n"
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line)
        return record

    def list_runs(self, tag: str | None = None) -> list[RunRecord]:
        """Read and return all recorded runs."""
        if not self.path.exists():
            return []

        runs = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    record_dict = json.loads(line_str)
                    record = RunRecord.from_dict(record_dict)
                    if tag is None or tag in record.tags:
                        runs.append(record)
                except Exception:
                    continue
        return runs

    def get(self, query: str) -> RunRecord | None:
        """Find a run by exact run_id, or by artifact sha256 substring."""
        for run in self.list_runs():
            if run.run_id == query:
                return run
            model_hash = run.artifacts.get("sha256", "")
            if query in model_hash:
                return run
        return None

    def leaderboard(
        self,
        sort_by: str = "deflated_sharpe_ratio",
        ascending: bool = False,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Return top runs ranked by specified metric."""
        runs = self.list_runs()
        rows = []
        for r in runs:
            metric_val = r.metrics.get(sort_by, float("-inf") if not ascending else float("inf"))
            rows.append({
                "run_id": r.run_id,
                "timestamp": r.timestamp,
                "model_type": r.parameters.get("model_type", "unknown"),
                "features": r.parameters.get("features", "unknown"),
                "sort_metric": metric_val,
                "accuracy": r.metrics.get("accuracy", 0.0),
                "sharpe": r.metrics.get("sharpe_ratio", 0.0),
                "dsr": r.metrics.get("deflated_sharpe_ratio", 0.0),
                "model_sha": r.artifacts.get("sha256", "none")[:23] + "...",
            })

        rows.sort(key=lambda x: x["sort_metric"], reverse=not ascending)
        return rows[:limit]
