"""Command Line Orchestrator for Omega.

Provides the user with commands to inspect status, launch alpha model trainings,
view leaderboard rankings, and inspect experiment provenance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .orchestrator.pipeline import execute_training_run
from .tracking import DEFAULT_MODELS_DIR, DEFAULT_REGISTRY_PATH, ExperimentRegistry, get_git_provenance


def cmd_status(args: argparse.Namespace) -> int:
    """Show system status, quant-platform link, git state, and tracked experiments."""
    print("=" * 65)
    print(" [OMEGA] SYSTEM & RESEARCH STATUS")
    print("=" * 65)

    # 1. quant-platform link
    try:
        import quant_platform
        qp_version = getattr(quant_platform, "__version__", "dev")
        qp_path = Path(quant_platform.__file__).parent
        print(f"quant-platform Core  : [OK] linked (v{qp_version}) at {qp_path}")
    except ImportError as exc:
        print(f"quant-platform Core  : [WARN] not installed on python path ({exc})")

    # 2. Git provenance
    git = get_git_provenance()
    dirty_str = " (DIRTY TREE)" if git.get("is_dirty") else " (CLEAN)"
    print(f"Git Current Branch   : {git.get('git_branch')}{dirty_str}")
    print(f"Git HEAD Commit      : {git.get('git_commit')[:12] if git.get('git_commit') else 'unknown'}")

    # 3. Experiment registry
    registry = ExperimentRegistry()
    runs = registry.list_runs()
    print(f"Tracked Runs Count   : {len(runs)} runs in {DEFAULT_REGISTRY_PATH.name}")

    # 4. Storage artifacts
    if DEFAULT_MODELS_DIR.exists():
        model_files = list(DEFAULT_MODELS_DIR.glob("*.bin"))
        total_size_bytes = sum(f.stat().st_size for f in model_files)
        total_kb = total_size_bytes / 1024.0
        print(f"Model Artifacts      : {len(model_files)} weights files ({total_kb:.1f} KB in data/models/)")
    else:
        print("Model Artifacts      : 0 files")

    print("=" * 65)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Launch an alpha training pipeline run and track it in the experiment registry."""
    print("=" * 65)
    print(f" [OMEGA] LAUNCHING ALPHA TRAINING PIPELINE: {args.name}")
    print("=" * 65)
    print(f"Model Type           : {args.model_type}")
    print(f"Lookback Window      : {args.lookback} bars")
    print(f"Prediction Horizon   : {args.horizon} bars")
    print(f"Walk-Forward Folds   : {args.folds} (Purged & Embargoed)")
    print(f"Seed                 : {args.seed}")
    print("-" * 65)

    registry = ExperimentRegistry()
    record = execute_training_run(
        name=args.name,
        model_type=args.model_type,
        lookback=args.lookback,
        horizon=args.horizon,
        folds=args.folds,
        seed=args.seed,
        registry=registry,
    )

    print("\n[SUCCESS] Training completed and experiment registered!")
    print(f"  Run ID             : {record.run_id}")
    print(f"  Model SHA-256      : {record.artifacts.get('sha256')}")
    print(f"  Saved Locator      : {record.artifacts.get('storage_locator')}")
    print("-" * 65)
    print(" EVALUATION SCORECARD:")
    print(f"  Accuracy           : {record.metrics.get('accuracy'):.2%}")
    print(f"  Spearman IC        : {record.metrics.get('spearman_ic'):.4f}")
    print(f"  Annualized Sharpe  : {record.metrics.get('sharpe_ratio'):.2f}")
    print(f"  Deflated Sharpe    : {record.metrics.get('deflated_sharpe_ratio'):.2f} (DSR)")
    print(f"  Max Drawdown       : {record.metrics.get('max_drawdown'):.2%}")
    print("=" * 65)
    print(f"View leaderboard:  python -m omega leaderboard")
    print(f"Inspect run:       python -m omega inspect {record.run_id}")
    return 0


def cmd_leaderboard(args: argparse.Namespace) -> int:
    """Print a clean leaderboard ranking all tracked training runs."""
    registry = ExperimentRegistry()
    metric_map = {
        "dsr": "deflated_sharpe_ratio",
        "sharpe": "sharpe_ratio",
        "accuracy": "accuracy",
        "ic": "spearman_ic",
    }
    sort_key = metric_map.get(args.sort_by, args.sort_by)
    top_runs = registry.leaderboard(sort_by=sort_key, ascending=False, limit=args.limit)

    print("=" * 80)
    print(f" [OMEGA] EXPERIMENT LEADERBOARD (Ranked by {sort_key.upper()})")
    print("=" * 80)

    if not top_runs:
        print("No training runs registered yet. Run one with: python omega.py train")
        print("=" * 80)
        return 0

    header = f"{'Rank':<5} | {'Run ID':<26} | {'Model':<12} | {'Acc':<7} | {'Sharpe':<8} | {'DSR':<7} | {'Model SHA':<14}"
    print(header)
    print("-" * 80)

    for idx, r in enumerate(top_runs, 1):
        line = (
            f"{idx:<5} | "
            f"{r['run_id'][:26]:<26} | "
            f"{r['model_type'][:12]:<12} | "
            f"{r['accuracy']:.2%} | "
            f"{r['sharpe']:>7.2f} | "
            f"{r['dsr']:>6.2f} | "
            f"{r['model_sha'][:14]:<14}"
        )
        print(line)

    print("=" * 80)
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Inspect full provenance and metrics of a single training run."""
    registry = ExperimentRegistry()
    run = registry.get(args.query)
    if not run:
        print(f"[ERROR] Could not find run matching: '{args.query}'", file=sys.stderr)
        return 1

    print("=" * 65)
    print(f" [OMEGA] RUN MANIFEST: {run.run_id}")
    print("=" * 65)
    print(json.dumps(run.to_dict(), indent=2))
    print("=" * 65)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="omega",
        description="Omega - Alpha Discovery and Experimentation Lab Orchestrator",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # status
    subparsers.add_parser("status", help="Show system status, git state, and experiment count")

    # train
    p_train = subparsers.add_parser("train", help="Launch a causal alpha model training run")
    p_train.add_argument("--name", type=str, default="orderflow_alpha", help="Identifier name for the run")
    p_train.add_argument("--model-type", type=str, default="centroid_classifier", help="Model architecture")
    p_train.add_argument("--lookback", type=int, default=20, help="Feature lookback window in bars")
    p_train.add_argument("--horizon", type=int, default=5, help="Forward return prediction horizon in bars")
    p_train.add_argument("--folds", type=int, default=3, help="Number of walk-forward validation folds")
    p_train.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # leaderboard
    p_lead = subparsers.add_parser("leaderboard", help="View ranked leaderboard of tracked runs")
    p_lead.add_argument("--sort-by", type=str, default="dsr", choices=["dsr", "sharpe", "accuracy", "ic"], help="Metric to rank by")
    p_lead.add_argument("--limit", type=int, default=10, help="Max runs to display")

    # inspect
    p_inspect = subparsers.add_parser("inspect", help="Inspect complete JSON provenance of a run")
    p_inspect.add_argument("query", type=str, help="Run ID or Model SHA-256 substring")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 0

    if args.command == "status":
        return cmd_status(args)
    elif args.command == "train":
        return cmd_train(args)
    elif args.command == "leaderboard":
        return cmd_leaderboard(args)
    elif args.command == "inspect":
        return cmd_inspect(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
