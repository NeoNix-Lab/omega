"""Command Line Orchestrator for Omega.

Provides the user with commands to orchestrate canonical quant-platform capabilities:
DataGateway, Feature Engine, StrategySpec, Deterministic Replay, and Supervised Learning.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .features import list_canonical_features
from .orchestrator.pipeline import orchestrate_canonical_replay, orchestrate_canonical_training
from .strategies import list_canonical_strategies
from .tracking import DEFAULT_REGISTRY_PATH, ExperimentRegistry, get_git_provenance


def cmd_status(args: argparse.Namespace) -> int:
    """Show system status, quant-platform link, git state, and canonical capabilities."""
    print("=" * 65)
    print(" [OMEGA] QUANT-PLATFORM CANONICAL ORCHESTRATOR STATUS")
    print("=" * 65)

    # 1. quant-platform link
    try:
        import quant_platform
        qp_version = getattr(quant_platform, "__version__", "dev")
        qp_path = Path(quant_platform.__file__).parent
        print(f"quant-platform Core  : [OK] linked (v{qp_version}) at {qp_path}")
    except ImportError as exc:
        print(f"quant-platform Core  : [FAIL] not installed on python path ({exc})")
        return 1

    # 2. Git provenance
    git = get_git_provenance()
    dirty_str = " (DIRTY TREE)" if git.get("is_dirty") else " (CLEAN)"
    print(f"Git Current Branch   : {git.get('git_branch')}{dirty_str}")
    print(f"Git HEAD Commit      : {git.get('git_commit')[:12] if git.get('git_commit') else 'unknown'}")

    # 3. Canonical Features and Strategies
    feats = list_canonical_features()
    strats = list_canonical_strategies()
    print(f"Canonical Features   : {len(feats)} frozen features in quant-platform")
    print(f"Canonical Strategies : {len(strats)} reference StrategySpec in quant-platform")

    # 4. Experiment registry
    registry = ExperimentRegistry()
    runs = registry.list_runs()
    print(f"Tracked Experiments  : {len(runs)} runs logged in {DEFAULT_REGISTRY_PATH.name}")

    print("=" * 65)
    return 0


def cmd_features(args: argparse.Namespace) -> int:
    """List frozen, canonical features and representations from quant-platform."""
    feats = list_canonical_features()
    print("=" * 80)
    print(f" [OMEGA] CANONICAL PLATFORM FEATURES ({len(feats)} available)")
    print("=" * 80)
    header = f"{'Feature Key':<24} | {'Domain':<14} | {'Version':<8} | {'Description':<30}"
    print(header)
    print("-" * 80)
    for f in feats:
        print(f"{f['feature_key']:<24} | {f['domain']:<14} | {f['version']:<8} | {f['description'][:30]}")
    print("=" * 80)
    return 0


def cmd_strategies(args: argparse.Namespace) -> int:
    """List canonical strategies and policies from quant-platform."""
    strats = list_canonical_strategies()
    print("=" * 80)
    print(f" [OMEGA] CANONICAL STRATEGY SPECIFICATIONS ({len(strats)} available)")
    print("=" * 80)
    header = f"{'Strategy Name':<20} | {'Strategy Identity':<35} | {'Description':<20}"
    print(header)
    print("-" * 80)
    for s in strats:
        print(f"{s['name']:<20} | {s['strategy_identity'][:35]:<35} | {s['description'][:20]}")
    print("=" * 80)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Orchestrate canonical Wave 5 supervised training from quant_platform.application."""
    print("=" * 65)
    print(f" [OMEGA] EXECUTING CANONICAL WAVE 5 SUPERVISED TRAINING")
    print("=" * 65)
    print(f"Target Name          : {args.name}")
    print(f"Code Reference       : {args.code_ref}")
    print("-" * 65)

    registry = ExperimentRegistry()
    record = orchestrate_canonical_training(
        name=args.name,
        code_ref=args.code_ref,
        registry=registry,
    )

    print("\n[SUCCESS] Canonical supervised training verified bitwise-deterministic!")
    print(f"  Run ID             : {record.run_id}")
    print(f"  Model Identity     : {record.parameters.get('model_identity')}")
    print(f"  Model Artifact     : {record.artifacts.get('sha256')}")
    print(f"  Normalizer Artifact: {record.artifacts.get('normalizer')}")
    print("-" * 65)
    print(" CANONICAL METRICS SCORECARD:")
    print(f"  Accuracy           : {record.metrics.get('accuracy'):.2%}")
    print(f"  Macro Precision    : {record.metrics.get('macro_precision'):.2%}")
    print(f"  Macro Recall       : {record.metrics.get('macro_recall'):.2%}")
    print(f"  Macro F1           : {record.metrics.get('macro_f1'):.2%}")
    print(f"  Brier Score        : {record.metrics.get('brier_score')}")
    print(f"  Evaluated Samples  : {int(record.metrics.get('row_count', 0))}")
    print("=" * 65)
    print(f"View leaderboard:  python omega.py leaderboard")
    print(f"Inspect manifest:  python omega.py inspect {record.run_id}")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    """Orchestrate canonical Wave 4 deterministic replay from quant_platform.application."""
    print("=" * 65)
    print(" [OMEGA] EXECUTING CANONICAL WAVE 4 REPLAY ORCHESTRATION")
    print("=" * 65)
    print(f"Start Window         : {args.start}")
    print(f"End Window           : {args.end}")
    print(f"Initial Capital      : ${args.capital}")
    print("-" * 65)

    res = orchestrate_canonical_replay(
        dsn=args.dsn,
        start=args.start,
        end=args.end,
        initial_capital=str(args.capital),
        lookback=args.lookback,
        batch_size=args.batch_size,
    )

    if res["status"] == "SPEC_BUILT_STANDALONE":
        print("[OK] Canonical ReplaySpec built successfully (No Live DSN passed):")
        print(f"  Spec Identity      : {res['spec_identity']}")
        print(f"  Dataset            : {res['dataset']}")
        print(f"  Strategy Identity  : {res['strategy']}")
        print(f"  Ordering Policy    : {res['ordering_policy']}")
        print("-" * 65)
        print(f"  Notice             : {res['message']}")
    else:
        print("[SUCCESS] Replay executed and verified against canonical catalog:")
        print(f"  Spec Identity      : {res['spec_identity']}")
        print(f"  Deterministic      : {res['deterministic']}")
        print(f"  Trace Fingerprint  : {res['trace_fingerprint']}")
        print(f"  Orders Realized    : {res['orders_count']}")
        print(f"  Fills Realized     : {res['fills_count']}")

    print("=" * 65)
    return 0


def cmd_leaderboard(args: argparse.Namespace) -> int:
    """Print leaderboard ranking tracked canonical training runs."""
    registry = ExperimentRegistry()
    top_runs = registry.leaderboard(sort_by=args.sort_by, ascending=False, limit=args.limit)

    print("=" * 80)
    print(f" [OMEGA] CANONICAL EXPERIMENT LEADERBOARD (Ranked by {args.sort_by.upper()})")
    print("=" * 80)

    if not top_runs:
        print("No training runs registered yet. Run one with: python omega.py train")
        print("=" * 80)
        return 0

    header = f"{'Rank':<5} | {'Run ID':<30} | {'Acc':<7} | {'Model Artifact SHA':<32}"
    print(header)
    print("-" * 80)

    for idx, r in enumerate(top_runs, 1):
        line = (
            f"{idx:<5} | "
            f"{r['run_id'][:30]:<30} | "
            f"{r['accuracy']:.2%} | "
            f"{r['model_sha'][:32]:<32}"
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
        description="Omega - Canonical Orchestrator for quant-platform",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # status
    subparsers.add_parser("status", help="Show system status and canonical quant-platform capabilities")

    # features
    p_feat = subparsers.add_parser("features", help="List canonical features in quant-platform")
    p_feat.add_argument("action", choices=["list"], default="list", nargs="?", help="Action (default: list)")

    # strategies
    p_strat = subparsers.add_parser("strategies", help="List canonical StrategySpecs in quant-platform")
    p_strat.add_argument("action", choices=["list"], default="list", nargs="?", help="Action (default: list)")

    # train
    p_train = subparsers.add_parser("train", help="Orchestrate canonical Wave 5 supervised training")
    p_train.add_argument("--name", type=str, default="wave5_supervised_model", help="Identifier name for the run")
    p_train.add_argument("--code-ref", type=str, default="omega-orchestrator-v1", help="Code reference version")

    # replay
    p_rep = subparsers.add_parser("replay", help="Orchestrate canonical Wave 4 deterministic replay")
    p_rep.add_argument("--dsn", type=str, default="", help="Catalog PostgreSQL DSN (or via GOLDEN_REPLAY_E2E_DSN)")
    p_rep.add_argument("--start", type=str, default="2024-01-15T00:00:00Z", help="Start timestamp")
    p_rep.add_argument("--end", type=str, default="2024-01-16T00:00:00Z", help="End timestamp")
    p_rep.add_argument("--capital", type=int, default=10000, help="Initial capital in base currency")
    p_rep.add_argument("--lookback", type=int, default=20, help="Lookback window")
    p_rep.add_argument("--batch-size", type=int, default=65536, help="DataGateway batch size")

    # leaderboard
    p_lead = subparsers.add_parser("leaderboard", help="View ranked leaderboard of tracked runs")
    p_lead.add_argument("--sort-by", type=str, default="accuracy", choices=["accuracy"], help="Metric to rank by")
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
    elif args.command == "features":
        return cmd_features(args)
    elif args.command == "strategies":
        return cmd_strategies(args)
    elif args.command == "train":
        return cmd_train(args)
    elif args.command == "replay":
        return cmd_replay(args)
    elif args.command == "leaderboard":
        return cmd_leaderboard(args)
    elif args.command == "inspect":
        return cmd_inspect(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
