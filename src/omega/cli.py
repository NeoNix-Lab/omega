"""Command Line Orchestrator for Omega.

Provides commands to manage derived features, strategies, model trainings,
leaderboard comparisons, and strategy replay simulations.
"""

from __future__ import annotations

import argparse
import json
import sys

from .features import list_features
from .orchestrator.pipeline import execute_training_run, simulate_strategy_replay
from .platform_link import PLATFORM_PIN, CompatStatus, check_compat
from .strategies import list_strategies
from .tracking import DEFAULT_MODELS_DIR, DEFAULT_REGISTRY_PATH, ExperimentRegistry, get_git_provenance


def cmd_status(args: argparse.Namespace) -> int:
    """Show system status, quant-platform link, git state, and tracked experiments."""
    print("=" * 65)
    print(" [OMEGA] SYSTEM & RESEARCH STATUS")
    print("=" * 65)

    # 1. quant-platform link
    compat = check_compat()
    print(f"quant-platform Core  : [{compat.status}] {compat.message}")
    print(f"quant-platform Pin   : {PLATFORM_PIN.version} @ {PLATFORM_PIN.commit[:12]} ({PLATFORM_PIN.tag})")

    # 2. Git provenance
    git = get_git_provenance()
    dirty_str = " (DIRTY TREE)" if git.get("is_dirty") else " (CLEAN)"
    print(f"Git Current Branch   : {git.get('git_branch')}{dirty_str}")
    print(f"Git HEAD Commit      : {git.get('git_commit')[:12] if git.get('git_commit') else 'unknown'}")

    # 3. Available Features and Strategies
    feats = list_features()
    strats = list_strategies()
    print(f"Registered Features  : {len(feats)} features in Feature Hub")
    print(f"Registered Strategies: {len(strats)} strategies in Strategy Hub")

    # 4. Experiment registry
    registry = ExperimentRegistry()
    runs = registry.list_runs()
    print(f"Tracked Runs Count   : {len(runs)} runs in {DEFAULT_REGISTRY_PATH.name}")

    # 5. Storage artifacts
    if DEFAULT_MODELS_DIR.exists():
        model_files = list(DEFAULT_MODELS_DIR.glob("*.bin"))
        total_size_bytes = sum(f.stat().st_size for f in model_files)
        total_kb = total_size_bytes / 1024.0
        print(f"Model Artifacts      : {len(model_files)} weights files ({total_kb:.1f} KB in data/models/)")
    else:
        print("Model Artifacts      : 0 files")

    print("=" * 65)
    if getattr(args, "strict", False) and compat.status is not CompatStatus.OK:
        print(f"[ERROR] --strict: platform check is {compat.status}", file=sys.stderr)
        return 1
    return 0


def cmd_features(args: argparse.Namespace) -> int:
    """List all registered derived features in the Feature Hub."""
    feats = list_features()
    print("=" * 80)
    print(f" [OMEGA] REGISTERED DERIVED FEATURES ({len(feats)} available)")
    print("=" * 80)
    header = f"{'Feature Name':<22} | {'Required Columns':<24} | {'Description':<30}"
    print(header)
    print("-" * 80)
    for f in feats:
        cols_str = ", ".join(f.required_columns)
        print(f"{f.name:<22} | {cols_str:<24} | {f.description[:30]}")
    print("=" * 80)
    print("Use any feature in training:  omega train --features cvd_zscore,rsi,realized_vol")
    return 0


def cmd_strategies(args: argparse.Namespace) -> int:
    """List all registered strategies in the Strategy Hub."""
    strats = list_strategies()
    print("=" * 80)
    print(f" [OMEGA] REGISTERED STRATEGIES ({len(strats)} available)")
    print("=" * 80)
    header = f"{'Strategy Name':<25} | {'Default Parameters':<25} | {'Description':<26}"
    print(header)
    print("-" * 80)
    for s in strats:
        params_str = ", ".join(f"{k}={v}" for k, v in list(s.default_params.items())[:2])
        print(f"{s.name:<25} | {params_str:<25} | {s.description[:26]}")
    print("=" * 80)
    print("Simulate a strategy:  omega replay --strategy orderflow_absorption")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Launch an alpha training pipeline run and track it in the experiment registry."""
    print("=" * 65)
    print(f" [OMEGA] LAUNCHING ALPHA TRAINING PIPELINE: {args.name}")
    print("=" * 65)
    print(f"Model Type           : {args.model_type}")
    print(f"Features             : {args.features}")
    print(f"Prediction Horizon   : {args.horizon} bars")
    print(f"Walk-Forward Folds   : {args.folds} (Purged & Embargoed)")
    print(f"Seed                 : {args.seed}")
    print("-" * 65)

    registry = ExperimentRegistry()
    record = execute_training_run(
        name=args.name,
        model_type=args.model_type,
        feature_names=args.features,
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
    print("View leaderboard:  omega leaderboard")
    print(f"Inspect run:       omega inspect {record.run_id}")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    """Run a deterministic strategy replay and display portfolio statistics."""
    print("=" * 65)
    print(f" [OMEGA] STRATEGY REPLAY SIMULATION: {args.strategy}")
    print("=" * 65)
    print(f"Features Fed         : {args.features}")
    print(f"Bars Simulated       : {args.bars}")
    print(f"Initial Capital      : ${args.capital:,.2f}")
    print("-" * 65)

    res = simulate_strategy_replay(
        strategy_name=args.strategy,
        feature_names=args.features,
        n_bars=args.bars,
        initial_capital=args.capital,
    )

    print("\n[SUCCESS] Replay completed!")
    print(f"  Strategy           : {res['strategy_name']}")
    print(f"  Description        : {res['description']}")
    print("-" * 65)
    print(" REPLAY SCORECARD:")
    print(f"  Final Equity       : ${res['final_equity']:,.2f} ({res['total_return_pct']:+.2f}%)")
    print(f"  Total Trades       : {res['trade_count']}")
    print(f"  Win Rate           : {res['win_rate_pct']:.2f}%")
    print(f"  Annualized Sharpe  : {res['annualized_sharpe']:.2f}")
    print(f"  Max Drawdown       : {res['max_drawdown_pct']:.2f}%")
    print("=" * 65)
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
        print("No training runs registered yet. Run one with: omega train")
        print("=" * 80)
        return 0

    header = (
        f"{'Rank':<5} | {'Run ID':<26} | {'Model':<12} | {'Acc':<7} | "
        f"{'Sharpe':<8} | {'DSR':<7} | {'Model SHA':<14}"
    )
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
    p_status = subparsers.add_parser("status", help="Show system status, git state, and experiment count")
    p_status.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when quant-platform is missing or does not match the pinned version",
    )

    # features
    p_feat = subparsers.add_parser("features", help="Manage and list derived features in Feature Hub")
    p_feat.add_argument("action", choices=["list"], default="list", nargs="?", help="Action (default: list)")

    # strategies
    p_strat = subparsers.add_parser("strategies", help="Manage and list strategies in Strategy Hub")
    p_strat.add_argument("action", choices=["list"], default="list", nargs="?", help="Action (default: list)")

    # train
    p_train = subparsers.add_parser("train", help="Launch a causal alpha model training run")
    p_train.add_argument("--name", type=str, default="orderflow_alpha", help="Identifier name for the run")
    p_train.add_argument("--model-type", type=str, default="centroid_classifier", help="Model architecture")
    p_train.add_argument(
        "--features",
        type=str,
        default="cvd_zscore,imbalance_ratio,price_momentum",
        help="Comma-separated feature names",
    )
    p_train.add_argument("--horizon", type=int, default=5, help="Forward return prediction horizon in bars")
    p_train.add_argument("--folds", type=int, default=3, help="Number of walk-forward validation folds")
    p_train.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # replay
    p_rep = subparsers.add_parser("replay", help="Run a strategy simulation / replay")
    p_rep.add_argument("--strategy", type=str, default="orderflow_absorption", help="Strategy to simulate")
    p_rep.add_argument(
        "--features",
        type=str,
        default="cvd_zscore,price_momentum,realized_vol",
        help="Features to compute for strategy",
    )
    p_rep.add_argument("--capital", type=float, default=10000.0, help="Initial simulation capital")
    p_rep.add_argument("--bars", type=int, default=1500, help="Number of market bars to simulate")

    # leaderboard
    p_lead = subparsers.add_parser("leaderboard", help="View ranked leaderboard of tracked runs")
    p_lead.add_argument(
        "--sort-by",
        type=str,
        default="dsr",
        choices=["dsr", "sharpe", "accuracy", "ic"],
        help="Metric to rank by",
    )
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
