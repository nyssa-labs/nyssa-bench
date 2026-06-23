from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from nyssa_bench.core.registry import ENGINE_REGISTRY, POLICY_REGISTRY
from nyssa_bench.core.suite import Suite, list_suites
from nyssa_bench.core.task import TaskSpec, list_tasks
from nyssa_bench.datasets.export_lerobot import export_lerobot
from nyssa_bench.reports.comparison import compare_runs, save_comparison_report, save_leaderboard
from nyssa_bench.runner import PolicyRunner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nyssa")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-suites")
    subparsers.add_parser("list-tasks")
    subparsers.add_parser("list-engines")
    subparsers.add_parser("list-policies")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--suite", required=True)
    run_parser.add_argument("--engine", default="dummy")
    run_parser.add_argument("--policy", default="random")
    run_parser.add_argument("--episodes", type=int, default=10)
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--out", required=True)

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("run")

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--run", required=True)
    export_parser.add_argument("--format", choices=["lerobot"], default="lerobot")

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("runs", nargs="+")
    compare_parser.add_argument("--out", required=True)

    leaderboard_parser = subparsers.add_parser("leaderboard")
    leaderboard_parser.add_argument("runs", nargs="+")
    leaderboard_parser.add_argument("--out", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("target")

    args = parser.parse_args(argv)

    if args.command == "list-suites":
        for suite in list_suites():
            print(suite)
        return 0

    if args.command == "list-tasks":
        for task in list_tasks():
            print(task)
        return 0

    if args.command == "list-engines":
        for engine in sorted(ENGINE_REGISTRY):
            print(engine)
        return 0

    if args.command == "list-policies":
        for policy in sorted(POLICY_REGISTRY):
            print(policy)
        return 0

    if args.command == "run":
        suite = Suite.load(args.suite)
        runner = PolicyRunner(
            policy=args.policy,
            engine=args.engine,
            episodes=args.episodes,
            seed=args.seed,
            out=args.out,
        )
        report = runner.evaluate(suite)
        print(f"report: {Path(args.out) / 'report.html'}")
        print(f"success_rate: {report.summary.get('success_rate', 0.0):.3f}")
        return 0

    if args.command == "report":
        run_dir = Path(args.run)
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Run metrics not found: {metrics_path}")
        print(metrics_path.read_text(encoding="utf-8"))
        return 0

    if args.command == "export":
        from nyssa_bench.core.episode import EpisodeResult

        run_dir = Path(args.run)
        episodes_path = run_dir / "episodes.json"
        data = json.loads(episodes_path.read_text(encoding="utf-8"))
        episodes = [
            EpisodeResult(
                task_id=item["task_id"],
                episode_index=item["episode_index"],
                seed=item["seed"],
                success=item["success"],
                failure_label=item["failure_label"],
                metrics=item["metrics"],
                steps=[],
            )
            for item in data
        ]
        out = export_lerobot(episodes, run_dir / "lerobot")
        print(f"exported: {out}")
        return 0

    if args.command == "compare":
        comparison = compare_runs(args.runs)
        out = save_comparison_report(comparison, args.out)
        print(f"comparison: {out}")
        return 0

    if args.command == "leaderboard":
        comparison = compare_runs(args.runs)
        out = save_leaderboard(comparison, args.out)
        print(f"leaderboard: {out}")
        return 0

    if args.command == "validate":
        _validate_target(args.target)
        print(f"valid: {args.target}")
        return 0

    return 1


def _validate_target(target: str) -> None:
    path = Path(target)
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if "tasks" in data:
            Suite.load(path)
            return
        TaskSpec.load(path)
        return

    try:
        Suite.load(target)
    except FileNotFoundError:
        TaskSpec.load(target)


if __name__ == "__main__":
    raise SystemExit(main())
