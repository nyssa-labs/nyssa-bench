from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from nyssa_bench.core.registry import ENGINE_REGISTRY, POLICY_REGISTRY
from nyssa_bench.core.suite import Suite, list_suites
from nyssa_bench.core.task import TaskSpec, list_tasks
from nyssa_bench.datasets.export_hdf5 import export_hdf5
from nyssa_bench.datasets.export_json import export_json
from nyssa_bench.datasets.export_jsonl import export_jsonl
from nyssa_bench.datasets.export_lerobot import export_lerobot
from nyssa_bench.datasets.export_parquet import export_parquet
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
    run_parser.add_argument("--no-replay", action="store_true")

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("run")

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--run", required=True)
    export_parser.add_argument("--format", choices=["json", "jsonl", "lerobot", "hdf5", "parquet"], default="lerobot")
    export_parser.add_argument("--out")

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
            capture_replay=not args.no_replay,
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
        run_dir = Path(args.run)
        episodes = _load_episodes(run_dir)
        out_arg = Path(args.out) if args.out else None
        if args.format == "lerobot":
            out = export_lerobot(episodes, out_arg or run_dir / "lerobot")
        elif args.format == "json":
            out = export_json(episodes, out_arg or run_dir / "episodes.export.json")
        elif args.format == "jsonl":
            out = export_jsonl(episodes, out_arg or run_dir / "episodes.export.jsonl")
        elif args.format == "hdf5":
            out = export_hdf5(episodes, out_arg or run_dir / "episodes.hdf5")
        elif args.format == "parquet":
            out = export_parquet(episodes, out_arg or run_dir / "episodes.parquet")
        else:
            raise ValueError(f"Unsupported export format: {args.format}")
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


def _load_episodes(run_dir: Path):
    from nyssa_bench.core.episode import EpisodeResult, StepRecord

    episodes_path = run_dir / "episodes.json"
    data = json.loads(episodes_path.read_text(encoding="utf-8"))
    episodes = []
    for item in data:
        steps = [
            StepRecord(
                observation=step.get("observation", {}),
                action=step.get("action"),
                reward=float(step.get("reward", 0.0)),
                terminated=bool(step.get("terminated", False)),
                truncated=bool(step.get("truncated", False)),
                info=step.get("info", {}),
            )
            for step in item.get("steps", [])
        ]
        episodes.append(
            EpisodeResult(
                task_id=item["task_id"],
                episode_index=item["episode_index"],
                seed=item["seed"],
                success=item["success"],
                failure_label=item["failure_label"],
                metrics=item["metrics"],
                steps=steps,
                replay_path=item.get("replay_path"),
                failure_clip_path=item.get("failure_clip_path"),
            )
        )
    return episodes


if __name__ == "__main__":
    raise SystemExit(main())
