from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nyssa_bench import PolicyRunner, Suite
from nyssa_bench.core.task import TaskSpec


BACKEND_SUITES = {
    "dummy": "tabletop_manipulation_v0",
    "maniskill": "maniskill_smoke_v0",
    "mujoco": "mujoco_control_v0",
}

EXPERIMENT_CONTRACTS = {
    "robocasa": "configs/experiments/robocasa_kitchen_v0.yaml",
    "genesis": "configs/experiments/genesis_contact_stress_v0.yaml",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a backend smoke validation.")
    parser.add_argument("backend", choices=sorted(set(BACKEND_SUITES) | set(EXPERIMENT_CONTRACTS)))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--out", default=None)
    parser.add_argument("--policy", default="random")
    parser.add_argument("--capture-replay", action="store_true")
    parser.add_argument("--run-experimental", action="store_true")
    args = parser.parse_args(argv)

    if args.backend in EXPERIMENT_CONTRACTS and not args.run_experimental:
        _validate_experiment_contract(args.backend)
        return 0
    if args.backend not in BACKEND_SUITES:
        raise RuntimeError(
            f"{args.backend} has an experiment contract but no runnable suite yet. "
            "Add a suite with concrete task mappings before using --run-experimental."
        )

    suite_id = BACKEND_SUITES[args.backend]
    out = args.out or f"runs/{args.backend}_smoke"
    suite = Suite.load(suite_id)
    PolicyRunner(
        policy=args.policy,
        engine=args.backend,
        episodes=args.episodes,
        seed=7,
        out=out,
        capture_replay=args.capture_replay,
    ).evaluate(suite)
    print(f"{args.backend} smoke passed: {out}")
    return 0


def _validate_experiment_contract(backend: str) -> None:
    path = Path(EXPERIMENT_CONTRACTS[backend])
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if data.get("engine") != backend:
        raise ValueError(f"{path} engine must be {backend}")
    candidate_tasks = data.get("candidate_tasks", [])
    if not candidate_tasks:
        raise ValueError(f"{path} must define candidate_tasks")
    for task_id in candidate_tasks:
        TaskSpec.load(task_id)
    required = data.get("required_mapping", {})
    missing = [key for key, enabled in required.items() if not enabled]
    if missing:
        raise ValueError(f"{path} has disabled required mappings: {', '.join(missing)}")
    print(f"{backend} experiment contract passed: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
