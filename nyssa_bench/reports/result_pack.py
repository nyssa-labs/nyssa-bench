from __future__ import annotations

from pathlib import Path
from typing import Any


def write_results_markdown(
    *,
    out_dir: str | Path,
    suite_id: str,
    engine: str,
    policies: list[str],
    seeds: list[int],
    episodes_per_task: int,
    run_dirs: list[Path],
    comparison_report: Path,
    leaderboard: Path,
    scorecard: Path,
) -> Path:
    out_dir = Path(out_dir)
    path = out_dir / "RESULTS.md"
    run_lines = "\n".join(f"- `{run_dir.as_posix()}`" for run_dir in run_dirs)
    policy_lines = "\n".join(f"- `{policy}`" for policy in policies)
    seed_text = ", ".join(str(seed) for seed in seeds)
    text = f"""# NyssaBench Results

## Run Definition

- Suite: `{suite_id}`
- Engine: `{engine}`
- Episodes per task per seed: `{episodes_per_task}`
- Seeds: `{seed_text}`

## Policies

{policy_lines}

## Artifacts

- Comparison report: `{comparison_report.as_posix()}`
- Leaderboard: `{leaderboard.as_posix()}`
- Scorecard: `{scorecard.as_posix()}`

## Run Directories

{run_lines}

## Interpretation Notes

- `random` is a weak sanity baseline.
- `scripted_oracle` should be treated as an upper-bound/control baseline only when backed by a real task-specific oracle factory.
- `bc_policy` should be treated as a learned baseline only when backed by a real checkpoint or factory.
- Public benchmark claims require each run's `public_claim_validation` to pass.
- Unsupported stressors are reported in each run summary and should not be claimed as active perturbations.
"""
    path.write_text(text, encoding="utf-8")
    return path


def write_experiment_manifest(
    *,
    out_dir: str | Path,
    suite_id: str,
    engine: str,
    policies: list[str],
    seeds: list[int],
    episodes_per_task: int,
    run_dirs: list[Path],
    artifacts: dict[str, Path],
) -> Path:
    import json

    out_dir = Path(out_dir)
    path = out_dir / "manifest.json"
    payload: dict[str, Any] = {
        "suite": suite_id,
        "engine": engine,
        "policies": policies,
        "seeds": seeds,
        "episodes_per_task": episodes_per_task,
        "run_dirs": [run_dir.as_posix() for run_dir in run_dirs],
        "artifacts": {key: value.as_posix() for key, value in artifacts.items()},
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
