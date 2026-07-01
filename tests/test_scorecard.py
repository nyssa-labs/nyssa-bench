import json
from pathlib import Path

import yaml

from nyssa_bench.cli import main
from nyssa_bench.reports.scorecard import build_scorecard, write_scorecard


def test_build_scorecard_from_real_run_artifacts(tmp_path: Path):
    run_dir = _make_run(tmp_path / "run_a", policy="random")

    scorecard = build_scorecard(
        [run_dir],
        benchmark="Test scorecard",
        scorecard_date="2026-06-29",
        comparison_report=tmp_path / "compare.html",
        leaderboard=tmp_path / "leaderboard.json",
    )

    assert scorecard["benchmark"] == "Test scorecard"
    assert scorecard["date"] == "2026-06-29"
    assert scorecard["public_claim"] is False
    assert scorecard["results"][0]["suite"] == "mujoco_control_v0"
    assert scorecard["results"][0]["policy"] == "random"
    assert scorecard["results"][0]["per_task"]["mujoco_reacher"]["success_count"] == 1
    assert scorecard["next_required_result"]["policy"] == "robomimic_or_diffusion"


def test_write_scorecard_outputs_related_artifacts(tmp_path: Path):
    run_a = _make_run(tmp_path / "run_a", policy="random", success_rate=0.1)
    run_b = _make_run(tmp_path / "run_b", policy="random", success_rate=0.2)
    out = tmp_path / "scorecard.json"
    comparison = tmp_path / "comparison.html"
    leaderboard = tmp_path / "leaderboard.json"

    paths = write_scorecard(
        [run_a, run_b],
        out=out,
        scorecard_date="2026-06-29",
        comparison_report=comparison,
        leaderboard=leaderboard,
    )

    assert paths["scorecard"] == out
    assert out.exists()
    assert comparison.exists()
    assert leaderboard.exists()
    ranking = json.loads(leaderboard.read_text(encoding="utf-8"))
    assert ranking[0]["run_dir"] == run_b.as_posix()


def test_cli_scorecard(tmp_path: Path):
    run_dir = _make_run(tmp_path / "run_a", policy="random")
    out = tmp_path / "scorecard.json"
    comparison = tmp_path / "comparison.html"
    leaderboard = tmp_path / "leaderboard.json"

    assert (
        main(
            [
                "scorecard",
                str(run_dir),
                "--out",
                str(out),
                "--date",
                "2026-06-29",
                "--comparison-out",
                str(comparison),
                "--leaderboard-out",
                str(leaderboard),
            ]
        )
        == 0
    )
    assert json.loads(out.read_text(encoding="utf-8"))["results"][0]["run_dir"] == run_dir.as_posix()


def _make_run(run_dir: Path, *, policy: str, success_rate: float = 0.1) -> Path:
    run_dir.mkdir(parents=True)
    metadata = {
        "run_id": f"mujoco_control_v0_{policy}_test",
        "suite_id": "mujoco_control_v0",
        "task_ids": ["mujoco_reacher"],
        "policy_name": policy,
        "engine_name": "mujoco",
        "episodes_per_task": 10,
        "seed": 42,
        "started_at": "2026-06-29T00:00:00+00:00",
        "finished_at": "2026-06-29T00:01:00+00:00",
    }
    metrics = {
        "episodes": 10,
        "success_count": int(success_rate * 10),
        "success_rate": success_rate,
        "success_rate_ci95": [0.0, 0.3],
        "failure_counts": {"missed_target": 9},
        "primary_failure_mode": "missed_target",
        "prototype_reliability_score": 0.55,
        "score_kind": "prototype_reliability_heuristic",
        "sim_to_real_score_deprecated": True,
        "benchmark_tier": "real",
        "public_claim": True,
        "public_claim_validation": {"status": "validated", "failures": []},
        "per_task": {
            "mujoco_reacher": {
                "episodes": 10,
                "success_count": int(success_rate * 10),
                "success_rate": success_rate,
                "success_rate_ci95": [0.0, 0.3],
                "failure_counts": {"missed_target": 9},
                "primary_failure_mode": "missed_target",
                "metrics": {"completion_time": 1.0},
                "metric_ci95": {"completion_time": [1.0, 1.0]},
            }
        },
        "per_seed": {
            "42": {
                "episodes": 10,
                "success_count": int(success_rate * 10),
                "success_rate": success_rate,
                "success_rate_ci95": [0.0, 0.3],
                "failure_counts": {"missed_target": 9},
                "primary_failure_mode": "missed_target",
            }
        },
    }
    (run_dir / "run.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    for name in [
        "config.yaml",
        "environment.json",
        "package_versions.json",
        "git_info.json",
        "episodes.jsonl",
        "report.html",
    ]:
        (run_dir / name).write_text("{}\n", encoding="utf-8")
    return run_dir
