from pathlib import Path

import pytest

from nyssa_bench import PolicyRunner, Suite
from nyssa_bench.policies.robomimic_adapter import RoboMimicPolicy


def test_suite_loads_tasks():
    suite = Suite.load("tabletop_manipulation_v0")
    assert suite.suite_id == "tabletop_manipulation_v0"
    assert len(suite.tasks) == 5
    assert suite.tasks[0].task_id == "pick_cube"

    articulated = Suite.load("articulated_object_v0")
    assert len(articulated.tasks) == 10

    maniskill = Suite.load("maniskill_smoke_v0")
    assert maniskill.tasks[0].success["engine_env_ids"]["maniskill"] == "PickCube-v1"

    mujoco = Suite.load("mujoco_control_v0")
    assert mujoco.tasks[0].success["engine_env_ids"]["mujoco"] == "Reacher-v5"


def test_runner_writes_artifacts(tmp_path: Path):
    suite = Suite.load("tabletop_manipulation_v0")
    runner = PolicyRunner(policy="scripted", engine="dummy", episodes=2, seed=123, out=tmp_path)
    report = runner.evaluate(suite)

    assert report.summary["episodes"] == 10
    assert 0.0 <= report.summary["success_rate"] <= 1.0
    assert len(report.summary["success_rate_ci95"]) == 2
    assert report.summary["benchmark_tier"] == "smoke"
    assert report.summary["public_claim"] is False
    assert "pick_cube" in report.summary["per_task"]
    assert "success_rate_ci95" in report.summary["per_task"]["pick_cube"]
    assert 0.0 <= report.summary["sim_to_real_score"] <= 1.0
    assert (tmp_path / "config.yaml").exists()
    assert (tmp_path / "run.yaml").exists()
    assert (tmp_path / "environment.json").exists()
    assert (tmp_path / "package_versions.json").exists()
    assert (tmp_path / "git_info.json").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "episodes.json").exists()
    assert (tmp_path / "episodes.jsonl").exists()
    assert (tmp_path / "replay_manifest.json").exists()
    assert (tmp_path / "replay.html").exists()
    assert (tmp_path / "videos").is_dir()
    assert any((tmp_path / "videos").glob("*.mp4"))
    assert (tmp_path / "failures").is_dir()
    assert (tmp_path / "plots").is_dir()
    assert (tmp_path / "report.html").exists()


def test_runner_loads_duck_typed_policy_file(tmp_path: Path):
    policy_path = tmp_path / "user_policy.py"
    policy_path.write_text(
        """
class PolicyAdapter:
    def act(self, observation):
        return 0.2
""".strip(),
        encoding="utf-8",
    )

    suite = Suite.load("tabletop_manipulation_v0")
    runner = PolicyRunner(policy=str(policy_path), engine="dummy", episodes=1, seed=123)
    report = runner.evaluate(suite)

    assert report.summary["episodes"] == 5


def test_external_policy_fallback_rejects_real_observations():
    policy = RoboMimicPolicy()
    with pytest.raises(RuntimeError, match="NYSSA_ROBOMIMIC_POLICY"):
        policy.act({"raw": [0.0, 1.0]})
