from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nyssa_bench import PolicyRunner, Suite
from nyssa_bench.engines.base import NyssaEngine
from nyssa_bench.policies.robomimic_adapter import RoboMimicPolicy
from nyssa_bench.plugins import get_plugin_registry


class UnitEngine(NyssaEngine):
    max_steps = 2

    def load_task(self, task_spec: Any) -> None:
        self.task_spec = task_spec

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        return _observation(), {"seed": seed}

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        return _observation(), 1.0, True, False, {
            "success": True,
            "completion_time": 1.0,
            "path_efficiency": 1.0,
            "grasp_success": True,
        }

    def render(self) -> Any:
        return np.zeros((32, 32, 3), dtype=np.uint8)

    def get_state(self) -> dict[str, Any]:
        return {}

    def close(self) -> None:
        return None


def _observation() -> dict[str, Any]:
    return {
        "raw": [0.0],
        "action_space": {
            "type": "box",
            "shape": [1],
            "low": [-1.0],
            "high": [1.0],
            "dtype": "float32",
        },
    }


def _register_unit_engine() -> None:
    get_plugin_registry().engines["unit_real"] = UnitEngine


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
    _register_unit_engine()
    suite = Suite.load("tabletop_manipulation_v0")
    runner = PolicyRunner(policy="random", engine="unit_real", episodes=2, seed=123, out=tmp_path)
    report = runner.evaluate(suite)

    assert report.summary["episodes"] == 10
    assert 0.0 <= report.summary["success_rate"] <= 1.0
    assert len(report.summary["success_rate_ci95"]) == 2
    assert report.summary["benchmark_tier"] == "real"
    assert report.summary["public_claim"] is True
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
    _register_unit_engine()
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
    runner = PolicyRunner(policy=str(policy_path), engine="unit_real", episodes=1, seed=123)
    report = runner.evaluate(suite)

    assert report.summary["episodes"] == 5


def test_external_policy_requires_real_model():
    with pytest.raises(RuntimeError, match="NYSSA_ROBOMIMIC_POLICY"):
        RoboMimicPolicy()
