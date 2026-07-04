from pathlib import Path
from typing import Any
import json

import pytest

from nyssa_bench.cli import main
from nyssa_bench.engines.base import NyssaEngine
from nyssa_bench.plugins import get_plugin_registry


class CliUnitEngine(NyssaEngine):
    max_steps = 1

    def load_task(self, task_spec: Any) -> None:
        self.task_spec = task_spec

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        return _observation(), {"seed": seed}

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        return _observation(), 1.0, True, False, {"success": True, "completion_time": 1.0, "path_efficiency": 1.0}

    def render(self) -> Any:
        return None

    def get_state(self) -> dict[str, Any]:
        return {}

    def close(self) -> None:
        return None


def _observation() -> dict[str, Any]:
    return {"raw": [0.0], "action_space": {"type": "box", "shape": [1], "low": [-1.0], "high": [1.0]}}


def _register_cli_engine() -> None:
    get_plugin_registry().engines["cli_real"] = CliUnitEngine


def test_cli_lists_and_validates():
    assert main(["list-suites"]) == 0
    assert main(["list-tasks"]) == 0
    assert main(["list-engines"]) == 0
    assert main(["list-policies"]) == 0
    assert main(["validate", "tabletop_manipulation_v0"]) == 0
    assert main(["validate", "pick_cube"]) == 0


def test_cli_run_and_export(tmp_path: Path):
    _register_cli_engine()
    run_dir = tmp_path / "run"
    other_run_dir = tmp_path / "other_run"

    assert main(
        [
            "run",
            "--suite",
            "warehouse_manipulation_v0",
            "--engine",
            "cli_real",
            "--policy",
            "random",
            "--episodes",
            "1",
            "--out",
            str(run_dir),
        ]
    ) == 0
    assert (run_dir / "report.html").exists()
    assert main(["report", str(run_dir)]) == 0

    assert main(["export", "--run", str(run_dir), "--format", "lerobot"]) == 0
    assert (run_dir / "lerobot" / "meta.json").exists()
    assert main(["export", "--run", str(run_dir), "--format", "jsonl"]) == 0
    assert (run_dir / "episodes.export.jsonl").exists()

    assert main(
        [
            "run",
            "--suite",
            "warehouse_manipulation_v0",
            "--engine",
            "cli_real",
            "--policy",
            "random",
            "--episodes",
            "1",
            "--out",
            str(other_run_dir),
        ]
    ) == 0
    assert main(["compare", str(run_dir), str(other_run_dir), "--out", str(tmp_path / "compare.html")]) == 0
    assert main(["leaderboard", str(run_dir), str(other_run_dir), "--out", str(tmp_path / "leaderboard.json")]) == 0
    assert (tmp_path / "compare.html").exists()
    assert (tmp_path / "leaderboard.json").exists()


def test_cli_robomimic_export_with_dataset_extra(tmp_path: Path):
    pytest.importorskip("h5py")
    _register_cli_engine()
    run_dir = tmp_path / "run"

    assert (
        main(
            [
                "run",
                "--suite",
                "warehouse_manipulation_v0",
                "--engine",
                "cli_real",
                "--policy",
                "random",
                "--episodes",
                "1",
                "--out",
                str(run_dir),
            ]
        )
        == 0
    )
    assert main(["export", "--run", str(run_dir), "--format", "robomimic"]) == 0
    assert (run_dir / "robomimic.hdf5").exists()


def test_cli_experiment_writes_result_pack(tmp_path: Path):
    _register_cli_engine()
    out = tmp_path / "experiment"

    assert (
        main(
            [
                "experiment",
                "--suite",
                "maniskill_manipulation_v0",
                "--engine",
                "cli_real",
                "--policies",
                "random",
                "--seeds",
                "0",
                "1",
                "--episodes",
                "1",
                "--out",
                str(out),
            ]
        )
        == 0
    )

    assert (out / "manifest.json").exists()
    assert (out / "RESULTS.md").exists()
    assert (out / "comparison.html").exists()
    assert (out / "leaderboard.json").exists()
    assert (out / "scorecard.json").exists()
    assert (out / "random" / "seed_0" / "metrics.json").exists()
    assert (out / "random" / "seed_1" / "metrics.json").exists()


def test_cli_train_bc_writes_checkpoint(tmp_path: Path):
    episodes = tmp_path / "episodes.json"
    episodes.write_text(
        json.dumps(
            [
                {
                    "steps": [
                        {
                            "observation": _observation(),
                            "action": [0.2],
                        }
                    ]
                }
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "bc_policy.json"

    assert main(["train-bc", str(episodes), "--out", str(out)]) == 0
    assert out.exists()


def test_linear_bc_resizes_action_to_live_action_space():
    import numpy as np

    from nyssa_bench.baselines.simple_bc import LinearBCPolicy

    policy = LinearBCPolicy(
        weights=np.zeros((4, 8), dtype=float),
        bias=np.arange(8, dtype=float),
        feature_dim=4,
        action_size=8,
    )
    observation = {
        "raw": [0.0],
        "action_space": {
            "type": "box",
            "shape": [7],
            "low": [-10.0] * 7,
            "high": [10.0] * 7,
        },
    }

    action = policy.predict_action(observation)

    assert action.shape == (7,)
    assert action.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_cli_imports_maniskill_demos(tmp_path: Path):
    pytest.importorskip("h5py")
    import h5py
    import numpy as np

    demos = tmp_path / "demos"
    demos.mkdir()
    h5_path = demos / "PickCube-v1.motionplanning.h5"
    with h5py.File(h5_path, "w") as handle:
        handle.attrs["env_id"] = "PickCube-v1"
        traj = handle.create_group("traj_0")
        traj.attrs["episode_seed"] = 7
        traj.create_dataset("actions", data=np.zeros((3, 4), dtype=np.float32))
        traj.create_dataset("success", data=np.asarray([False, False, True]))
        obs = traj.create_group("obs")
        obs.create_dataset("agent_qpos", data=np.zeros((4, 9), dtype=np.float32))

    out = tmp_path / "imported"
    assert main(["import-maniskill-demos", "--input", str(demos), "--out", str(out)]) == 0

    episodes = json.loads((out / "episodes.json").read_text(encoding="utf-8"))
    assert len(episodes) == 1
    assert episodes[0]["task_id"] == "maniskill_pick_cube"
    assert episodes[0]["seed"] == 7
    assert episodes[0]["success"] is True
    assert len(episodes[0]["steps"]) == 3
    assert episodes[0]["steps"][0]["observation"]["action_space"]["shape"] == [4]
    assert (out / "maniskill_pick_cube" / "episodes.json").exists()


def test_scripts_smoke(tmp_path: Path):
    from scripts.release_checklist import main as release_checklist
    from scripts.validate_backend import main as validate_backend
    from scripts.validate_configs import main as validate_configs

    assert validate_configs() == 0
    assert validate_backend(["robocasa"]) == 0
    assert validate_backend(["genesis"]) == 0
    assert release_checklist() == 0
