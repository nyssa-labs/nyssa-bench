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


def _observation_with_action_size(size: int, raw: list[float] | None = None) -> dict[str, Any]:
    return {
        "raw": raw if raw is not None else [0.0],
        "action_space": {"type": "box", "shape": [size], "low": [-1.0] * size, "high": [1.0] * size},
    }


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


def test_cli_run_filters_tasks(tmp_path: Path):
    _register_cli_engine()
    run_dir = tmp_path / "filtered_run"

    assert (
        main(
            [
                "run",
                "--suite",
                "warehouse_manipulation_v0",
                "--tasks",
                "pick_from_bin",
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

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["episodes"] == 1
    assert set(metrics["per_task"]) == {"pick_from_bin"}


def test_cli_robomimic_export_with_dataset_extra(tmp_path: Path):
    pytest.importorskip("h5py")
    import h5py

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
    robomimic_path = run_dir / "robomimic.hdf5"
    assert robomimic_path.exists()
    with h5py.File(robomimic_path, "r") as handle:
        env_args = json.loads(handle["data"].attrs["env_args"])
    assert env_args["env_name"] == "NyssaFlat-v0"
    assert "env_kwargs" in env_args


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


def test_cli_ablate_writes_variant_pack(tmp_path: Path):
    _register_cli_engine()
    out = tmp_path / "ablation"

    assert (
        main(
            [
                "ablate",
                "--suite",
                "maniskill_smoke_v0",
                "--engine",
                "cli_real",
                "--policy",
                "random",
                "--seeds",
                "0",
                "--episodes",
                "1",
                "--variants",
                "base",
                "verifier",
                "--out",
                str(out),
                "--no-replay",
            ]
        )
        == 0
    )

    assert (out / "manifest.json").exists()
    assert (out / "RESULTS.md").exists()
    assert (out / "base" / "seed_0" / "metrics.json").exists()
    assert (out / "verifier" / "seed_0" / "metrics.json").exists()


def test_cli_ablate_filters_tasks(tmp_path: Path):
    _register_cli_engine()
    out = tmp_path / "filtered_ablation"

    assert (
        main(
            [
                "ablate",
                "--suite",
                "warehouse_manipulation_v0",
                "--tasks",
                "pick_from_bin",
                "--engine",
                "cli_real",
                "--policy",
                "random",
                "--seeds",
                "0",
                "--episodes",
                "1",
                "--variants",
                "base",
                "--out",
                str(out),
                "--no-replay",
            ]
        )
        == 0
    )

    metrics = json.loads((out / "base" / "seed_0" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["episodes"] == 1
    assert set(metrics["per_task"]) == {"pick_from_bin"}


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


def test_cli_train_recovery_bc_from_ablation_root(tmp_path: Path):
    root = tmp_path / "ablation"
    recovery_dir = root / "verifier_recovery" / "seed_0" / "recovery_dataset"
    recovery_dir.mkdir(parents=True)
    recovery_dir.joinpath("episodes.json").write_text(
        json.dumps(
            [
                {
                    "task_id": "mujoco_reacher",
                    "episode_index": 0,
                    "seed": 0,
                    "success": False,
                    "failure_label": "low_reward",
                    "steps": [
                        {
                            "step_index": 0,
                            "observation": _observation(),
                            "action": [0.4],
                            "reward": 0.0,
                            "terminated": False,
                            "truncated": False,
                            "info": {"expert_intervention": True},
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "recovery_bc_policy.json"
    merged_out = tmp_path / "merged_recovery.json"

    assert main(["train-recovery-bc", str(root), "--out", str(out), "--merged-out", str(merged_out)]) == 0
    assert out.exists()
    merged = json.loads(merged_out.read_text(encoding="utf-8"))
    assert len(merged) == 1
    assert merged[0]["task_id"] == "mujoco_reacher"


def test_cli_train_recovery_bc_by_task(tmp_path: Path):
    root = tmp_path / "run"
    recovery_dir = root / "recovery_dataset"
    recovery_dir.mkdir(parents=True)
    recovery_dir.joinpath("episodes.json").write_text(
        json.dumps(
            [
                {"task_id": "mujoco_reacher", "steps": [{"observation": _observation(), "action": [0.1]}]},
                {"task_id": "maniskill_pick_cube_joint", "steps": [{"observation": _observation(), "action": [0.2]}]},
            ]
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "bc_by_task"

    assert main(["train-recovery-bc", str(root), "--by-task", "--out-dir", str(out_dir)]) == 0
    assert (out_dir / "mujoco_reacher.json").exists()
    assert (out_dir / "maniskill_pick_cube.json").exists()


def test_cli_train_recovery_bc_auto_routes_mixed_action_spaces_by_task(tmp_path: Path):
    root = tmp_path / "run"
    recovery_dir = root / "recovery_dataset"
    recovery_dir.mkdir(parents=True)
    recovery_dir.joinpath("episodes.json").write_text(
        json.dumps(
            [
                {
                    "task_id": "mujoco_reacher",
                    "steps": [{"observation": _observation_with_action_size(2), "action": [0.1, 0.2]}],
                },
                {
                    "task_id": "mujoco_inverted_pendulum",
                    "steps": [{"observation": _observation_with_action_size(1), "action": [0.3]}],
                },
            ]
        ),
        encoding="utf-8",
    )
    global_out = tmp_path / "global.json"
    out_dir = tmp_path / "bc_by_task"

    assert main(["train-recovery-bc", str(root), "--out", str(global_out), "--out-dir", str(out_dir)]) == 0
    assert not global_out.exists()
    assert (out_dir / "mujoco_reacher.json").exists()
    assert (out_dir / "mujoco_inverted_pendulum.json").exists()


def test_cli_train_recovery_bc_global_rejects_mixed_action_spaces(tmp_path: Path):
    root = tmp_path / "run"
    recovery_dir = root / "recovery_dataset"
    recovery_dir.mkdir(parents=True)
    recovery_dir.joinpath("episodes.json").write_text(
        json.dumps(
            [
                {
                    "task_id": "mujoco_reacher",
                    "steps": [{"observation": _observation_with_action_size(2), "action": [0.1, 0.2]}],
                },
                {
                    "task_id": "mujoco_inverted_pendulum",
                    "steps": [{"observation": _observation_with_action_size(1), "action": [0.3]}],
                },
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mixed action sizes"):
        main(["train-recovery-bc", str(root), "--routing", "global", "--out", str(tmp_path / "global.json")])


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


def test_robomimic_policy_flattens_and_clips_action():
    import numpy as np

    from nyssa_bench.policies.robomimic_adapter import RoboMimicPolicy

    class DummyRoboMimic:
        def __init__(self) -> None:
            self.started = False
            self.last_obs = None

        def start_episode(self) -> None:
            self.started = True

        def get_action(self, obs):
            self.last_obs = obs
            return np.asarray([2.0, -2.0, 0.5])

    model = DummyRoboMimic()
    policy = RoboMimicPolicy(model=model)
    policy.reset()
    observation = {
        "raw": {"x": [1.0, 2.0]},
        "action_space": {
            "type": "box",
            "shape": [2],
            "low": [-1.0, -1.0],
            "high": [1.0, 1.0],
        },
    }

    action = policy.act(observation)

    assert model.started is True
    assert set(model.last_obs) == {"flat"}
    assert model.last_obs["flat"].shape == (256,)
    assert action.tolist() == [1.0, -1.0]


def test_task_routed_linear_bc_uses_task_checkpoint(tmp_path: Path):
    import numpy as np

    from nyssa_bench.baselines.simple_bc import LinearBCPolicy, TaskRoutedLinearBCPolicy

    checkpoint_dir = tmp_path / "checkpoints"
    LinearBCPolicy(
        weights=np.zeros((4, 2), dtype=float),
        bias=np.asarray([0.25, -0.25], dtype=float),
        feature_dim=4,
        action_size=2,
    ).save(checkpoint_dir / "maniskill_pick_cube.json")
    policy = TaskRoutedLinearBCPolicy(checkpoint_dir)
    task = type("Task", (), {"task_id": "maniskill_pick_cube_joint"})()
    policy.reset(task=task)
    observation = {
        "raw": [0.0],
        "action_space": {
            "type": "box",
            "shape": [2],
            "low": [-1.0, -1.0],
            "high": [1.0, 1.0],
        },
    }

    action = policy.predict_action(observation)

    assert action.tolist() == [0.25, -0.25]


def test_task_routed_linear_bc_can_zero_fill_missing_task(tmp_path: Path):
    from nyssa_bench.baselines.simple_bc import TaskRoutedLinearBCPolicy

    policy = TaskRoutedLinearBCPolicy(tmp_path / "missing_checkpoints", missing_task="zero")
    task = type("Task", (), {"task_id": "mujoco_inverted_pendulum"})()
    policy.reset(task=task)
    observation = {
        "raw": [0.0],
        "action_space": {
            "type": "box",
            "shape": [2],
            "low": [0.2, -1.0],
            "high": [1.0, 1.0],
        },
    }

    action = policy.predict_action(observation)

    assert action.tolist() == [0.2, 0.0]


def test_task_routed_bc_loads_knn_checkpoint(tmp_path: Path):
    import numpy as np

    from nyssa_bench.baselines.simple_bc import KNNBCPolicy, TaskRoutedLinearBCPolicy

    checkpoint_dir = tmp_path / "checkpoints"
    KNNBCPolicy(
        features=np.asarray([[0.0, 0.0], [3.0, 3.0]], dtype=float),
        actions=np.asarray([[0.4], [-0.4]], dtype=float),
        feature_mean=np.asarray([0.0, 0.0], dtype=float),
        feature_scale=np.asarray([1.0, 1.0], dtype=float),
        feature_dim=2,
        action_size=1,
        k=1,
    ).save(checkpoint_dir / "maniskill_pick_cube.json")
    policy = TaskRoutedLinearBCPolicy(checkpoint_dir)
    task = type("Task", (), {"task_id": "maniskill_pick_cube_joint"})()
    policy.reset(task=task)
    observation = {
        "raw": [0.1, 0.1],
        "action_space": {
            "type": "box",
            "shape": [1],
            "low": [-1.0],
            "high": [1.0],
        },
    }

    action = policy.predict_action(observation)

    assert action.tolist() == pytest.approx([0.4])


def test_cli_train_bc_knn(tmp_path: Path):
    episodes = tmp_path / "episodes.json"
    episodes.write_text(
        json.dumps(
            [
                {
                    "task_id": "unit",
                    "steps": [
                        {"observation": _observation_with_action_size(1, raw=[0.0, 0.0]), "action": [0.1]},
                        {"observation": _observation_with_action_size(1, raw=[5.0, 5.0]), "action": [0.9]},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "knn.json"

    assert main(["train-bc", str(episodes), "--out", str(out), "--model", "knn", "--feature-dim", "2"]) == 0

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["format"] == "nyssa-knn-bc-v1"
    assert payload["feature_dim"] == 2
    assert len(payload["features"]) == 2


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


def test_cli_imports_maniskill_demos_falls_back_from_empty_obs_to_env_states(tmp_path: Path):
    pytest.importorskip("h5py")
    import h5py
    import numpy as np

    demos = tmp_path / "demos"
    demos.mkdir()
    h5_path = demos / "PushCube-v1.motionplanning.h5"
    with h5py.File(h5_path, "w") as handle:
        handle.attrs["env_id"] = "PushCube-v1"
        traj = handle.create_group("traj_0")
        traj.create_dataset("actions", data=np.zeros((2, 8), dtype=np.float32))
        traj.create_dataset("success", data=np.asarray([False, True]))
        traj.create_group("obs")
        traj.create_dataset("env_states", data=np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    out = tmp_path / "imported"
    assert main(["import-maniskill-demos", "--input", str(demos), "--out", str(out)]) == 0

    episodes = json.loads((out / "episodes.json").read_text(encoding="utf-8"))
    first_raw = episodes[0]["steps"][0]["observation"]["raw"]
    assert episodes[0]["task_id"] == "maniskill_push_cube"
    assert first_raw == {"env_states": [1.0, 2.0, 3.0]}


def test_cli_collects_maniskill_demos_with_template(tmp_path: Path):
    pytest.importorskip("h5py")

    generator = tmp_path / "write_demo.py"
    generator.write_text(
        """
import sys
from pathlib import Path

import h5py
import numpy as np

env_id, raw_task_dir = sys.argv[1], Path(sys.argv[2])
raw_task_dir.mkdir(parents=True, exist_ok=True)
with h5py.File(raw_task_dir / f"{env_id}.h5", "w") as handle:
    handle.attrs["env_id"] = env_id
    traj = handle.create_group("traj_0")
    traj.attrs["episode_seed"] = 3
    traj.create_dataset("actions", data=np.zeros((2, 3), dtype=np.float32))
    traj.create_dataset("success", data=np.asarray([False, True]))
""".strip(),
        encoding="utf-8",
    )
    raw_dir = tmp_path / "raw"
    out = tmp_path / "imported"
    template = f"python {generator} {{env_id}} {{raw_task_dir}}"

    assert (
        main(
            [
                "collect-maniskill-demos",
                "--env-ids",
                "PickCube-v1",
                "--num-traj",
                "1",
                "--raw-dir",
                str(raw_dir),
                "--out",
                str(out),
                "--command-template",
                template,
            ]
        )
        == 0
    )

    episodes = json.loads((out / "episodes.json").read_text(encoding="utf-8"))
    collect_manifest = json.loads((out / "collect_manifest.json").read_text(encoding="utf-8"))
    assert len(episodes) == 1
    assert episodes[0]["task_id"] == "maniskill_pick_cube"
    assert collect_manifest["failure_count"] == 0
    assert "PickCube-v1" in collect_manifest["runs"][0]["command"]


def test_cli_writes_robomimic_config(tmp_path: Path):
    data = tmp_path / "data.hdf5"
    data.write_bytes(b"")
    out = tmp_path / "robomimic.json"

    assert (
        main(
            [
                "write-robomimic-config",
                "--data",
                str(data),
                "--out",
                str(out),
                "--epochs",
                "3",
                "--batch-size",
                "8",
            ]
        )
        == 0
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["train"]["data"] == str(data.resolve())
    assert Path(payload["train"]["output_dir"]).is_absolute()
    assert payload["train"]["num_epochs"] == 3
    assert payload["train"]["batch_size"] == 8
    assert payload["train"]["hdf5_normalize_obs"] is False
    assert payload["observation"]["modalities"]["obs"]["low_dim"] == ["flat"]
    assert payload["experiment"]["rollout"]["enabled"] is False


def test_scripts_smoke(tmp_path: Path):
    from scripts.release_checklist import main as release_checklist
    from scripts.validate_backend import main as validate_backend
    from scripts.validate_configs import main as validate_configs

    assert validate_configs() == 0
    assert validate_backend(["robocasa"]) == 0
    assert validate_backend(["genesis"]) == 0
    assert release_checklist() == 0
