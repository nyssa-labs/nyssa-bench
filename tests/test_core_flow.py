from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nyssa_bench import PolicyRunner, Suite
from nyssa_bench.engines.base import NyssaEngine
from nyssa_bench.core.episode import EpisodeResult, StepRecord
from nyssa_bench.experts import ExpertActionScore, ExpertProvider, make_expert_provider
from nyssa_bench.metrics.failure_mapper import FailureMapper
from nyssa_bench.metrics.run_claims import RunClaimValidator
from nyssa_bench.policies.robomimic_adapter import RoboMimicPolicy
from nyssa_bench.plugins import get_plugin_registry
from nyssa_bench.validation.run_claim import RunClaimValidator as StableRunClaimValidator


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

    focused = Suite.load("maniskill_manipulation_v0")
    assert [task.task_id for task in focused.tasks] == [
        "maniskill_pick_cube",
        "maniskill_stack_cube",
        "maniskill_push_cube",
    ]
    assert all(task.success["control_mode"] == "pd_ee_delta_pose" for task in focused.tasks)

    planner_bc = Suite.load("maniskill_planner_bc_v0")
    assert [task.task_id for task in planner_bc.tasks] == [
        "maniskill_pick_cube_joint",
        "maniskill_stack_cube_joint",
        "maniskill_push_cube_joint",
    ]
    assert all(task.success["control_mode"] == "pd_joint_pos" for task in planner_bc.tasks)

    mujoco = Suite.load("mujoco_control_v0")
    assert mujoco.tasks[0].success["engine_env_ids"]["mujoco"] == "Reacher-v5"


def test_mujoco_adapter_falls_back_to_available_gym_version():
    from nyssa_bench.engines.mujoco_adapter import _make_mujoco_env, _mujoco_env_id_candidates

    class VersionNotFound(Exception):
        pass

    class ErrorNamespace:
        pass

    ErrorNamespace.VersionNotFound = VersionNotFound

    class FakeGym:
        error = ErrorNamespace

        def __init__(self) -> None:
            self.requested: list[str] = []

        def make(self, env_id: str, **kwargs: Any) -> dict[str, Any]:
            self.requested.append(env_id)
            if env_id == "Reacher-v5":
                raise VersionNotFound(env_id)
            return {"env_id": env_id, "kwargs": kwargs}

    assert _mujoco_env_id_candidates("Reacher-v5") == ["Reacher-v5", "Reacher-v4", "Reacher-v2"]

    gym = FakeGym()
    env = _make_mujoco_env(gym, "Reacher-v5", {"render_mode": "rgb_array"})

    assert env["env_id"] == "Reacher-v4"
    assert env["kwargs"] == {"render_mode": "rgb_array"}
    assert gym.requested == ["Reacher-v5", "Reacher-v4"]


def test_runner_writes_artifacts(tmp_path: Path):
    _register_unit_engine()
    suite = Suite.load("tabletop_manipulation_v0")
    runner = PolicyRunner(policy="random", engine="unit_real", episodes=2, seed=123, out=tmp_path)
    report = runner.evaluate(suite)

    assert report.summary["episodes"] == 10
    assert 0.0 <= report.summary["success_rate"] <= 1.0
    assert len(report.summary["success_rate_ci95"]) == 2
    assert report.summary["benchmark_tier"] == "prototype"
    assert report.summary["public_claim"] is False
    assert "minimum_episodes_per_task" in report.summary["public_claim_validation"]["failures"]
    assert "pick_cube" in report.summary["per_task"]
    assert "success_rate_ci95" in report.summary["per_task"]["pick_cube"]
    assert report.summary["per_seed"]
    assert 0.0 <= report.summary["prototype_reliability_score"] <= 1.0
    assert report.summary["sim_to_real_score_deprecated"] is True
    assert report.summary["metrics"]["expert_intervention_rate"] == 0.0
    assert report.summary["metrics"]["recovery_success_rate"] == 0.0
    assert report.summary["metrics"]["verifier_rejection_rate"] == 0.0
    assert "unsupported_stressors" in report.summary["stressor_support"]
    assert (tmp_path / "config.yaml").exists()
    assert (tmp_path / "run.yaml").exists()
    assert (tmp_path / "environment.json").exists()
    assert (tmp_path / "package_versions.json").exists()
    assert (tmp_path / "git_info.json").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "episodes.json").exists()
    assert (tmp_path / "episodes.jsonl").exists()
    assert (tmp_path / "dataset_manifest.json").exists()
    assert (tmp_path / "failure_gallery.html").exists()
    assert (tmp_path / "recovery_dataset" / "manifest.json").exists()
    assert (tmp_path / "replay_manifest.json").exists()
    assert (tmp_path / "replay.html").exists()
    assert (tmp_path / "videos").is_dir()
    assert any((tmp_path / "videos").glob("*.mp4"))
    assert (tmp_path / "failures").is_dir()
    assert (tmp_path / "plots").is_dir()
    assert (tmp_path / "report.html").exists()


def test_runner_records_expert_verifier_interventions(tmp_path: Path):
    _register_unit_engine()

    class RejectingExpert(ExpertProvider):
        provider_id = "rejecting_unit"

        def score_action(self, observation, action, *, task, engine=None):
            return ExpertActionScore(accepted=False, confidence=0.1, reason="unit_reject")

        def act(self, observation, *, task, engine=None):
            return 0.0

        def recover(self, *, state, failure, task, engine=None):
            return [0.0]

    suite = Suite.load("maniskill_smoke_v0")
    runner = PolicyRunner(
        policy="random",
        engine="unit_real",
        episodes=1,
        seed=123,
        out=tmp_path,
        expert_provider=RejectingExpert(),
        enable_verifier=True,
        enable_recovery=True,
        capture_replay=False,
    )
    report = runner.evaluate(suite)

    assert report.summary["metrics"]["expert_intervention_rate"] == 1.0
    assert report.summary["metrics"]["verifier_rejection_rate"] == 1.0
    assert report.summary["metrics"]["expert_intervention_rate"] <= 1.0
    assert report.summary["metrics"]["recovery_success_rate"] == 1.0
    assert runner.run_metadata["expert_provider"]["provider_id"] == "rejecting_unit"
    episodes = (tmp_path / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    assert '"verifier_rejected": true' in episodes[0]


def test_builtin_expert_providers_emit_actions():
    task = Suite.load("maniskill_smoke_v0").tasks[0]
    observation = _observation()

    bounds = make_expert_provider("bounds-verifier")
    assert bounds.score_action(observation, [2.0], task=task).accepted is False
    assert bounds.act(observation, task=task) is not None

    mujoco = make_expert_provider("mujoco-heuristic")
    assert mujoco.act(observation, task=task) is not None
    assert "recover" in mujoco.metadata()["capabilities"]

    scripted = make_expert_provider("maniskill-scripted")
    assert scripted.act(observation, task=task) is not None
    assert scripted.metadata()["provider_id"] == "maniskill-scripted"


def test_mujoco_expert_uses_rollout_scoring_and_restores_state():
    task = Suite.load("mujoco_control_v0").tasks[0]
    expert = make_expert_provider("mujoco-heuristic")
    engine = _FakeMuJoCoEngine()
    observation = {
        "raw": [1.0, -1.0],
        "action_space": {
            "type": "box",
            "shape": [2],
            "low": [-1.0, -1.0],
            "high": [1.0, 1.0],
        },
    }

    score = expert.score_action(observation, [1.0, 1.0], task=task, engine=engine)
    recovery = expert.recover(state={"observation": observation}, failure="bad_action", task=task, engine=engine)

    assert score.accepted is False
    assert score.reason == "lower_than_candidate_reward"
    assert recovery is not None
    assert np.asarray(recovery[0]).tolist() == [-1.0, 1.0]
    assert engine.env.unwrapped.data.qpos.tolist() == [0.0, 0.0]
    assert engine.env.unwrapped.data.qvel.tolist() == [0.0, 0.0]
    assert engine.elapsed_steps == 0
    assert engine.episode_return == 0.0


def test_runner_executes_action_chunks(tmp_path: Path):
    class TwoStepEngine(UnitEngine):
        max_steps = 2

        def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
            self.elapsed = 0
            return _observation(), {"seed": seed}

        def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
            self.elapsed += 1
            return _observation(), 1.0, self.elapsed >= 2, False, {
                "success": self.elapsed >= 2,
                "completion_time": float(self.elapsed),
                "path_efficiency": 1.0,
                "grasp_success": True,
            }

    get_plugin_registry().engines["chunk_unit"] = TwoStepEngine

    class ChunkPolicy:
        def __init__(self) -> None:
            self.calls = 0

        def act(self, observation):
            self.calls += 1
            return np.asarray([[0.1], [0.2], [0.3]], dtype=float)

    policy = ChunkPolicy()
    suite = Suite.load("maniskill_smoke_v0")
    runner = PolicyRunner(
        policy=policy,
        engine="chunk_unit",
        episodes=1,
        seed=123,
        out=tmp_path,
        capture_replay=False,
        policy_action_horizon=3,
        policy_execution_horizon=2,
    )
    report = runner.evaluate(suite)

    assert policy.calls == 3
    assert report.summary["metrics"]["policy_action_chunk_count"] == 1.0
    assert report.summary["metrics"]["policy_cached_action_count"] == 1.0
    assert runner.episode_results[0].steps[0].info["policy_action_chunk_size"] == 2
    assert runner.episode_results[0].steps[1].info["policy_cached_action"] is True


class _FakeMuJoCoData:
    def __init__(self) -> None:
        self.qpos = np.asarray([0.0, 0.0], dtype=float)
        self.qvel = np.asarray([0.0, 0.0], dtype=float)
        self.time = 0.0


class _FakeMuJoCoUnwrapped:
    def __init__(self) -> None:
        self.data = _FakeMuJoCoData()
        self.model = None

    def set_state(self, qpos, qvel) -> None:
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel


class _FakeMuJoCoEnv:
    def __init__(self) -> None:
        self.unwrapped = _FakeMuJoCoUnwrapped()
        self._elapsed_steps = 0

    def step(self, action):
        action_array = np.asarray(action, dtype=float).reshape(-1)
        self.unwrapped.data.qpos += action_array
        self.unwrapped.data.qvel[:] = action_array
        self.unwrapped.data.time += 1.0
        self._elapsed_steps += 1
        reward = -float(np.linalg.norm(action_array - np.asarray([-1.0, 1.0])))
        return {}, reward, False, False, {}


class _FakeMuJoCoEngine:
    def __init__(self) -> None:
        self.env = _FakeMuJoCoEnv()
        self.episode_return = 0.0
        self.elapsed_steps = 0


def test_public_claim_requires_replay_video_evidence(tmp_path: Path):
    suite = Suite.load("mujoco_control_v0")
    episodes = []
    for task in suite.tasks:
        for index in range(100):
            episodes.append(
                EpisodeResult(
                    task_id=task.task_id,
                    episode_index=index,
                    seed=index,
                    success=False,
                    failure_label="timeout",
                    failure_label_source="mapper",
                    metrics={},
                    steps=[
                        StepRecord(
                            observation={},
                            action=0.0,
                            reward=0.0,
                            terminated=False,
                            truncated=True,
                            info={},
                        )
                    ],
                )
            )

    validation = RunClaimValidator().validate(
        suite=suite,
        engine_name="mujoco",
        episodes_per_task=100,
        episodes=episodes,
        out_dir=tmp_path,
        package_versions={"mujoco": "test"},
        git_info={"commit": "test"},
    )

    assert validation.public_claim is False
    assert "replay_video_evidence" in validation.failures


def test_stable_validation_import_path():
    assert StableRunClaimValidator is RunClaimValidator


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
    from nyssa_bench.policies.openvla_adapter import OpenVLAPolicy

    with pytest.raises(RuntimeError, match="NYSSA_OPENVLA_POLICY"):
        OpenVLAPolicy()

    from nyssa_bench.policies.bc_policy import BCPolicy
    from nyssa_bench.policies.scripted_oracle_policy import ScriptedOraclePolicy

    scripted = ScriptedOraclePolicy()
    action = scripted.act(_observation())
    assert action is not None

    with pytest.raises(RuntimeError, match="BC checkpoint not found"):
        BCPolicy()

    with pytest.raises(RuntimeError, match="RoboMimic checkpoint not found"):
        RoboMimicPolicy()


def test_failure_mapper_does_not_default_to_first_label():
    task = Suite.load("maniskill_smoke_v0").tasks[0]
    classification = FailureMapper().classify({}, task_spec=task, step_count=0)

    assert classification.label == "missed_target"
    assert classification.source == "mapper"

    unknown_task = task.__class__(
        task_id="unknown_failure_task",
        engine=task.engine,
        robot=task.robot,
        scene=task.scene,
        description=task.description,
        success=task.success,
        failure_labels=["bad_grasp"],
    )
    unknown = FailureMapper().classify({}, task_spec=unknown_task, step_count=0)
    assert unknown.label == "unknown_failure"
