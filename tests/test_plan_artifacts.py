from nyssa_bench.core.suite import Suite
from nyssa_bench.core.task import TaskSpec
from nyssa_bench.randomization import summarize_randomization
from nyssa_bench.replay.timeline import episode_timeline
from nyssa_bench.replay.trajectory import state_trajectory
from nyssa_bench.specs import SuiteSpec, TaskSpec as StableTaskSpec


def test_stable_spec_imports():
    assert StableTaskSpec is TaskSpec
    assert SuiteSpec is Suite


def test_randomization_summary():
    summary = summarize_randomization({"lighting": True, "camera_pose": False, "friction_range": [0.3, 1.0]})
    assert summary["enabled_keys"] == ["friction_range", "lighting"]


def test_replay_helpers():
    suite = Suite.load("tabletop_manipulation_v0")
    assert suite.tasks
    assert episode_timeline.__name__ == "episode_timeline"
    assert state_trajectory.__name__ == "state_trajectory"
