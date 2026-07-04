from nyssa_bench.core.suite import Suite
from nyssa_bench.core.task import TaskSpec
from nyssa_bench.randomization import summarize_randomization, summarize_stressor_support
from nyssa_bench.replay.timeline import episode_timeline
from nyssa_bench.replay.trajectory import state_trajectory
from nyssa_bench.replay.video import write_episode_video
from nyssa_bench.specs import SuiteSpec, TaskSpec as StableTaskSpec
from pathlib import Path


def test_stable_spec_imports():
    assert StableTaskSpec is TaskSpec
    assert SuiteSpec is Suite


def test_randomization_summary():
    summary = summarize_randomization({"lighting": True, "camera_pose": False, "friction_range": [0.3, 1.0]})
    assert summary["enabled_keys"] == ["friction_range", "lighting"]
    support = summarize_stressor_support({"lighting": True, "seed": True}, "maniskill")
    assert support["supported_stressors"] == ["seed"]
    assert support["unsupported_stressors"] == ["lighting"]


def test_replay_helpers():
    suite = Suite.load("tabletop_manipulation_v0")
    assert suite.tasks
    assert episode_timeline.__name__ == "episode_timeline"
    assert state_trajectory.__name__ == "state_trajectory"


def test_video_writer_handles_missing_frames(tmp_path):
    assert write_episode_video([], tmp_path, "task", 0) is None


def test_static_release_artifacts_exist():
    required = [
        "docker/Dockerfile",
        "docker/Dockerfile.maniskill",
        "docker/Dockerfile.mujoco",
        "docker/docker-compose.yml",
        "site/leaderboard/index.html",
        "docs/paper/nyssabench_v0_protocol.md",
        "docs/api_stability.md",
        "docs/plugins.md",
    ]
    for path in required:
        assert Path(path).exists()
