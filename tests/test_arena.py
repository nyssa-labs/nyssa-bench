from nyssa_bench.arena import compare_episode_pairs
from nyssa_bench.core.episode import EpisodeResult


def test_compare_episode_pairs_counts_wins_and_failure_deltas():
    policy_a = [
        _episode("pick", 0, 0, success=True),
        _episode("pick", 1, 0, success=False, failure_label="timeout"),
        _episode("stack", 0, 0, success=False, failure_label="bad_grasp"),
    ]
    policy_b = [
        _episode("pick", 0, 0, success=False, failure_label="timeout"),
        _episode("pick", 1, 0, success=False, failure_label="timeout"),
        _episode("stack", 0, 0, success=True),
        _episode("stack", 99, 0, success=True),
    ]

    summary = compare_episode_pairs(policy_a, policy_b, policy_a_label="a", policy_b_label="b")

    assert summary.total_pairs == 3
    assert summary.wins == {"a": 1, "tie_failure": 1, "b": 1}
    assert summary.failure_deltas == {"b:timeout": 1, "a:bad_grasp": 1}
    assert summary.outcomes[0].winner == "a"


def _episode(task_id: str, seed: int, episode_index: int, *, success: bool, failure_label: str | None = None):
    return EpisodeResult(
        task_id=task_id,
        episode_index=episode_index,
        seed=seed,
        success=success,
        failure_label=failure_label,
        failure_label_source="mapper" if failure_label else None,
        metrics={},
    )

