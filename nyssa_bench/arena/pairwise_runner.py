from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from nyssa_bench.core.episode import EpisodeResult


@dataclass(frozen=True)
class PairwiseOutcome:
    task_id: str
    seed: int
    episode_index: int
    winner: str
    policy_a_success: bool
    policy_b_success: bool
    policy_a_failure: str | None
    policy_b_failure: str | None


@dataclass(frozen=True)
class PairwiseSummary:
    outcomes: tuple[PairwiseOutcome, ...]
    wins: dict[str, int]
    failure_deltas: dict[str, int]

    @property
    def total_pairs(self) -> int:
        return len(self.outcomes)


def compare_episode_pairs(
    policy_a: list[EpisodeResult],
    policy_b: list[EpisodeResult],
    *,
    policy_a_label: str = "policy_a",
    policy_b_label: str = "policy_b",
) -> PairwiseSummary:
    """Compare matched episodes from two policies.

    Episodes are matched by `(task_id, seed, episode_index)`. Unmatched episodes
    are ignored so callers can compare partial result sets while keeping the
    outcome definition deterministic.
    """

    b_by_key = {_key(episode): episode for episode in policy_b}
    outcomes: list[PairwiseOutcome] = []
    wins: Counter[str] = Counter()
    failure_deltas: Counter[str] = Counter()

    for episode_a in policy_a:
        key = _key(episode_a)
        episode_b = b_by_key.get(key)
        if episode_b is None:
            continue

        winner = _winner(episode_a, episode_b, policy_a_label, policy_b_label)
        wins[winner] += 1
        _count_failure_delta(failure_deltas, episode_a, episode_b, policy_a_label, policy_b_label)
        outcomes.append(
            PairwiseOutcome(
                task_id=episode_a.task_id,
                seed=episode_a.seed,
                episode_index=episode_a.episode_index,
                winner=winner,
                policy_a_success=episode_a.success,
                policy_b_success=episode_b.success,
                policy_a_failure=episode_a.failure_label,
                policy_b_failure=episode_b.failure_label,
            )
        )

    return PairwiseSummary(
        outcomes=tuple(outcomes),
        wins=dict(wins),
        failure_deltas=dict(failure_deltas),
    )


def _key(episode: EpisodeResult) -> tuple[str, int, int]:
    return (episode.task_id, episode.seed, episode.episode_index)


def _winner(episode_a: EpisodeResult, episode_b: EpisodeResult, policy_a_label: str, policy_b_label: str) -> str:
    if episode_a.success and not episode_b.success:
        return policy_a_label
    if episode_b.success and not episode_a.success:
        return policy_b_label
    if episode_a.success and episode_b.success:
        return "tie_success"
    return "tie_failure"


def _count_failure_delta(
    failure_deltas: Counter[str],
    episode_a: EpisodeResult,
    episode_b: EpisodeResult,
    policy_a_label: str,
    policy_b_label: str,
) -> None:
    if episode_a.failure_label and episode_a.failure_label != episode_b.failure_label:
        failure_deltas[f"{policy_a_label}:{episode_a.failure_label}"] += 1
    if episode_b.failure_label and episode_b.failure_label != episode_a.failure_label:
        failure_deltas[f"{policy_b_label}:{episode_b.failure_label}"] += 1

