from __future__ import annotations

from nyssa_bench.core.episode import EpisodeResult


def episode_timeline(episode: EpisodeResult) -> list[dict[str, object]]:
    return [
        {
            "step": index,
            "reward": step.reward,
            "terminated": step.terminated,
            "truncated": step.truncated,
            "failure_label": step.info.get("failure_label"),
        }
        for index, step in enumerate(episode.steps)
    ]
