from __future__ import annotations

from nyssa_bench.core.episode import EpisodeResult


def state_trajectory(episode: EpisodeResult) -> list[dict[str, object]]:
    points = []
    for index, step in enumerate(episode.steps):
        state = dict(step.observation.get("state", {}))
        state["step"] = index
        points.append(state)
    return points
