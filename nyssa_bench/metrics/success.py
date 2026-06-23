from __future__ import annotations

from collections import Counter

from nyssa_bench.core.episode import EpisodeResult


def aggregate_episodes(episodes: list[EpisodeResult]) -> dict[str, object]:
    if not episodes:
        return {"episodes": 0, "success_rate": 0.0, "failure_counts": {}}

    failure_counts = Counter(ep.failure_label for ep in episodes if ep.failure_label)
    success_count = sum(1 for ep in episodes if ep.success)
    metric_keys = sorted({key for ep in episodes for key in ep.metrics})
    metric_means = {
        key: sum(float(ep.metrics.get(key, 0.0)) for ep in episodes) / len(episodes)
        for key in metric_keys
    }

    return {
        "episodes": len(episodes),
        "success_rate": success_count / len(episodes),
        "failure_counts": dict(failure_counts),
        "primary_failure_mode": failure_counts.most_common(1)[0][0] if failure_counts else None,
        "metrics": metric_means,
    }
