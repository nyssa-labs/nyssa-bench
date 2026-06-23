from __future__ import annotations

from collections import Counter

from nyssa_bench.core.episode import EpisodeResult


def cluster_failures(episodes: list[EpisodeResult]) -> dict[str, int]:
    return dict(Counter(ep.failure_label for ep in episodes if ep.failure_label))
