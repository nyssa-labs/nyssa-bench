from __future__ import annotations

import json
from pathlib import Path

from nyssa_bench.core.episode import EpisodeResult


def export_json(episodes: list[EpisodeResult], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump([episode.to_dict() for episode in episodes], handle, indent=2)
    return path
