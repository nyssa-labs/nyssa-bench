from __future__ import annotations

import json
from pathlib import Path

from nyssa_bench.core.episode import EpisodeResult


def export_jsonl(episodes: list[EpisodeResult], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode.to_dict()) + "\n")
    return path
