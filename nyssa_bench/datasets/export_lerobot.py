from __future__ import annotations

import json
from pathlib import Path

from nyssa_bench.core.episode import EpisodeResult


def export_lerobot(episodes: list[EpisodeResult], out_dir: str | Path) -> Path:
    """Write a lightweight LeRobot-style manifest and episode index.

    This keeps v0.1 export inspectable without requiring the LeRobot package.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "codebase_version": "nyssa-bench-v0.1",
        "format": "lerobot-compatible-lite",
        "episodes": len(episodes),
        "features": {
            "observation.state": {"dtype": "dict"},
            "action": {"dtype": "json"},
            "reward": {"dtype": "float32"},
        },
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    with (out_dir / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode.to_dict()) + "\n")
    return out_dir
