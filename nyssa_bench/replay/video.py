from __future__ import annotations

import json
from pathlib import Path

from nyssa_bench.core.episode import EpisodeResult


def write_replay_manifest(episodes: list[EpisodeResult], out_dir: str | Path) -> Path:
    """Write replay metadata even when a backend cannot export MP4 frames."""

    out_dir = Path(out_dir)
    videos_dir = out_dir / "videos"
    failures_dir = out_dir / "failures"
    videos_dir.mkdir(parents=True, exist_ok=True)
    failures_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "format": "nyssa-replay-lite",
        "video_export": "unavailable_for_current_engine",
        "episodes": [
            {
                "task_id": episode.task_id,
                "episode_index": episode.episode_index,
                "success": episode.success,
                "failure_label": episode.failure_label,
                "steps": len(episode.steps),
            }
            for episode in episodes
        ],
    }
    path = out_dir / "replay_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
