from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from nyssa_bench.core.episode import EpisodeResult


def write_episode_video(frames: list[Any], out_dir: str | Path, task_id: str, episode_index: int, fps: int = 20) -> str | None:
    if not frames:
        return None

    try:
        import imageio.v3 as iio
    except ImportError:
        return None

    path = Path(out_dir) / "videos" / f"{task_id}_episode_{episode_index:06d}.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        iio.imwrite(path, frames, fps=fps)
    except Exception:
        return None
    return path.relative_to(Path(out_dir)).as_posix()


def write_failure_clip(out_dir: str | Path, episode: EpisodeResult) -> str | None:
    if episode.success or not episode.replay_path:
        return None

    out_dir = Path(out_dir)
    source = out_dir / episode.replay_path
    if not source.exists():
        return None

    label = episode.failure_label or "failure"
    target = out_dir / "failures" / f"{label}_{episode.task_id}_{episode.episode_index:06d}.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    episode.failure_clip_path = target.relative_to(out_dir).as_posix()
    return episode.failure_clip_path


def write_replay_manifest(episodes: list[EpisodeResult], out_dir: str | Path) -> Path:
    """Write replay metadata and any available MP4 paths."""

    out_dir = Path(out_dir)
    videos_dir = out_dir / "videos"
    failures_dir = out_dir / "failures"
    videos_dir.mkdir(parents=True, exist_ok=True)
    failures_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "format": "nyssa-replay-lite",
        "video_export": "available" if any(episode.replay_path for episode in episodes) else "unavailable_for_current_engine",
        "episodes": [
            {
                "task_id": episode.task_id,
                "episode_index": episode.episode_index,
                "success": episode.success,
                "failure_label": episode.failure_label,
                "steps": len(episode.steps),
                "replay_path": episode.replay_path,
                "failure_clip_path": episode.failure_clip_path,
            }
            for episode in episodes
        ],
    }
    path = out_dir / "replay_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
