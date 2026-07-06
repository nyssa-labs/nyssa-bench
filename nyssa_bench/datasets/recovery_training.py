from __future__ import annotations

import json
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nyssa_bench.baselines.simple_bc import task_checkpoint_key, train_linear_bc


@dataclass(frozen=True)
class RecoveryBCTrainingResult:
    source_paths: list[Path]
    merged_path: Path | None
    checkpoints: dict[str, Path]
    episodes: int
    steps: int


def collect_recovery_episode_paths(sources: list[str | Path]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for source in sources:
        source_path = Path(source)
        candidates: list[Path] = []
        if source_path.is_file():
            candidates.append(source_path)
        elif source_path.is_dir():
            direct = source_path / "recovery_dataset" / "episodes.json"
            if direct.exists():
                candidates.append(direct)
            recovery_dir = source_path / "episodes.json"
            if source_path.name == "recovery_dataset" and recovery_dir.exists():
                candidates.append(recovery_dir)
            candidates.extend(sorted(source_path.glob("**/recovery_dataset/episodes.json")))
        else:
            raise FileNotFoundError(f"Recovery source not found: {source_path}")

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                paths.append(candidate)
                seen.add(resolved)

    if not paths:
        raise FileNotFoundError("No recovery_dataset/episodes.json files found in the provided sources")
    return paths


def load_recovery_episodes(paths: list[str | Path], *, min_steps: int = 1) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Recovery episodes file must contain a list: {path}")
        for item in data:
            if not isinstance(item, dict):
                continue
            steps = item.get("steps", [])
            if not isinstance(steps, list) or len(steps) < min_steps:
                continue
            episode = dict(item)
            episode["steps"] = steps
            episodes.append(episode)
    if not episodes:
        raise ValueError("No recovery episodes with enough steps were found")
    return episodes


def train_recovery_bc(
    sources: list[str | Path],
    *,
    out: str | Path = "checkpoints/recovery_bc_policy.json",
    by_task: bool = False,
    out_dir: str | Path = "checkpoints/bc_by_task",
    merged_out: str | Path | None = None,
    feature_dim: int = 256,
    ridge: float = 1e-3,
    min_steps: int = 1,
) -> RecoveryBCTrainingResult:
    source_paths = collect_recovery_episode_paths(sources)
    episodes = load_recovery_episodes(source_paths, min_steps=min_steps)
    merged_path = _write_json(episodes, merged_out) if merged_out else None
    checkpoints = (
        _train_by_task(episodes, out_dir=out_dir, feature_dim=feature_dim, ridge=ridge)
        if by_task
        else {"global": _train_episodes(episodes, out=out, feature_dim=feature_dim, ridge=ridge)}
    )
    return RecoveryBCTrainingResult(
        source_paths=source_paths,
        merged_path=merged_path,
        checkpoints=checkpoints,
        episodes=len(episodes),
        steps=sum(len(item.get("steps", [])) for item in episodes),
    )


def _train_by_task(
    episodes: list[dict[str, Any]],
    *,
    out_dir: str | Path,
    feature_dim: int,
    ridge: float,
) -> dict[str, Path]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        task_id = str(episode.get("task_id") or "unknown_task")
        grouped[task_id].append(episode)

    checkpoint_dir = Path(out_dir)
    checkpoints: dict[str, Path] = {}
    for task_id, task_episodes in sorted(grouped.items()):
        key = task_checkpoint_key(task_id)
        checkpoints[task_id] = _train_episodes(
            task_episodes,
            out=checkpoint_dir / f"{key}.json",
            feature_dim=feature_dim,
            ridge=ridge,
        )
    return checkpoints


def _train_episodes(
    episodes: list[dict[str, Any]],
    *,
    out: str | Path,
    feature_dim: int,
    ridge: float,
) -> Path:
    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as handle:
        json.dump(episodes, handle)
        episodes_path = Path(handle.name)
    try:
        return train_linear_bc(episodes_path, out, feature_dim=feature_dim, ridge=ridge)
    finally:
        try:
            episodes_path.unlink()
        except OSError:
            pass


def _write_json(episodes: list[dict[str, Any]], out: str | Path | None) -> Path:
    if out is None:
        raise ValueError("merged output path is required")
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(episodes, indent=2) + "\n", encoding="utf-8")
    return path
