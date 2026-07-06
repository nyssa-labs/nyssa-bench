from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nyssa_bench.core.episode import EpisodeResult


def write_recovery_dataset(episodes: list[EpisodeResult], out_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    recovery_dir = out_dir / "recovery_dataset"
    recovery_dir.mkdir(parents=True, exist_ok=True)

    items = []
    for episode in episodes:
        recovery_steps = []
        for step_index, step in enumerate(episode.steps):
            info = step.info or {}
            if info.get("expert_intervention") or info.get("recovery_attempted") or info.get("recovery_applied"):
                recovery_steps.append(
                    {
                        "step_index": step_index,
                        "observation": step.to_dict()["observation"],
                        "action": step.to_dict()["action"],
                        "reward": step.reward,
                        "terminated": step.terminated,
                        "truncated": step.truncated,
                        "info": step.to_dict()["info"],
                    }
                )
        if recovery_steps:
            items.append(
                {
                    "task_id": episode.task_id,
                    "episode_index": episode.episode_index,
                    "seed": episode.seed,
                    "success": episode.success,
                    "failure_label": episode.failure_label,
                    "steps": recovery_steps,
                }
            )

    manifest: dict[str, Any] = {
        "format": "nyssa-recovery-dataset-v1",
        "episodes": len(items),
        "steps": sum(len(item["steps"]) for item in items),
        "source": "expert_intervention_or_recovery_steps",
    }
    manifest_path = recovery_dir / "manifest.json"
    json_path = recovery_dir / "episodes.json"
    jsonl_path = recovery_dir / "episodes.jsonl"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(items, indent=2) + "\n", encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item) + "\n")
    return {"manifest": manifest_path, "episodes": json_path, "episodes_jsonl": jsonl_path}
