from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def write_dataset_manifest(
    *,
    out_dir: str | Path,
    suite: Any,
    run_metadata: dict[str, Any],
    artifact_names: list[str],
) -> Path:
    out_dir = Path(out_dir)
    artifacts = {}
    for name in artifact_names:
        path = out_dir / name
        if path.exists() and path.is_file():
            artifacts[name] = {
                "path": name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    payload = {
        "format": "nyssa-dataset-manifest-v1",
        "run": run_metadata,
        "suite": suite.to_dict(),
        "tasks": [
            {
                "task_id": task.task_id,
                "source_path": task.source_path.as_posix() if task.source_path else None,
                "observation": task.observation,
                "action": task.action,
                "goal": task.goal,
                "experts": task.experts,
                "ood_splits": task.ood_splits,
                "randomization": task.randomization,
                "success": task.success,
            }
            for task in suite.tasks
        ],
        "artifacts": artifacts,
        "license": "See repository LICENSE and source dataset licenses.",
    }
    path = out_dir / "dataset_manifest.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
