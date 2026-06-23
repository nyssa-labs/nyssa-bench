from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def export_metrics_csv(summary: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, Any]] = [
        ("episodes", summary.get("episodes", 0)),
        ("success_rate", summary.get("success_rate", 0.0)),
        ("sim_to_real_score", summary.get("sim_to_real_score", 0.0)),
        ("primary_failure_mode", summary.get("primary_failure_mode")),
    ]
    rows.extend((f"metrics.{key}", value) for key, value in sorted(dict(summary.get("metrics", {})).items()))
    rows.extend((f"failure_counts.{key}", value) for key, value in sorted(dict(summary.get("failure_counts", {})).items()))

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)
    return path
