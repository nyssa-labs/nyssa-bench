from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from nyssa_bench.metrics.sim_to_real import score_summary


def load_run_summary(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "config.yaml"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Run metrics not found: {metrics_path}")

    summary = json.loads(metrics_path.read_text(encoding="utf-8"))
    summary.setdefault("run_dir", str(run_dir))
    if config_path.exists():
        summary["config_path"] = str(config_path)
    summary["sim_to_real_score"] = float(summary.get("sim_to_real_score", score_summary(summary)))
    return summary


def compare_runs(run_dirs: list[str | Path]) -> dict[str, Any]:
    runs = [load_run_summary(path) for path in run_dirs]
    ranked = sorted(
        runs,
        key=lambda item: (
            float(item.get("success_rate", 0.0)),
            float(item.get("sim_to_real_score", 0.0)),
        ),
        reverse=True,
    )
    return {
        "runs": runs,
        "ranking": [
            {
                "rank": index + 1,
                "run_dir": item.get("run_dir"),
                "success_rate": item.get("success_rate", 0.0),
                "success_rate_ci95": item.get("success_rate_ci95", [0.0, 0.0]),
                "sim_to_real_score": item.get("sim_to_real_score", 0.0),
                "benchmark_tier": item.get("benchmark_tier", "unknown"),
                "public_claim": item.get("public_claim", False),
                "primary_failure_mode": item.get("primary_failure_mode"),
            }
            for index, item in enumerate(ranked)
        ],
    }


def save_comparison_report(comparison: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_comparison_html(comparison), encoding="utf-8")
    return path


def save_leaderboard(comparison: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(comparison["ranking"], indent=2), encoding="utf-8")
    return path


def _comparison_html(comparison: dict[str, Any]) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{item['rank']}</td>"
        f"<td>{html.escape(str(item['run_dir']))}</td>"
        f"<td>{float(item['success_rate']) * 100:.1f}%</td>"
        f"<td>{html.escape(_format_ci_percent(item.get('success_rate_ci95')))}</td>"
        f"<td>{float(item['sim_to_real_score']):.3f}</td>"
        f"<td>{html.escape(str(item.get('benchmark_tier') or 'unknown'))}</td>"
        f"<td>{html.escape(str(item.get('primary_failure_mode') or 'none'))}</td>"
        "</tr>"
        for item in comparison["ranking"]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>NyssaBench Policy Comparison</title>
  <style>
    body {{ font-family: Inter, Arial, sans-serif; margin: 40px; color: #17202a; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #d8dee4; padding: 8px; text-align: left; }}
  </style>
</head>
<body>
  <h1>Policy Comparison</h1>
  <table>
    <thead>
      <tr><th>Rank</th><th>Run</th><th>Success</th><th>95% CI</th><th>Sim-to-real</th><th>Tier</th><th>Primary failure</th></tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
"""


def _format_ci_percent(value: Any) -> str:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return "n/a"
    return f"{float(value[0]) * 100:.1f}-{float(value[1]) * 100:.1f}%"
