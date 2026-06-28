from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Report:
    suite_id: str
    policy: str
    engine: str
    summary: dict[str, Any]
    run_dir: Path | None = None

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_html(), encoding="utf-8")
        return path

    def to_html(self) -> str:
        success_rate = float(self.summary.get("success_rate", 0.0)) * 100
        success_ci = self.summary.get("success_rate_ci95", [0.0, 0.0])
        sim_to_real = float(self.summary.get("sim_to_real_score", 0.0))
        primary_failure = self.summary.get("primary_failure_mode") or "none"
        benchmark_tier = self.summary.get("benchmark_tier", "unknown")
        metrics = self.summary.get("metrics", {})
        failure_counts = self.summary.get("failure_counts", {})
        per_task = self.summary.get("per_task", {})
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>NyssaBench Report - {html.escape(self.suite_id)}</title>
  <style>
    body {{ font-family: Inter, Arial, sans-serif; margin: 40px; color: #17202a; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
    th, td {{ border-bottom: 1px solid #d8dee4; padding: 8px; text-align: left; }}
    .metric {{ display: inline-block; margin: 12px 24px 12px 0; }}
    .value {{ font-size: 28px; font-weight: 700; }}
    pre {{ background: #f6f8fa; padding: 16px; overflow: auto; }}
  </style>
</head>
<body>
  <h1>NyssaBench Report</h1>
  <p><strong>Policy:</strong> {html.escape(self.policy)}<br>
  <strong>Suite:</strong> {html.escape(self.suite_id)}<br>
  <strong>Engine:</strong> {html.escape(self.engine)}<br>
  <strong>Episodes:</strong> {self.summary.get("episodes", 0)}</p>

  <section>
    <div class="metric"><div>Success rate</div><div class="value">{success_rate:.1f}%</div></div>
    <div class="metric"><div>95% CI</div><div class="value">{_format_ci_percent(success_ci)}</div></div>
    <div class="metric"><div>Sim-to-real score</div><div class="value">{sim_to_real:.3f}</div></div>
    <div class="metric"><div>Primary failure mode</div><div class="value">{html.escape(str(primary_failure))}</div></div>
    <div class="metric"><div>Benchmark tier</div><div class="value">{html.escape(str(benchmark_tier))}</div></div>
  </section>

  <h2>Per-Task Results</h2>
  {_per_task_table(per_task)}

  <h2>Aggregate Metrics</h2>
  {_table(metrics)}

  <h2>Failure Clusters</h2>
  {_table(failure_counts)}

  <h2>Raw Summary</h2>
  <pre>{html.escape(json.dumps(self.summary, indent=2))}</pre>
</body>
</html>
"""


def _table(data: dict[str, Any]) -> str:
    rows = "\n".join(
        f"<tr><td>{html.escape(str(key))}</td><td>{html.escape(_format_value(value))}</td></tr>"
        for key, value in sorted(data.items())
    )
    return f"<table><tbody>{rows}</tbody></table>" if rows else "<p>No data.</p>"


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _format_ci_percent(value: Any) -> str:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return "n/a"
    return f"{float(value[0]) * 100:.1f}-{float(value[1]) * 100:.1f}%"


def _per_task_table(data: dict[str, Any]) -> str:
    if not data:
        return "<p>No per-task data.</p>"
    rows = []
    for task_id, summary in sorted(data.items()):
        task_summary = dict(summary)
        success_rate = float(task_summary.get("success_rate", 0.0)) * 100
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(task_id))}</td>"
            f"<td>{int(task_summary.get('episodes', 0))}</td>"
            f"<td>{success_rate:.1f}%</td>"
            f"<td>{html.escape(_format_ci_percent(task_summary.get('success_rate_ci95', [])))}</td>"
            f"<td>{html.escape(str(task_summary.get('primary_failure_mode') or 'none'))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Task</th><th>Episodes</th><th>Success</th>"
        "<th>95% CI</th><th>Primary failure</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )
