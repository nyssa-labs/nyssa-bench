from __future__ import annotations

import html
import json
from pathlib import Path

from nyssa_bench.arena.pairwise_runner import PairwiseSummary
from nyssa_bench.arena.preference_schema import PreferenceRecord


def save_pairwise_results(summary: PairwiseSummary, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "pairwise_results.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for outcome in summary.outcomes:
            handle.write(json.dumps(outcome.__dict__, sort_keys=True) + "\n")
    return path


def save_preference_table(preferences: list[PreferenceRecord], out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "preference_table.csv"
    with path.open("w", encoding="utf-8") as handle:
        handle.write("task_id,seed,episode_index,choice,reason,evaluator_id,blinded\n")
        for item in preferences:
            handle.write(
                ",".join(
                    [
                        _csv(item.task_id),
                        str(item.seed),
                        str(item.episode_index),
                        _csv(item.choice),
                        _csv(item.reason),
                        _csv(item.evaluator_id or ""),
                        str(item.blinded).lower(),
                    ]
                )
                + "\n"
            )
    return path


def save_arena_report(summary: PairwiseSummary, out_dir: str | Path, *, title: str = "NyssaBench Arena Report") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(item.task_id)}</td>"
        f"<td>{item.seed}</td>"
        f"<td>{item.episode_index}</td>"
        f"<td>{html.escape(item.winner)}</td>"
        f"<td>{html.escape(str(item.policy_a_failure or ''))}</td>"
        f"<td>{html.escape(str(item.policy_b_failure or ''))}</td>"
        "</tr>"
        for item in summary.outcomes
    )
    wins = ", ".join(f"{html.escape(key)}: {value}" for key, value in sorted(summary.wins.items()))
    failure_deltas = ", ".join(
        f"{html.escape(key)}: {value}" for key, value in sorted(summary.failure_deltas.items())
    )
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Inter, Arial, sans-serif; margin: 40px; color: #17202a; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #d8dee4; padding: 8px; text-align: left; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p>Total pairs: {summary.total_pairs}</p>
  <p>Wins: {wins or "none"}</p>
  <p>Failure deltas: {failure_deltas or "none"}</p>
  <table>
    <thead>
      <tr><th>Task</th><th>Seed</th><th>Episode</th><th>Winner</th><th>Policy A failure</th><th>Policy B failure</th></tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
"""
    path = out_dir / "arena_report.html"
    path.write_text(body, encoding="utf-8")
    return path


def _csv(value: str) -> str:
    escaped = value.replace('"', '""')
    if any(char in escaped for char in [",", "\n", '"']):
        return f'"{escaped}"'
    return escaped

