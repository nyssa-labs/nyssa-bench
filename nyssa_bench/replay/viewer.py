from __future__ import annotations

from pathlib import Path


def replay_viewer_placeholder(out_dir: str | Path) -> Path:
    path = Path(out_dir) / "replay.html"
    path.write_text(
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>NyssaBench Replay</title>
  <style>
    body { font-family: Inter, Arial, sans-serif; margin: 32px; color: #17202a; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border-bottom: 1px solid #d8dee4; padding: 8px; text-align: left; }
    video { max-width: 360px; width: 100%; }
  </style>
</head>
<body>
<h1>NyssaBench Replay</h1>
<p>This viewer reads <code>replay_manifest.json</code>. MP4 playback appears when the selected engine exports render frames.</p>
<table>
  <thead><tr><th>Task</th><th>Episode</th><th>Status</th><th>Failure</th><th>Replay</th></tr></thead>
  <tbody id="rows"></tbody>
</table>
<script>
fetch("replay_manifest.json")
  .then((response) => response.json())
  .then((manifest) => {
    const rows = document.getElementById("rows");
    for (const episode of manifest.episodes) {
      const tr = document.createElement("tr");
      const status = episode.success ? "success" : "failure";
      const replay = episode.replay_path
        ? `<video controls src="${episode.replay_path}"></video>`
        : "No video exported";
      tr.innerHTML = `<td>${episode.task_id}</td><td>${episode.episode_index}</td><td>${status}</td><td>${episode.failure_label || ""}</td><td>${replay}</td>`;
      rows.appendChild(tr);
    }
  });
</script>
</body>
</html>
""",
        encoding="utf-8",
    )
    return path
