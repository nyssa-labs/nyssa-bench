from __future__ import annotations

from pathlib import Path


def replay_viewer_placeholder(out_dir: str | Path) -> Path:
    path = Path(out_dir) / "replay.html"
    path.write_text(
        """<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>NyssaBench Replay</title></head>
<body>
<h1>NyssaBench Replay</h1>
<p>This run includes replay metadata. Full video playback is available when the selected engine exports frames.</p>
</body>
</html>
""",
        encoding="utf-8",
    )
    return path
