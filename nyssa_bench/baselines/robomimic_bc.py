from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def load_robomimic_checkpoint(path: str | Path | None = None) -> Any:
    checkpoint = Path(path or os.getenv("NYSSA_ROBOMIMIC_CHECKPOINT", "checkpoints/robomimic_policy.pth"))
    if not checkpoint.exists():
        raise RuntimeError(
            f"RoboMimic checkpoint not found: {checkpoint}. Train one with `nyssa train-robomimic` "
            "or set NYSSA_ROBOMIMIC_POLICY=module:factory."
        )
    try:
        from robomimic.utils.file_utils import policy_from_checkpoint
    except ImportError as exc:
        raise RuntimeError("RoboMimic integration requires: uv sync --extra robomimic") from exc
    policy, _ = policy_from_checkpoint(ckpt_path=str(checkpoint))
    return policy


def create_robomimic_policy() -> Any:
    return load_robomimic_checkpoint()


def train_robomimic(config: str | Path, *, name: str | None = None, debug: bool = False) -> None:
    try:
        import robomimic  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("RoboMimic training requires: uv sync --extra robomimic") from exc

    command = [sys.executable, "-m", "robomimic.scripts.train", "--config", str(config)]
    if name:
        command.extend(["--name", name])
    if debug:
        command.append("--debug")
    subprocess.run(command, check=True)
