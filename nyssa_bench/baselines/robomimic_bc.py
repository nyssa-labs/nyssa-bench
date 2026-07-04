from __future__ import annotations

import os
import json
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


def write_robomimic_bc_config(
    *,
    data: str | Path,
    out: str | Path,
    output_dir: str | Path = "checkpoints/robomimic",
    name: str = "nyssa_robomimic_bc_flat",
    epochs: int = 50,
    batch_size: int = 64,
    seed: int = 1,
    learning_rate: float = 1e-4,
) -> Path:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "algo_name": "bc",
        "experiment": {
            "name": name,
            "validate": False,
            "logging": {
                "terminal_output_to_txt": True,
                "log_tb": True,
            },
            "save": {
                "enabled": True,
                "every_n_seconds": 600,
                "every_n_epochs": 1,
            },
            "epoch_every_n_steps": 1000,
            "validation_epoch_every_n_steps": 100,
        },
        "train": {
            "data": str(data),
            "output_dir": str(output_dir),
            "num_epochs": int(epochs),
            "batch_size": int(batch_size),
            "num_data_workers": 0,
            "hdf5_cache_mode": "low_dim",
            "hdf5_normalize_obs": True,
            "seed": int(seed),
        },
        "observation": {
            "modalities": {
                "obs": {
                    "low_dim": ["flat"],
                    "rgb": [],
                    "depth": [],
                    "scan": [],
                }
            }
        },
        "algo": {
            "optim_params": {
                "policy": {
                    "learning_rate": {
                        "initial": float(learning_rate),
                    }
                }
            },
            "actor_layer_dims": [1024, 1024],
            "gmm": {
                "enabled": False,
            },
            "gaussian": {
                "enabled": False,
            },
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
