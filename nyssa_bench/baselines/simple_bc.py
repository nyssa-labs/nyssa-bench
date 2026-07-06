from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from nyssa_bench.baselines.features import action_bounds, fit_action_to_observation, flatten_observation, normalize_action


class LinearBCPolicy:
    def __init__(self, weights: np.ndarray, bias: np.ndarray, feature_dim: int, action_size: int) -> None:
        self.weights = weights
        self.bias = bias
        self.feature_dim = feature_dim
        self.action_size = action_size

    def predict_action(self, observation: dict[str, Any]) -> Any:
        features = flatten_observation(observation, self.feature_dim)
        action = features @ self.weights + self.bias
        return fit_action_to_observation(action, observation)

    @classmethod
    def load(cls, path: str | Path) -> "LinearBCPolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            weights=np.asarray(payload["weights"], dtype=float),
            bias=np.asarray(payload["bias"], dtype=float),
            feature_dim=int(payload["feature_dim"]),
            action_size=int(payload["action_size"]),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "nyssa-linear-bc-v1",
            "feature_dim": self.feature_dim,
            "action_size": self.action_size,
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path


class TaskRoutedLinearBCPolicy:
    def __init__(self, checkpoint_dir: str | Path) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.current_task_id: str | None = None
        self._models: dict[str, LinearBCPolicy] = {}

    def reset(self, task: Any | None = None, seed: int | None = None) -> None:
        self.current_task_id = str(getattr(task, "task_id", "")) or None

    def predict_action(self, observation: dict[str, Any]) -> Any:
        if not self.current_task_id:
            raise RuntimeError("Task-routed BC policy was used before reset(task=...)")
        return self._model_for_task(self.current_task_id).predict_action(observation)

    def _model_for_task(self, task_id: str) -> LinearBCPolicy:
        key = _checkpoint_key(task_id)
        if key not in self._models:
            path = self.checkpoint_dir / f"{key}.json"
            if not path.exists():
                raise RuntimeError(
                    f"Task BC checkpoint not found for task '{task_id}': {path}. "
                    "Train one checkpoint per task under NYSSA_TASK_BC_DIR."
                )
            self._models[key] = LinearBCPolicy.load(path)
        return self._models[key]


def train_linear_bc(
    episodes_path: str | Path,
    out: str | Path,
    *,
    feature_dim: int = 256,
    ridge: float = 1e-3,
) -> Path:
    rows_x: list[np.ndarray] = []
    rows_y: list[np.ndarray] = []
    action_size: int | None = None
    for episode in json.loads(Path(episodes_path).read_text(encoding="utf-8")):
        for step in episode.get("steps", []):
            observation = step.get("observation", {})
            low, high, shape = action_bounds(observation)
            size = int(np.prod(shape))
            action_size = action_size or size
            rows_x.append(flatten_observation(observation, feature_dim))
            rows_y.append(normalize_action(step.get("action"), action_size))
    if not rows_x or action_size is None:
        raise ValueError(f"No training steps found in {episodes_path}")

    x = np.vstack(rows_x)
    y = np.vstack(rows_y)
    x_aug = np.hstack([x, np.ones((x.shape[0], 1))])
    regularizer = ridge * np.eye(x_aug.shape[1])
    solution = np.linalg.solve(x_aug.T @ x_aug + regularizer, x_aug.T @ y)
    policy = LinearBCPolicy(
        weights=solution[:-1, :],
        bias=solution[-1, :],
        feature_dim=feature_dim,
        action_size=action_size,
    )
    return policy.save(out)


def create_bc_policy() -> LinearBCPolicy:
    checkpoint = os.getenv("NYSSA_BC_CHECKPOINT", "checkpoints/bc_policy.json")
    if not Path(checkpoint).exists():
        raise RuntimeError(
            f"BC checkpoint not found: {checkpoint}. Run `nyssa train-bc ... --out {checkpoint}` "
            "or set NYSSA_BC_POLICY=module:factory."
        )
    return LinearBCPolicy.load(checkpoint)


def create_task_bc_policy() -> TaskRoutedLinearBCPolicy:
    checkpoint_dir = os.getenv("NYSSA_TASK_BC_DIR", "checkpoints/bc_by_task")
    return TaskRoutedLinearBCPolicy(checkpoint_dir)


def task_checkpoint_key(task_id: str) -> str:
    aliases = {
        "maniskill_pick_cube_joint": "maniskill_pick_cube",
        "maniskill_stack_cube_joint": "maniskill_stack_cube",
        "maniskill_push_cube_joint": "maniskill_push_cube",
    }
    return aliases.get(task_id, task_id.removesuffix("_joint"))


def _checkpoint_key(task_id: str) -> str:
    return task_checkpoint_key(task_id)
