from __future__ import annotations

import random
from typing import Any

import numpy as np

from nyssa_bench.core.task import TaskSpec
from nyssa_bench.engines.base import NyssaEngine
from nyssa_bench.metrics.taxonomy import FAILURE_LABELS


class DummyEngine(NyssaEngine):
    """Small deterministic engine for tests, docs, and first-run demos."""

    def __init__(self) -> None:
        self.task_spec: TaskSpec | None = None
        self.rng = random.Random()
        self.step_count = 0
        self.max_steps = 12
        self.target = 1.0
        self.position = 0.0
        self.collisions = 0

    def load_task(self, task_spec: TaskSpec) -> None:
        self.task_spec = task_spec
        self.max_steps = int(task_spec.success.get("max_steps", 12))

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self.rng.seed(seed)
        self.step_count = 0
        self.position = self.rng.uniform(-0.25, 0.25)
        self.target = self.rng.uniform(0.8, 1.2)
        self.collisions = 0
        return self._observation(), {"seed": seed}

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        scalar = _action_to_scalar(action)
        noise = self.rng.uniform(-0.02, 0.02)
        self.position += max(min(scalar, 0.35), -0.35) + noise

        if abs(scalar) > 0.32 and self.rng.random() < 0.15:
            self.collisions += 1

        distance = abs(self.target - self.position)
        success = distance < 0.08
        truncated = self.step_count >= self.max_steps and not success
        terminated = success
        reward = 1.0 if success else -distance

        failure_label = None
        if truncated:
            labels = self.task_spec.failure_labels if self.task_spec and self.task_spec.failure_labels else FAILURE_LABELS
            failure_label = labels[(self.step_count + self.collisions) % len(labels)]

        info = {
            "success": success,
            "failure_label": failure_label,
            "collision_count": self.collisions,
            "completion_time": float(self.step_count),
            "path_efficiency": max(0.0, 1.0 - distance),
            "grasp_success": success or distance < 0.2,
            "safety_violation": self.collisions > 2,
        }
        return self._observation(), reward, terminated, truncated, info

    def render(self) -> Any:
        """Render an inspectable RGB frame for reports and demo videos."""

        width, height = 640, 352
        frame = np.full((height, width, 3), 248, dtype=np.uint8)
        frame[:, :, 0] = 236
        frame[:, :, 1] = 242
        frame[:, :, 2] = 245

        track_y = height // 2
        frame[track_y - 4 : track_y + 4, 80 : width - 80] = np.array([52, 64, 84], dtype=np.uint8)

        left, right = 80, width - 80
        low, high = -0.5, 1.5
        target_x = _scale_to_pixel(self.target, low, high, left, right)
        position_x = _scale_to_pixel(self.position, low, high, left, right)

        _draw_circle(frame, target_x, track_y, 20, (33, 150, 83))
        _draw_circle(frame, position_x, track_y, 16, (36, 99, 235))

        progress = min(1.0, max(0.0, self.step_count / max(1, self.max_steps)))
        progress_right = int(80 + progress * (width - 160))
        frame[height - 42 : height - 28, 80:progress_right] = np.array([36, 99, 235], dtype=np.uint8)
        frame[height - 42 : height - 28, progress_right : width - 80] = np.array([203, 213, 225], dtype=np.uint8)

        for index in range(min(self.collisions, 8)):
            x = 88 + index * 18
            frame[30:44, x : x + 12] = np.array([220, 53, 69], dtype=np.uint8)

        return frame

    def get_state(self) -> dict[str, Any]:
        return {
            "step": self.step_count,
            "position": self.position,
            "target": self.target,
            "collisions": self.collisions,
        }

    def close(self) -> None:
        return None

    def _observation(self) -> dict[str, Any]:
        return {
            "state": {
                "position": round(self.position, 5),
                "target": round(self.target, 5),
                "distance": round(self.target - self.position, 5),
            }
        }


def _action_to_scalar(action: Any) -> float:
    if isinstance(action, dict):
        value = action.get("delta", action.get("action", 0.0))
        return _action_to_scalar(value)
    if isinstance(action, (list, tuple)):
        return float(action[0]) if action else 0.0
    return float(action)


def _scale_to_pixel(value: float, low: float, high: float, left: int, right: int) -> int:
    normalized = min(1.0, max(0.0, (value - low) / (high - low)))
    return int(left + normalized * (right - left))


def _draw_circle(frame: np.ndarray, center_x: int, center_y: int, radius: int, color: tuple[int, int, int]) -> None:
    height, width = frame.shape[:2]
    y, x = np.ogrid[:height, :width]
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2
    frame[mask] = np.array(color, dtype=np.uint8)
