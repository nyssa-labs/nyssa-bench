from __future__ import annotations

import random
from typing import Any

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

    def render(self) -> dict[str, Any]:
        return {"type": "dummy_frame", "step": self.step_count, "position": self.position, "target": self.target}

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
