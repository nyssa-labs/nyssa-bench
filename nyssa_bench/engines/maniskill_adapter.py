from __future__ import annotations

from typing import Any

from nyssa_bench.core.task import TaskSpec
from nyssa_bench.engines.base import NyssaEngine


class ManiSkillEngine(NyssaEngine):
    """Adapter boundary for ManiSkill environments."""

    def __init__(self) -> None:
        self.env: Any | None = None
        self.task_spec: TaskSpec | None = None
        self.max_steps = 1000

    def load_task(self, task_spec: TaskSpec) -> None:
        self.task_spec = task_spec
        self.max_steps = int(task_spec.success.get("max_steps", self.max_steps))
        try:
            import gymnasium as gym
            import mani_skill  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Install NyssaBench with the ManiSkill extra: pip install -e '.[maniskill]'") from exc

        env_id = _resolve_env_id(task_spec, "maniskill")
        self.env = gym.make(env_id, render_mode=task_spec.success.get("render_mode", "rgb_array"))

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self._require_env()
        observation, info = self.env.reset(seed=seed)
        return {"raw": observation}, dict(info)

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._require_env()
        action = self._coerce_action(action)
        observation, reward, terminated, truncated, info = self.env.step(action)
        return {"raw": observation}, float(reward), bool(terminated), bool(truncated), dict(info)

    def render(self) -> Any:
        self._require_env()
        return self.env.render()

    def get_state(self) -> dict[str, Any]:
        if self.env is not None and hasattr(self.env, "get_state"):
            return {"raw": self.env.get_state()}
        return {}

    def close(self) -> None:
        if self.env is not None:
            self.env.close()

    def _require_env(self) -> None:
        if self.env is None:
            raise RuntimeError("No ManiSkill environment loaded. Call load_task first.")

    def _coerce_action(self, action: Any) -> Any:
        if self.env is None or not hasattr(self.env, "action_space"):
            return action
        action_space = self.env.action_space
        if hasattr(action_space, "shape") and action_space.shape and isinstance(action, (int, float)):
            try:
                import numpy as np
            except ImportError:
                return action
            low = getattr(action_space, "low", None)
            high = getattr(action_space, "high", None)
            value = np.full(action_space.shape, float(action), dtype=getattr(action_space, "dtype", float))
            if low is not None and high is not None:
                value = np.clip(value, low, high)
            return value
        return action


def _resolve_env_id(task_spec: TaskSpec, engine: str) -> str:
    engine_env_ids = task_spec.success.get("engine_env_ids", {})
    if isinstance(engine_env_ids, dict) and engine_env_ids.get(engine):
        return str(engine_env_ids[engine])
    return str(task_spec.success.get(f"{engine}_env_id") or task_spec.task_id)
