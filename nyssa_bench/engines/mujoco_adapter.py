from __future__ import annotations

from typing import Any

from nyssa_bench.core.task import TaskSpec
from nyssa_bench.engines.base import NyssaEngine


class MuJoCoEngine(NyssaEngine):
    """Adapter boundary for Gymnasium/MuJoCo environments."""

    def __init__(self) -> None:
        self.env: Any | None = None
        self.task_spec: TaskSpec | None = None

    def load_task(self, task_spec: TaskSpec) -> None:
        self.task_spec = task_spec
        try:
            import gymnasium as gym
            import mujoco  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Install NyssaBench with the MuJoCo extra: pip install -e '.[mujoco]'") from exc

        env_id = task_spec.success.get("mujoco_env_id") or task_spec.task_id
        self.env = gym.make(env_id)

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self._require_env()
        observation, info = self.env.reset(seed=seed)
        return {"raw": observation}, dict(info)

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._require_env()
        observation, reward, terminated, truncated, info = self.env.step(action)
        return {"raw": observation}, float(reward), bool(terminated), bool(truncated), dict(info)

    def render(self) -> Any:
        self._require_env()
        return self.env.render()

    def get_state(self) -> dict[str, Any]:
        if self.env is not None and hasattr(self.env, "data"):
            return {"time": float(getattr(self.env.data, "time", 0.0))}
        return {}

    def close(self) -> None:
        if self.env is not None:
            self.env.close()

    def _require_env(self) -> None:
        if self.env is None:
            raise RuntimeError("No MuJoCo environment loaded. Call load_task first.")
