from __future__ import annotations

from typing import Any

from nyssa_bench.core.task import TaskSpec
from nyssa_bench.engines.base import NyssaEngine


class GenesisEngine(NyssaEngine):
    """Experimental Genesis adapter boundary."""

    def __init__(self) -> None:
        self.task_spec: TaskSpec | None = None

    def load_task(self, task_spec: TaskSpec) -> None:
        self.task_spec = task_spec
        raise RuntimeError(
            "Genesis support is experimental. Use configs/experiments/genesis_contact_stress_v0.yaml "
            "as the integration contract before enabling this adapter."
        )

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        raise RuntimeError("Genesis adapter has not loaded an environment")

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        raise RuntimeError("Genesis adapter has not loaded an environment")

    def render(self) -> Any:
        return None

    def get_state(self) -> dict[str, Any]:
        return {}

    def close(self) -> None:
        return None
