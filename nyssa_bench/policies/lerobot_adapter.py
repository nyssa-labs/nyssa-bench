from __future__ import annotations

from typing import Any

from nyssa_bench.policies.base import Policy


class LeRobotPolicy(Policy):
    """Thin placeholder for LeRobot policies.

    v0.1 keeps the interface stable while dataset export support lands first.
    """

    def __init__(self, model: Any | None = None) -> None:
        self.model = model

    def act(self, observation: dict[str, Any]) -> Any:
        if self.model is None:
            raise RuntimeError("LeRobotPolicy requires a loaded LeRobot model")
        return self.model.select_action(observation)
