from __future__ import annotations

from typing import Any

from nyssa_bench.policies.base import Policy


class OpenVLAPolicy(Policy):
    def __init__(self, model: Any | None = None) -> None:
        self.model = model
        self.instruction = "complete the manipulation task"

    def act(self, observation: dict[str, Any]) -> Any:
        if self.model is None:
            state = observation.get("state", {})
            distance = float(state.get("distance", 0.0))
            return max(min(distance * 0.55, 0.3), -0.3)
        return self.model.predict_action(observation)
