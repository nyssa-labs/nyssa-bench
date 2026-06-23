from __future__ import annotations

from typing import Any

from nyssa_bench.policies.base import Policy


class RoboMimicPolicy(Policy):
    def __init__(self, model: Any | None = None) -> None:
        self.model = model

    def act(self, observation: dict[str, Any]) -> Any:
        if self.model is None:
            state = observation.get("state", {})
            distance = float(state.get("distance", 0.0))
            return max(min(distance * 0.45, 0.22), -0.22)
        return self.model(observation)
