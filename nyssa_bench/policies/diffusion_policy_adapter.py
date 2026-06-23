from __future__ import annotations

from typing import Any

from nyssa_bench.policies.base import Policy


class DiffusionPolicyAdapter(Policy):
    def __init__(self, model: Any | None = None) -> None:
        self.model = model
        self._last_action = 0.0

    def act(self, observation: dict[str, Any]) -> Any:
        if self.model is None:
            state = observation.get("state", {})
            distance = float(state.get("distance", 0.0))
            proposal = max(min(distance * 0.5, 0.28), -0.28)
            self._last_action = 0.65 * self._last_action + 0.35 * proposal
            return self._last_action
        return self.model.predict_action(observation)
