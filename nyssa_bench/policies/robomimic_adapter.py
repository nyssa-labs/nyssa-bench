from __future__ import annotations

from typing import Any

from nyssa_bench.policies.base import Policy
from nyssa_bench.policies.loaders import call_model, load_callable_from_env


class RoboMimicPolicy(Policy):
    def __init__(self, model: Any | None = None) -> None:
        self.model = model if model is not None else load_callable_from_env("NYSSA_ROBOMIMIC_POLICY")

    def act(self, observation: dict[str, Any]) -> Any:
        if self.model is None:
            state = observation.get("state", {})
            distance = float(state.get("distance", 0.0))
            return max(min(distance * 0.45, 0.22), -0.22)
        return call_model(self.model, observation, ("predict_action", "select_action", "act"))
