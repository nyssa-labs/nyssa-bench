from __future__ import annotations

from typing import Any

from nyssa_bench.policies.base import Policy
from nyssa_bench.policies.loaders import call_model, dummy_state_fallback_action, load_callable_from_env


class RoboMimicPolicy(Policy):
    def __init__(self, model: Any | None = None) -> None:
        self.model = model if model is not None else load_callable_from_env("NYSSA_ROBOMIMIC_POLICY")

    def act(self, observation: dict[str, Any]) -> Any:
        if self.model is None:
            return dummy_state_fallback_action(
                observation,
                gain=0.45,
                limit=0.22,
                policy_name="RoboMimicPolicy",
                env_var="NYSSA_ROBOMIMIC_POLICY",
            )
        return call_model(self.model, observation, ("predict_action", "select_action", "act"))
