from __future__ import annotations

from typing import Any

from nyssa_bench.policies.base import Policy
from nyssa_bench.policies.loaders import call_model, load_callable_from_env, require_model


class RoboMimicPolicy(Policy):
    def __init__(self, model: Any | None = None) -> None:
        self.model = require_model(
            model if model is not None else load_callable_from_env("NYSSA_ROBOMIMIC_POLICY"),
            policy_name="RoboMimicPolicy",
            env_var="NYSSA_ROBOMIMIC_POLICY",
        )

    def act(self, observation: dict[str, Any]) -> Any:
        return call_model(self.model, observation, ("predict_action", "select_action", "get_action", "act"))
