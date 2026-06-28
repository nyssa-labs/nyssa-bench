from __future__ import annotations

from typing import Any

from nyssa_bench.policies.base import Policy
from nyssa_bench.policies.loaders import call_model, dummy_state_fallback_action, load_callable_from_env


class LeRobotPolicy(Policy):
    """LeRobot-compatible adapter with a deterministic smoke fallback."""

    def __init__(self, model: Any | None = None) -> None:
        self.model = model if model is not None else load_callable_from_env("NYSSA_LEROBOT_POLICY")

    def act(self, observation: dict[str, Any]) -> Any:
        if self.model is None:
            return dummy_state_fallback_action(
                observation,
                gain=0.5,
                limit=0.25,
                policy_name="LeRobotPolicy",
                env_var="NYSSA_LEROBOT_POLICY",
            )
        return call_model(self.model, observation, ("select_action", "predict_action", "act"))
