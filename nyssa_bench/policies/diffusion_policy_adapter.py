from __future__ import annotations

from typing import Any

from nyssa_bench.policies.base import Policy
from nyssa_bench.policies.loaders import call_model, dummy_state_fallback_action, load_callable_from_env


class DiffusionPolicyAdapter(Policy):
    def __init__(self, model: Any | None = None) -> None:
        self.model = model if model is not None else load_callable_from_env("NYSSA_DIFFUSION_POLICY")
        self._last_action = 0.0

    def act(self, observation: dict[str, Any]) -> Any:
        if self.model is None:
            proposal = dummy_state_fallback_action(
                observation,
                gain=0.5,
                limit=0.28,
                policy_name="DiffusionPolicyAdapter",
                env_var="NYSSA_DIFFUSION_POLICY",
            )
            self._last_action = 0.65 * self._last_action + 0.35 * proposal
            return self._last_action
        return call_model(self.model, observation, ("predict_action", "select_action", "act"))
