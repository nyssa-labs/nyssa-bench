from __future__ import annotations

from typing import Any

from nyssa_bench.baselines.scripted_maniskill import create_scripted_oracle
from nyssa_bench.policies.base import Policy
from nyssa_bench.policies.loaders import call_model, load_callable_from_env, require_model


class ScriptedOraclePolicy(Policy):
    """Adapter for task-specific scripted/oracle controllers."""

    def __init__(self, controller: Any | None = None) -> None:
        loaded = controller if controller is not None else load_callable_from_env("NYSSA_SCRIPTED_ORACLE_POLICY")
        self.controller = require_model(
            loaded if loaded is not None else create_scripted_oracle(),
            policy_name="ScriptedOraclePolicy",
            env_var="NYSSA_SCRIPTED_ORACLE_POLICY",
        )

    def reset(self, task: Any | None = None, seed: int | None = None) -> None:
        reset = getattr(self.controller, "reset", None)
        if callable(reset):
            reset(task=task, seed=seed)

    def act(self, observation: dict[str, Any]) -> Any:
        return call_model(self.controller, observation, ("act", "get_action", "select_action", "predict_action"))

    def close(self) -> None:
        close = getattr(self.controller, "close", None)
        if callable(close):
            close()
