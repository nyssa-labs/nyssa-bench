from __future__ import annotations

from typing import Any

from nyssa_bench.baselines.simple_bc import create_task_bc_policy
from nyssa_bench.policies.base import Policy
from nyssa_bench.policies.loaders import call_model, load_callable_from_env, require_model


class TaskBCPolicy(Policy):
    """Task-routed behavior cloning policy for suites with task-specific checkpoints."""

    def __init__(self, model: Any | None = None) -> None:
        loaded = model if model is not None else load_callable_from_env("NYSSA_TASK_BC_POLICY")
        self.model = require_model(
            loaded if loaded is not None else create_task_bc_policy(),
            policy_name="TaskBCPolicy",
            env_var="NYSSA_TASK_BC_POLICY",
        )

    def reset(self, task: Any | None = None, seed: int | None = None) -> None:
        reset = getattr(self.model, "reset", None)
        if callable(reset):
            reset(task=task, seed=seed)

    def act(self, observation: dict[str, Any]) -> Any:
        return call_model(self.model, observation, ("predict_action", "select_action", "get_action", "act"))

    def close(self) -> None:
        close = getattr(self.model, "close", None)
        if callable(close):
            close()
