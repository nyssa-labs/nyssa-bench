from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from nyssa_bench.baselines.features import fit_action_to_observation, flatten_observation
from nyssa_bench.baselines.robomimic_bc import create_robomimic_policy, load_robomimic_checkpoint
from nyssa_bench.policies.base import Policy
from nyssa_bench.policies.loaders import call_model, load_callable_from_env, normalize_action, require_model


class RoboMimicPolicy(Policy):
    def __init__(self, model: Any | None = None) -> None:
        loaded = model if model is not None else load_callable_from_env("NYSSA_ROBOMIMIC_POLICY")
        self.model = require_model(
            loaded if loaded is not None else create_robomimic_policy(),
            policy_name="RoboMimicPolicy",
            env_var="NYSSA_ROBOMIMIC_POLICY",
        )
        self.feature_dim = int(os.getenv("NYSSA_ROBOMIMIC_FEATURE_DIM", "256"))

    def reset(self, task: Any | None = None, seed: int | None = None) -> None:
        _reset_robomimic_model(self.model)

    def act(self, observation: dict[str, Any]) -> Any:
        return _robomimic_action(self.model, observation, feature_dim=self.feature_dim)


class TaskRoboMimicPolicy(Policy):
    """Task-routed RoboMimic policy for one checkpoint per task."""

    def __init__(self) -> None:
        self.checkpoint_dir = Path(os.getenv("NYSSA_TASK_ROBOMIMIC_DIR", "checkpoints/robomimic_by_task"))
        self.feature_dim = int(os.getenv("NYSSA_ROBOMIMIC_FEATURE_DIM", "256"))
        self.current_task_id: str | None = None
        self._models: dict[str, Any] = {}

    def reset(self, task: Any | None = None, seed: int | None = None) -> None:
        self.current_task_id = str(getattr(task, "task_id", "")) or None
        if self.current_task_id:
            _reset_robomimic_model(self._model_for_task(self.current_task_id))

    def act(self, observation: dict[str, Any]) -> Any:
        if not self.current_task_id:
            raise RuntimeError("Task-routed RoboMimic policy was used before reset(task=...)")
        return _robomimic_action(self._model_for_task(self.current_task_id), observation, feature_dim=self.feature_dim)

    def _model_for_task(self, task_id: str) -> Any:
        key = _checkpoint_key(task_id)
        if key not in self._models:
            path = self.checkpoint_dir / f"{key}.pth"
            self._models[key] = load_robomimic_checkpoint(path)
        return self._models[key]


def _robomimic_action(model: Any, observation: dict[str, Any], *, feature_dim: int) -> Any:
    flat_observation = {"flat": flatten_observation(observation, feature_dim)}
    action = _call_robomimic_model(model, flat_observation)
    return fit_action_to_observation(action, observation)


def _reset_robomimic_model(model: Any) -> None:
    start_episode = getattr(model, "start_episode", None)
    if callable(start_episode):
        start_episode()
    reset = getattr(model, "reset", None)
    if callable(reset):
        reset()


def _call_robomimic_model(model: Any, observation: dict[str, Any]) -> Any:
    for method_name in ("get_action", "predict_action", "select_action", "act"):
        method = getattr(model, method_name, None)
        if callable(method):
            return normalize_action(method(observation))
    if callable(model):
        return normalize_action(model(observation))
    return call_model(model, observation, ("get_action", "predict_action", "select_action", "act"))


def _checkpoint_key(task_id: str) -> str:
    aliases = {
        "maniskill_pick_cube_joint": "maniskill_pick_cube",
        "maniskill_stack_cube_joint": "maniskill_stack_cube",
        "maniskill_push_cube_joint": "maniskill_push_cube",
    }
    return aliases.get(task_id, task_id.removesuffix("_joint"))
