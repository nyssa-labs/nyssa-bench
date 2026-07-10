from __future__ import annotations

import os
from typing import Any

from nyssa_bench.core.task import TaskSpec
from nyssa_bench.engines.base import NyssaEngine
from nyssa_bench.engines.spaces import wrap_observation


class ManiSkillEngine(NyssaEngine):
    """Adapter boundary for ManiSkill environments."""

    def __init__(self) -> None:
        self.env: Any | None = None
        self.task_spec: TaskSpec | None = None
        self.max_steps = 1000

    def load_task(self, task_spec: TaskSpec) -> None:
        self.task_spec = task_spec
        self.max_steps = int(task_spec.success.get("max_steps", self.max_steps))
        try:
            import gymnasium as gym
            import mani_skill  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Install NyssaBench with the ManiSkill extra: pip install -e '.[maniskill]'") from exc

        env_id = _resolve_env_id(task_spec, "maniskill")
        env_kwargs = _maniskill_env_kwargs(task_spec)
        self.env = gym.make(env_id, **env_kwargs)

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self._require_env()
        observation, info = self.env.reset(seed=seed)
        return wrap_observation(self.env, observation), dict(info)

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._require_env()
        action = self._coerce_action(action)
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["success"] = _extract_success(info, self.task_spec)
        return wrap_observation(self.env, observation), float(reward), bool(terminated), bool(truncated), info

    def render(self) -> Any:
        self._require_env()
        return self.env.render()

    def get_state(self) -> dict[str, Any]:
        if self.env is not None and hasattr(self.env, "get_state"):
            return {"raw": self.env.get_state()}
        return {}

    def set_state(self, state: Any) -> dict[str, Any] | None:
        self._require_env()
        target = getattr(self.env, "unwrapped", self.env)
        state_payload = _to_numpy_state(state)
        if isinstance(state_payload, dict) and hasattr(target, "set_state_dict"):
            target.set_state_dict(state_payload)
        elif hasattr(target, "set_state"):
            target.set_state(state_payload)
        else:
            raise RuntimeError("Loaded ManiSkill environment does not support state restore.")
        observation = _get_observation_after_state_restore(self.env)
        return wrap_observation(self.env, observation) if observation is not None else None

    def close(self) -> None:
        if self.env is not None:
            self.env.close()

    def _require_env(self) -> None:
        if self.env is None:
            raise RuntimeError("No ManiSkill environment loaded. Call load_task first.")

    def _coerce_action(self, action: Any) -> Any:
        if self.env is None or not hasattr(self.env, "action_space"):
            return action
        action_space = self.env.action_space
        if hasattr(action_space, "shape") and action_space.shape and isinstance(action, (int, float)):
            try:
                import numpy as np
            except ImportError:
                return action
            low = getattr(action_space, "low", None)
            high = getattr(action_space, "high", None)
            value = np.full(action_space.shape, float(action), dtype=getattr(action_space, "dtype", float))
            if low is not None and high is not None:
                value = np.clip(value, low, high)
            return value
        return action


def _resolve_env_id(task_spec: TaskSpec, engine: str) -> str:
    engine_env_ids = task_spec.success.get("engine_env_ids", {})
    if isinstance(engine_env_ids, dict) and engine_env_ids.get(engine):
        return str(engine_env_ids[engine])
    legacy = task_spec.success.get(f"{engine}_env_id")
    if legacy:
        return str(legacy)
    raise RuntimeError(
        f"Task '{task_spec.task_id}' is missing success.engine_env_ids.{engine}. "
        "Real simulator tasks must define explicit environment mappings."
    )


def _maniskill_env_kwargs(task_spec: TaskSpec) -> dict[str, Any]:
    render_mode = _env_or_task_value(
        "NYSSA_MANISKILL_RENDER_MODE",
        task_spec,
        "render_mode",
        "rgb_array",
    )
    env_kwargs: dict[str, Any] = {}
    if str(render_mode).lower() not in {"", "none", "null"}:
        env_kwargs["render_mode"] = render_mode
    for env_name, key in (
        ("NYSSA_MANISKILL_OBS_MODE", "obs_mode"),
        ("NYSSA_MANISKILL_CONTROL_MODE", "control_mode"),
        ("NYSSA_MANISKILL_ROBOT_UIDS", "robot_uids"),
        ("NYSSA_MANISKILL_SIM_BACKEND", "sim_backend"),
        ("NYSSA_MANISKILL_RENDER_DEVICE", "render_device"),
        ("NYSSA_MANISKILL_SHADER_DIR", "shader_dir"),
    ):
        value = _env_or_task_value(env_name, task_spec, key, None)
        if value is not None and str(value).lower() not in {"", "none", "null"}:
            env_kwargs[key] = value
    return env_kwargs


def _env_or_task_value(env_name: str, task_spec: TaskSpec, key: str, default: Any) -> Any:
    value = os.getenv(env_name)
    if value is not None:
        return value
    return task_spec.success.get(key, default)


def _extract_success(info: dict[str, Any], task_spec: TaskSpec | None) -> bool:
    configured_keys = []
    if task_spec is not None:
        configured = task_spec.success.get("success_info_keys", [])
        if isinstance(configured, str):
            configured_keys.append(configured)
        elif isinstance(configured, list):
            configured_keys.extend(str(key) for key in configured)
    for key in [*configured_keys, "success", "is_success", "success_once"]:
        if key in info:
            return _as_bool(info[key])
    return False


def _as_bool(value: Any) -> bool:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "item"):
        try:
            return bool(value.item())
        except ValueError:
            pass
    if hasattr(value, "all"):
        return bool(value.all())
    return bool(value)


def _to_numpy_state(value: Any) -> Any:
    try:
        import numpy as np
    except ImportError:
        return value
    if isinstance(value, dict):
        if set(value) == {"raw"}:
            return _to_numpy_state(value["raw"])
        for key in ("env_states", "states", "state"):
            if key in value:
                return _to_numpy_state(value[key])
        return {key: _to_numpy_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return np.asarray(value)
    return value


def _get_observation_after_state_restore(env: Any) -> Any | None:
    for target in (env, getattr(env, "unwrapped", None)):
        if target is None:
            continue
        for name in ("get_obs", "_get_obs"):
            method = getattr(target, name, None)
            if method is None:
                continue
            try:
                return method()
            except TypeError:
                continue
    return None
