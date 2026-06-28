from __future__ import annotations

from typing import Any


def action_space_spec(env: Any) -> dict[str, Any] | None:
    action_space = getattr(env, "action_space", None)
    if action_space is None:
        return None

    shape = getattr(action_space, "shape", None)
    low = getattr(action_space, "low", None)
    high = getattr(action_space, "high", None)
    dtype = getattr(action_space, "dtype", None)
    if shape is not None and low is not None and high is not None:
        return {
            "type": "box",
            "shape": list(shape),
            "low": _jsonable(low),
            "high": _jsonable(high),
            "dtype": str(dtype) if dtype is not None else None,
        }

    if hasattr(action_space, "n"):
        return {"type": "discrete", "n": int(action_space.n)}

    return {"type": action_space.__class__.__name__}


def wrap_observation(env: Any, observation: Any) -> dict[str, Any]:
    wrapped = {"raw": observation}
    spec = action_space_spec(env)
    if spec is not None:
        wrapped["action_space"] = spec
    return wrapped


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value
