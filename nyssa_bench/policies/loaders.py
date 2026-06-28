from __future__ import annotations

import os
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any


def load_callable_from_env(env_var: str) -> Any | None:
    value = os.getenv(env_var)
    if not value:
        return None
    loaded = load_object(value)
    return loaded() if isinstance(loaded, type) else loaded


def load_object(path: str) -> Any:
    if ":" not in path:
        raise ValueError(f"Expected object path as module:attribute or file.py:attribute, got {path!r}")
    module_ref, _, attr = path.partition(":")
    if module_ref.endswith(".py") or Path(module_ref).exists():
        module_path = Path(module_ref)
        spec = spec_from_file_location(module_path.stem, module_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not load module from {module_path}")
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = import_module(module_ref)
    return getattr(module, attr)


def call_model(model: Any, observation: dict[str, Any], method_names: tuple[str, ...]) -> Any:
    for method_name in method_names:
        method = getattr(model, method_name, None)
        if callable(method):
            return method(observation)
    if callable(model):
        return model(observation)
    raise TypeError(f"Model must be callable or implement one of: {', '.join(method_names)}")


def dummy_state_fallback_action(
    observation: dict[str, Any],
    *,
    gain: float,
    limit: float,
    policy_name: str,
    env_var: str,
) -> float:
    state = observation.get("state")
    if not isinstance(state, dict) or "distance" not in state:
        raise RuntimeError(
            f"{policy_name} requires {env_var}=module:factory for real simulator observations. "
            "The built-in fallback only supports the dummy smoke-test engine."
        )
    distance = float(state.get("distance", 0.0))
    return max(min(distance * gain, limit), -limit)
