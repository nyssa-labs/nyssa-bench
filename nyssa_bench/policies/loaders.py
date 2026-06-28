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
            return normalize_action(method(observation))
    if callable(model):
        return normalize_action(model(observation))
    raise TypeError(f"Model must be callable or implement one of: {', '.join(method_names)}")


def require_model(model: Any | None, *, policy_name: str, env_var: str) -> Any:
    if model is None:
        raise RuntimeError(
            f"{policy_name} requires a real model. Pass one from Python or set "
            f"{env_var}=module:factory for CLI runs."
        )
    return model


def normalize_action(action: Any) -> Any:
    if isinstance(action, dict):
        for key in ("action", "actions", "pred_action", "pred_actions"):
            if key in action:
                return normalize_action(action[key])
        return action
    if hasattr(action, "detach"):
        action = action.detach()
    if hasattr(action, "cpu"):
        action = action.cpu()
    if hasattr(action, "numpy"):
        return action.numpy()
    if hasattr(action, "item"):
        try:
            return action.item()
        except ValueError:
            pass
    return action
