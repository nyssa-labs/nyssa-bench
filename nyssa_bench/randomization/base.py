from __future__ import annotations

from typing import Any


SUPPORTED_RANDOMIZATION_KEYS: dict[str, set[str]] = {
    "maniskill": {"seed"},
    "mujoco": {"seed"},
}


def summarize_randomization(randomization: dict[str, Any]) -> dict[str, Any]:
    return {
        "enabled_keys": sorted(key for key, value in randomization.items() if bool(value)),
        "raw": randomization,
    }


def summarize_stressor_support(randomization: dict[str, Any], engine: str) -> dict[str, Any]:
    enabled = sorted(key for key, value in randomization.items() if _is_enabled(value))
    supported = SUPPORTED_RANDOMIZATION_KEYS.get(engine, set())
    unsupported = sorted(key for key in enabled if key not in supported)
    return {
        "enabled_stressors": enabled,
        "supported_stressors": sorted(key for key in enabled if key in supported),
        "unsupported_stressors": unsupported,
    }


def aggregate_stressor_support(task_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    unsupported: dict[str, list[str]] = {}
    supported: dict[str, list[str]] = {}
    for task_id, summary in task_summaries.items():
        if summary.get("unsupported_stressors"):
            unsupported[task_id] = list(summary["unsupported_stressors"])
        if summary.get("supported_stressors"):
            supported[task_id] = list(summary["supported_stressors"])
    return {
        "supported_by_task": supported,
        "unsupported_by_task": unsupported,
        "unsupported_stressors": sorted({item for values in unsupported.values() for item in values}),
    }


def _is_enabled(value: Any) -> bool:
    if value is False or value is None:
        return False
    if isinstance(value, (list, tuple, dict)):
        return bool(value)
    return bool(value)
