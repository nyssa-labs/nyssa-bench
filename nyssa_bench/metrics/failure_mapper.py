from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nyssa_bench.core.task import TaskSpec


UNKNOWN_FAILURE = "unknown_failure"


@dataclass(frozen=True)
class FailureClassification:
    label: str | None
    source: str
    reason: str


class FailureMapper:
    """Classify failures from environment events and episode terminal info."""

    def classify(
        self,
        info: dict[str, Any],
        *,
        task_spec: TaskSpec | None = None,
        step_count: int = 0,
        terminated: bool = False,
        truncated: bool = False,
    ) -> FailureClassification:
        if bool(info.get("success", False)):
            return FailureClassification(None, "none", "episode succeeded")

        configured = set(task_spec.failure_labels if task_spec is not None else [])
        explicit = info.get("failure_label")
        if explicit:
            return FailureClassification(str(explicit), str(info.get("failure_label_source", "env")), "environment label")

        if _truthy_any(info, "collision", "contact_violation", "safety_violation") or _float(info.get("collision_count")) > 0:
            return _classification("collision", configured, "mapper", "collision or safety event")

        if _truthy_any(info, "wrong_object", "wrong_object_selected"):
            return _classification("wrong_object", configured, "mapper", "wrong object event")

        if _truthy_any(info, "object_slip", "object_dropped", "dropped", "drop"):
            return _classification("object_slip", configured, "mapper", "object slip/drop event")

        if _truthy_any(info, "grasp_failed", "bad_grasp") or info.get("grasp_success") is False:
            return _classification("bad_grasp", configured, "mapper", "grasp failure event")

        if _truthy_any(info, "joint_limit", "joint_limit_failure"):
            return _classification("joint_limit", configured, "mapper", "joint limit event")

        if _truthy_any(info, "planner_stuck", "stuck"):
            return _classification("planner_stuck", configured, "mapper", "planner stuck event")

        if _truthy_any(info, "latency_failure") or _float(info.get("latency_ms")) > _float(info.get("max_latency_ms"), default=1e9):
            return _classification("latency_failure", configured, "mapper", "latency threshold event")

        if _truthy_any(info, "out_of_distribution_layout", "ood_layout", "out_of_distribution"):
            return _classification("out_of_distribution_layout", configured, "mapper", "out-of-distribution event")

        if terminated and "unstable_contact" in configured:
            return FailureClassification("unstable_contact", "mapper", "environment terminated before success")

        max_steps = int((task_spec.success if task_spec is not None else {}).get("max_steps", 0) or 0)
        if truncated or bool(info.get("TimeLimit.truncated", False)) or (max_steps and step_count >= max_steps):
            return _classification("timeout", configured, "mapper", "episode reached step/time limit")

        if "missed_target" in configured:
            return FailureClassification("missed_target", "mapper", "no target success event")

        return FailureClassification(UNKNOWN_FAILURE, "mapper", "no diagnostic event matched")


def _classification(label: str, configured: set[str], source: str, reason: str) -> FailureClassification:
    if label == "joint_limit" and "joint_limit_failure" in configured:
        return FailureClassification("joint_limit_failure", source, reason)
    if label in configured or not configured:
        return FailureClassification(label, source, reason)
    return FailureClassification(UNKNOWN_FAILURE, source, f"{reason}; label not configured")


def _truthy_any(info: dict[str, Any], *keys: str) -> bool:
    return any(bool(info.get(key, False)) for key in keys)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
