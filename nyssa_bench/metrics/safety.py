from __future__ import annotations


def safety_metrics(info: dict[str, object]) -> dict[str, float]:
    return {
        "collision_count": float(info.get("collision_count", 0.0)),
        "safety_violation_rate": 1.0 if bool(info.get("safety_violation", False)) else 0.0,
    }
