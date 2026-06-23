from __future__ import annotations


def robustness_metrics(info: dict[str, object]) -> dict[str, float]:
    failure_label = info.get("failure_label")
    return {
        "object_slip_rate": 1.0 if failure_label == "object_slip" else 0.0,
        "wrong_object_rate": 1.0 if failure_label == "wrong_object" else 0.0,
        "out_of_distribution_failure_rate": 1.0 if failure_label == "out_of_distribution_layout" else 0.0,
    }
