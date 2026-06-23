from __future__ import annotations


from typing import Any


def sim_to_real_score(metrics: dict[str, float]) -> float:
    """Prototype score for future sim-to-real ranking from flat metrics."""

    success = metrics.get("success_rate", 0.0)
    safety = 1.0 - metrics.get("safety_violation_rate", 0.0)
    robustness = 1.0 - metrics.get("out_of_distribution_failure_rate", 0.0)
    return max(0.0, min(1.0, 0.5 * success + 0.25 * safety + 0.25 * robustness))


def score_summary(summary: dict[str, Any]) -> float:
    """Score a NyssaBench run summary on a 0..1 sim-to-real readiness scale."""

    metric_means = dict(summary.get("metrics", {}))
    flat = {
        "success_rate": float(summary.get("success_rate", 0.0)),
        "safety_violation_rate": float(metric_means.get("safety_violation_rate", 0.0)),
        "out_of_distribution_failure_rate": float(metric_means.get("out_of_distribution_failure_rate", 0.0)),
    }
    return sim_to_real_score(flat)
