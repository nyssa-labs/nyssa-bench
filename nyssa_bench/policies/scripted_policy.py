from __future__ import annotations

from typing import Any

from nyssa_bench.policies.base import Policy


class ScriptedPolicy(Policy):
    """Simple oracle-like controller for the dummy engine."""

    def act(self, observation: dict[str, Any]) -> float:
        state = observation.get("state", {})
        distance = float(state.get("distance", 0.0))
        return max(min(distance * 0.6, 0.3), -0.3)
