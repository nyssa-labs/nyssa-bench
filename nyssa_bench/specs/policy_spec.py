from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PolicySpec:
    policy_id: str
    description: str = ""
