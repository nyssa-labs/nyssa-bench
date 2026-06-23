from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepRecord:
    observation: dict[str, Any]
    action: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation,
            "action": self.action,
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "info": self.info,
        }


@dataclass
class EpisodeResult:
    task_id: str
    episode_index: int
    seed: int
    success: bool
    failure_label: str | None
    metrics: dict[str, float]
    steps: list[StepRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "episode_index": self.episode_index,
            "seed": self.seed,
            "success": self.success,
            "failure_label": self.failure_label,
            "metrics": self.metrics,
            "steps": [step.to_dict() for step in self.steps],
        }
