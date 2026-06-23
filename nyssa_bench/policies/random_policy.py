from __future__ import annotations

import random
from typing import Any

from nyssa_bench.policies.base import Policy


class RandomPolicy(Policy):
    def __init__(self) -> None:
        self.rng = random.Random()

    def reset(self, task: Any | None = None, seed: int | None = None) -> None:
        self.rng.seed(seed)

    def act(self, observation: dict[str, Any]) -> float:
        return self.rng.uniform(-0.35, 0.35)
