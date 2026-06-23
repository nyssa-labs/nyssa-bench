from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RandomizationSpec:
    values: dict[str, Any] = field(default_factory=dict)
