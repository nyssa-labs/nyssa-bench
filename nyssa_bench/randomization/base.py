from __future__ import annotations

from typing import Any


def summarize_randomization(randomization: dict[str, Any]) -> dict[str, Any]:
    return {
        "enabled_keys": sorted(key for key, value in randomization.items() if bool(value)),
        "raw": randomization,
    }
