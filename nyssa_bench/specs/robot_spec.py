from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RobotSpec:
    robot_id: str
    description: str = ""
    end_effector: str | None = None
    supported_engines: tuple[str, ...] = ()
