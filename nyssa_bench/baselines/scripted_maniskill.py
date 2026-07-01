from __future__ import annotations

from typing import Any

import numpy as np

from nyssa_bench.baselines.features import action_bounds, find_vector


class ManiSkillScriptedHeuristic:
    """Simple task-aware controller for common ManiSkill manipulation observations.

    This is a reference scripted baseline, not a guaranteed privileged solver.
    It uses common observation fields when present: tcp/ee pose, object pose,
    and goal/target pose. If those fields are unavailable it emits a zero action
    clipped to the action space.
    """

    def __init__(self, gain: float = 4.0, lift_height: float = 0.18) -> None:
        self.gain = gain
        self.lift_height = lift_height
        self.task_id = ""
        self.stage = 0

    def reset(self, task: Any | None = None, seed: int | None = None) -> None:
        self.task_id = str(getattr(task, "task_id", ""))
        self.stage = 0

    def act(self, observation: dict[str, Any]) -> Any:
        low, high, shape = action_bounds(observation)
        size = int(np.prod(shape))
        action = np.zeros(size, dtype=float)

        tcp = _first_vector(
            observation,
            ("tcp_pose", "ee_pose", "end_effector_pose", "tcp_pos", "ee_pos", "eef_pos"),
        )
        obj = _first_vector(observation, ("obj_pose", "object_pose", "cube_pose", "cube_pos", "obj_pos"))
        goal = _first_vector(observation, ("goal_pose", "target_pose", "goal_pos", "target_pos"))
        if tcp is None or obj is None:
            return np.clip(action.reshape(shape), low, high)

        target = self._target(tcp=tcp, obj=obj, goal=goal)
        delta = np.clip((target - tcp[:3]) * self.gain, -1.0, 1.0)
        action[: min(3, size)] = delta[: min(3, size)]
        if size >= 4:
            action[-1] = self._gripper(tcp=tcp, obj=obj, goal=goal)
        return np.clip(action.reshape(shape), low, high)

    def _target(self, *, tcp: np.ndarray, obj: np.ndarray, goal: np.ndarray | None) -> np.ndarray:
        obj_pos = obj[:3]
        if "push" in self.task_id and goal is not None:
            return obj_pos + np.clip(goal[:3] - obj_pos, -0.05, 0.05)
        if "stack" in self.task_id and goal is not None:
            if np.linalg.norm(tcp[:3] - obj_pos) > 0.06 and self.stage == 0:
                return obj_pos + np.array([0.0, 0.0, 0.05])
            self.stage = 1
            return goal[:3] + np.array([0.0, 0.0, self.lift_height])
        if np.linalg.norm(tcp[:3] - obj_pos) > 0.06 and self.stage == 0:
            return obj_pos + np.array([0.0, 0.0, 0.04])
        self.stage = 1
        return obj_pos + np.array([0.0, 0.0, self.lift_height])

    def _gripper(self, *, tcp: np.ndarray, obj: np.ndarray, goal: np.ndarray | None) -> float:
        if "push" in self.task_id:
            return 1.0
        return -1.0 if np.linalg.norm(tcp[:3] - obj[:3]) < 0.08 else 1.0


def create_scripted_oracle() -> ManiSkillScriptedHeuristic:
    return ManiSkillScriptedHeuristic()


def _first_vector(observation: dict[str, Any], names: tuple[str, ...]) -> np.ndarray | None:
    vector = find_vector(observation, names)
    if vector is None:
        return None
    return vector
