from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from nyssa_bench.baselines.features import fit_action_to_observation
from nyssa_bench.baselines.simple_bc import task_checkpoint_key
from nyssa_bench.policies.base import Policy


class DemoReplayPolicy(Policy):
    """Task-routed replay of imported demonstration actions.

    This is a strong teacher/reference baseline for checking that imported
    planner demonstrations can execute through Nyssa's live evaluator. It is not
    a learned policy: it replays successful demo action sequences by task.
    """

    def __init__(self, demo_dir: str | Path | None = None) -> None:
        self.demo_dir = Path(demo_dir or os.getenv("NYSSA_DEMO_REPLAY_DIR", "benchmark_results/maniskill_official_demos_import_v2"))
        self.current_task_id: str | None = None
        self.current_actions: list[Any] = []
        self.cursor = 0
        self._episodes_by_task: dict[str, list[dict[str, Any]]] = {}

    def reset(self, task: Any | None = None, seed: int | None = None) -> None:
        self.current_task_id = task_checkpoint_key(str(getattr(task, "task_id", "")))
        episodes = self._load_task_episodes(self.current_task_id)
        if not episodes:
            self.current_actions = []
            self.cursor = 0
            return
        index = int(seed or 0) % len(episodes)
        self.current_actions = [step.get("action") for step in episodes[index].get("steps", [])]
        self.cursor = 0

    def act(self, observation: dict[str, Any]) -> Any:
        if not self.current_actions:
            return fit_action_to_observation(0.0, observation)
        index = min(self.cursor, len(self.current_actions) - 1)
        self.cursor += 1
        return fit_action_to_observation(self.current_actions[index], observation)

    def _load_task_episodes(self, task_id: str) -> list[dict[str, Any]]:
        if task_id not in self._episodes_by_task:
            path = self.demo_dir / task_id / "episodes.json"
            if not path.exists():
                raise RuntimeError(
                    f"Demo replay episodes not found for task '{task_id}': {path}. "
                    "Set NYSSA_DEMO_REPLAY_DIR to an imported ManiSkill demo directory."
                )
            episodes = json.loads(path.read_text(encoding="utf-8"))
            self._episodes_by_task[task_id] = [episode for episode in episodes if bool(episode.get("success", False))]
        return self._episodes_by_task[task_id]
