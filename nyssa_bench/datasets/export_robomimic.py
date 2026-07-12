from __future__ import annotations

import json
from pathlib import Path

from typing import Any

from nyssa_bench.baselines.features import flatten_observation, normalize_action
from nyssa_bench.core.episode import EpisodeResult


def export_robomimic_hdf5(
    episodes: list[EpisodeResult],
    path: str | Path,
    *,
    feature_dim: int = 256,
) -> Path:
    try:
        import h5py
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("RoboMimic export requires: uv sync --extra dataset") from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        data.attrs["env_args"] = json.dumps(
            {
                "env_name": "NyssaFlat-v0",
                "type": 0,
                "env_kwargs": {
                    "description": "Flat low-dimensional NyssaBench export for offline RoboMimic BC training.",
                },
            }
        )
        total = 0
        for index, episode in enumerate(episodes):
            group = data.create_group(f"demo_{index}")
            observations = [flatten_observation(_without_simulator_state(step.observation), feature_dim) for step in episode.steps]
            action_size = _action_size(episode)
            actions = [normalize_action(step.action, action_size) for step in episode.steps]
            rewards = [step.reward for step in episode.steps]
            dones = [bool(step.terminated or step.truncated) for step in episode.steps]
            if dones:
                dones[-1] = True
            group.create_dataset("actions", data=np.asarray(actions, dtype=float))
            group.create_dataset("rewards", data=np.asarray(rewards, dtype=float))
            group.create_dataset("dones", data=np.asarray(dones, dtype=bool))
            obs_group = group.create_group("obs")
            next_obs_group = group.create_group("next_obs")
            obs_array = np.asarray(observations, dtype=float)
            obs_group.create_dataset("flat", data=obs_array)
            if len(obs_array) > 1:
                next_flat = np.vstack([obs_array[1:], obs_array[-1:]])
            else:
                next_flat = obs_array
            next_obs_group.create_dataset("flat", data=next_flat)
            group.attrs["num_samples"] = len(episode.steps)
            group.attrs["task_id"] = episode.task_id
            total += len(episode.steps)
        data.attrs["total"] = total
    return path


def _action_size(episode: EpisodeResult) -> int:
    for step in episode.steps:
        spec = step.observation.get("action_space", {})
        shape = spec.get("shape")
        if shape:
            size = 1
            for value in shape:
                size *= int(value)
            return size
    return 1


def _without_simulator_state(observation: dict[str, Any]) -> dict[str, Any]:
    raw = observation.get("raw", observation)
    if not isinstance(raw, dict):
        return observation
    filtered_raw = {key: value for key, value in raw.items() if key not in {"env_states", "states", "state"}}
    if "raw" not in observation:
        return filtered_raw
    return {**observation, "raw": filtered_raw}
