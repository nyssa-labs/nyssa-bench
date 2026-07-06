from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nyssa_bench.baselines.features import action_bounds, fit_action_to_observation, flatten_observation


@dataclass(frozen=True)
class ExpertActionScore:
    accepted: bool = True
    confidence: float | None = None
    reason: str | None = None
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "confidence": self.confidence,
            "reason": self.reason,
            "details": self.details,
        }


class ExpertProvider:
    """Interface for planner, MPC, scripted, or learned expert assistance."""

    provider_id = "expert"

    def reset(self, *, task: Any | None = None, seed: int | None = None, engine: Any | None = None) -> None:
        return None

    def act(self, observation: dict[str, Any], *, task: Any, engine: Any | None = None) -> Any | None:
        return None

    def plan(
        self,
        *,
        state: dict[str, Any],
        goal: dict[str, Any] | None,
        task: Any,
        engine: Any | None = None,
    ) -> list[Any] | None:
        return None

    def recover(
        self,
        *,
        state: dict[str, Any],
        failure: str | None,
        task: Any,
        engine: Any | None = None,
    ) -> list[Any] | None:
        return None

    def score_action(
        self,
        observation: dict[str, Any],
        action: Any,
        *,
        task: Any,
        engine: Any | None = None,
    ) -> ExpertActionScore:
        return ExpertActionScore()

    def metadata(self) -> dict[str, Any]:
        return {"provider_id": self.provider_id, "capabilities": []}

    def close(self) -> None:
        return None


class NoOpExpertProvider(ExpertProvider):
    provider_id = "none"


class PolicyExpertProvider(ExpertProvider):
    """Use any Nyssa policy as an expert action source."""

    def __init__(self, policy_name: str) -> None:
        from nyssa_bench.core.registry import make_policy

        self.provider_id = f"policy:{policy_name}"
        self.policy_name = policy_name
        self.policy = make_policy(policy_name)

    def reset(self, *, task: Any | None = None, seed: int | None = None, engine: Any | None = None) -> None:
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset(task=task, seed=seed)

    def act(self, observation: dict[str, Any], *, task: Any, engine: Any | None = None) -> Any | None:
        return self.policy.act(observation)

    def recover(
        self,
        *,
        state: dict[str, Any],
        failure: str | None,
        task: Any,
        engine: Any | None = None,
    ) -> list[Any] | None:
        observation = state.get("observation")
        if not isinstance(observation, dict):
            return None
        return [self.act(observation, task=task, engine=engine)]

    def metadata(self) -> dict[str, Any]:
        return {"provider_id": self.provider_id, "capabilities": ["act"]}

    def close(self) -> None:
        close = getattr(self.policy, "close", None)
        if callable(close):
            close()


class BoundsVerifierExpertProvider(ExpertProvider):
    """Verifier that rejects actions that need clipping to fit the live action space."""

    provider_id = "bounds-verifier"

    def act(self, observation: dict[str, Any], *, task: Any, engine: Any | None = None) -> Any | None:
        low, high, shape = action_bounds(observation)
        del high
        return fit_action_to_observation(low.reshape(shape), observation)

    def score_action(
        self,
        observation: dict[str, Any],
        action: Any,
        *,
        task: Any,
        engine: Any | None = None,
    ) -> ExpertActionScore:
        fitted = fit_action_to_observation(action, observation)
        try:
            import numpy as np

            original = np.asarray(action, dtype=float).reshape(-1)
            clipped = np.asarray(fitted, dtype=float).reshape(-1)
            size = min(original.size, clipped.size)
            if size and not np.allclose(original[:size], clipped[:size], atol=1e-6):
                return ExpertActionScore(accepted=False, confidence=1.0, reason="action_out_of_bounds")
        except Exception:
            return ExpertActionScore(accepted=True, confidence=None, reason=None)
        return ExpertActionScore(accepted=True, confidence=1.0, reason=None)

    def metadata(self) -> dict[str, Any]:
        return {"provider_id": self.provider_id, "capabilities": ["score_action", "act"]}


class ManiSkillScriptedExpertProvider(ExpertProvider):
    """Built-in ManiSkill scripted expert and recovery provider."""

    provider_id = "maniskill-scripted"

    def __init__(self) -> None:
        from nyssa_bench.baselines.scripted_maniskill import ManiSkillScriptedHeuristic

        self.controller = ManiSkillScriptedHeuristic()
        self._bounds = BoundsVerifierExpertProvider()

    def reset(self, *, task: Any | None = None, seed: int | None = None, engine: Any | None = None) -> None:
        self.controller.reset(task=task, seed=seed)

    def act(self, observation: dict[str, Any], *, task: Any, engine: Any | None = None) -> Any | None:
        return self.controller.act(observation)

    def recover(
        self,
        *,
        state: dict[str, Any],
        failure: str | None,
        task: Any,
        engine: Any | None = None,
    ) -> list[Any] | None:
        observation = state.get("observation")
        if not isinstance(observation, dict):
            return None
        action = self.act(observation, task=task, engine=engine)
        return [action] if action is not None else None

    def score_action(
        self,
        observation: dict[str, Any],
        action: Any,
        *,
        task: Any,
        engine: Any | None = None,
    ) -> ExpertActionScore:
        bounds_score = self._bounds.score_action(observation, action, task=task, engine=engine)
        if not bounds_score.accepted:
            return bounds_score
        return ExpertActionScore(accepted=True, confidence=0.75, reason=None)

    def metadata(self) -> dict[str, Any]:
        return {"provider_id": self.provider_id, "capabilities": ["act", "recover", "score_action"]}


class MuJoCoHeuristicExpertProvider(ExpertProvider):
    """MuJoCo heuristic expert with calibrated short-horizon rollout scoring."""

    provider_id = "mujoco-heuristic"

    def __init__(
        self,
        gain: float = 0.5,
        reject_idle_threshold: float = 1e-6,
        rollout_margin: float | None = None,
        rollout_horizon: int | None = None,
        candidate_count: int | None = None,
        random_seed: int | None = None,
        pusher_shaping_scale: float | None = None,
    ) -> None:
        self.gain = gain
        self.reject_idle_threshold = reject_idle_threshold
        self.rollout_margin = float(
            rollout_margin if rollout_margin is not None else os.getenv("NYSSA_MUJOCO_ROLLOUT_MARGIN", "0.25")
        )
        self.rollout_horizon = max(
            1,
            int(rollout_horizon if rollout_horizon is not None else os.getenv("NYSSA_MUJOCO_ROLLOUT_HORIZON", "3")),
        )
        self.candidate_count = max(
            0,
            int(candidate_count if candidate_count is not None else os.getenv("NYSSA_MUJOCO_CANDIDATES", "32")),
        )
        self.random_seed = int(random_seed if random_seed is not None else os.getenv("NYSSA_MUJOCO_SEED", "0"))
        self.pusher_shaping_scale = float(
            pusher_shaping_scale
            if pusher_shaping_scale is not None
            else os.getenv("NYSSA_MUJOCO_PUSHER_SHAPING", "5.0")
        )
        self._rng: Any | None = None
        self._bounds = BoundsVerifierExpertProvider()

    def reset(self, *, task: Any | None = None, seed: int | None = None, engine: Any | None = None) -> None:
        del task, engine
        try:
            import numpy as np

            self._rng = np.random.default_rng(self.random_seed if seed is None else self.random_seed + int(seed))
        except Exception:
            self._rng = None

    def act(self, observation: dict[str, Any], *, task: Any, engine: Any | None = None) -> Any | None:
        if engine is not None:
            candidate = self._best_rollout_candidate(observation, task=task, engine=engine)
            if candidate is not None:
                return candidate
        return self._heuristic_action(observation)

    def _heuristic_action(self, observation: dict[str, Any]) -> Any | None:
        low, high, shape = action_bounds(observation)
        size = int(_prod(shape))
        features = flatten_observation(observation, max_dim=max(size, 1))
        action = -self.gain * features[:size]
        try:
            import numpy as np

            action = np.asarray(action, dtype=float).reshape(shape)
            return np.clip(action, low, high)
        except Exception:
            return fit_action_to_observation(action, observation)

    def recover(
        self,
        *,
        state: dict[str, Any],
        failure: str | None,
        task: Any,
        engine: Any | None = None,
    ) -> list[Any] | None:
        observation = state.get("observation")
        if not isinstance(observation, dict):
            return None
        action = self.act(observation, task=task, engine=engine)
        return [action] if action is not None else None

    def score_action(
        self,
        observation: dict[str, Any],
        action: Any,
        *,
        task: Any,
        engine: Any | None = None,
    ) -> ExpertActionScore:
        bounds_score = self._bounds.score_action(observation, action, task=task, engine=engine)
        if not bounds_score.accepted:
            return bounds_score
        score = self._rollout_score_against_candidates(observation, action, task=task, engine=engine)
        if score is not None:
            return score
        try:
            import numpy as np

            action_norm = float(np.linalg.norm(np.asarray(action, dtype=float).reshape(-1)))
            state_norm = float(np.linalg.norm(flatten_observation(observation, max_dim=16)))
            if state_norm > 1e-3 and action_norm <= self.reject_idle_threshold:
                return ExpertActionScore(accepted=False, confidence=0.5, reason="idle_action_on_nonzero_state")
        except Exception:
            pass
        return ExpertActionScore(accepted=True, confidence=0.5, reason=None)

    def metadata(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "capabilities": ["act", "recover", "score_action", "short_horizon_rollout_score"],
            "rollout_margin": self.rollout_margin,
            "rollout_horizon": self.rollout_horizon,
            "candidate_count": self.candidate_count,
            "pusher_shaping_scale": self.pusher_shaping_scale,
        }

    def _rollout_score_against_candidates(
        self,
        observation: dict[str, Any],
        action: Any,
        *,
        task: Any,
        engine: Any | None,
    ) -> ExpertActionScore | None:
        if engine is None:
            return None
        proposed_return = _evaluate_mujoco_action_sequence(
            engine,
            [action] * self.rollout_horizon,
            task=task,
            pusher_shaping_scale=self.pusher_shaping_scale,
        )
        if proposed_return is None:
            return None
        best_return = proposed_return
        best_index = None
        candidates = self._candidate_action_sequences(observation, include_zero=True)
        for candidate_index, candidate_sequence in enumerate(candidates):
            candidate_return = _evaluate_mujoco_action_sequence(
                engine,
                candidate_sequence,
                task=task,
                pusher_shaping_scale=self.pusher_shaping_scale,
            )
            if candidate_return is not None and candidate_return > best_return:
                best_return = candidate_return
                best_index = candidate_index
        gap = best_return - proposed_return
        details = {
            "proposed_return": proposed_return,
            "best_candidate_return": best_return,
            "score_gap": gap,
            "rollout_margin": self.rollout_margin,
            "rollout_horizon": self.rollout_horizon,
            "candidate_count": len(candidates),
            "best_candidate_index": best_index,
            "rollout_score_kind": _mujoco_rollout_score_kind(task),
            "pusher_shaping_scale": self.pusher_shaping_scale if _is_mujoco_pusher_task(task) else None,
        }
        if gap > self.rollout_margin:
            confidence = min(1.0, max(0.0, gap / max(1.0, abs(best_return), abs(proposed_return))))
            return ExpertActionScore(
                accepted=False,
                confidence=confidence,
                reason="lower_than_candidate_reward",
                details=details,
            )
        return ExpertActionScore(accepted=True, confidence=0.75, reason=None, details=details)

    def _best_rollout_candidate(self, observation: dict[str, Any], *, task: Any, engine: Any) -> Any | None:
        best_action = None
        best_return = None
        for candidate_sequence in self._candidate_action_sequences(observation, include_zero=True):
            candidate_return = _evaluate_mujoco_action_sequence(
                engine,
                candidate_sequence,
                task=task,
                pusher_shaping_scale=self.pusher_shaping_scale,
            )
            if candidate_return is not None and (best_return is None or candidate_return > best_return):
                best_return = candidate_return
                best_action = candidate_sequence[0] if candidate_sequence else None
        return best_action

    def _candidate_action_sequences(self, observation: dict[str, Any], *, include_zero: bool) -> list[list[Any]]:
        candidates = [[candidate] * self.rollout_horizon for candidate in self._candidate_actions(observation, include_zero=include_zero)]
        candidates.extend(self._random_action_sequences(observation))
        return candidates

    def _candidate_actions(self, observation: dict[str, Any], *, include_zero: bool) -> list[Any]:
        try:
            import numpy as np

            low, high, shape = action_bounds(observation)
            heuristic = self._heuristic_action(observation)
            candidates = []
            if heuristic is not None:
                heuristic_array = np.asarray(heuristic, dtype=float).reshape(shape)
                candidates.append(heuristic_array)
                candidates.append(np.clip(-heuristic_array, low, high))
            if include_zero:
                candidates.append(np.zeros(shape, dtype=float))
            candidates.append(low)
            candidates.append(high)
            size = int(_prod(shape))
            if size <= 4:
                for mask in range(1 << size):
                    corner = np.array(
                        [high.reshape(-1)[idx] if (mask >> idx) & 1 else low.reshape(-1)[idx] for idx in range(size)],
                        dtype=float,
                    ).reshape(shape)
                    candidates.append(corner)
            return _unique_arrays(candidates)
        except Exception:
            fallback = self._heuristic_action(observation)
            return [fallback] if fallback is not None else []

    def _random_action_sequences(self, observation: dict[str, Any]) -> list[list[Any]]:
        if self.candidate_count <= 0:
            return []
        try:
            import numpy as np

            low, high, shape = action_bounds(observation)
            rng = self._rng if self._rng is not None else np.random.default_rng(self.random_seed)
            sequences: list[list[Any]] = []
            for _ in range(self.candidate_count):
                sequence = []
                for _ in range(self.rollout_horizon):
                    sequence.append(rng.uniform(low=low, high=high, size=shape))
                sequences.append(sequence)
            return sequences
        except Exception:
            return []


def make_expert_provider(provider: str | Path | ExpertProvider | None) -> ExpertProvider:
    if provider is None:
        return NoOpExpertProvider()
    if isinstance(provider, ExpertProvider):
        return provider
    name = str(provider)
    if name in {"", "none", "noop", "no-op"}:
        return NoOpExpertProvider()
    if name in {"bounds", "bounds-verifier"}:
        return BoundsVerifierExpertProvider()
    if name in {"scripted", "scripted-oracle", "maniskill-scripted"}:
        return ManiSkillScriptedExpertProvider()
    if name in {"mujoco", "mujoco-heuristic", "mujoco-random-shooting", "random-shooting"}:
        return MuJoCoHeuristicExpertProvider()
    if name.startswith("policy:"):
        return PolicyExpertProvider(name.split(":", 1)[1])
    path = Path(name)
    if path.exists() or path.suffix == ".py":
        return _load_expert_from_path(path)
    raise ValueError(f"Unknown expert provider '{name}'. Available built-in provider: none")


def _load_expert_from_path(path: Path) -> ExpertProvider:
    import importlib.util

    spec = importlib.util.spec_from_file_location("nyssa_user_expert", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load expert provider from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    factory = getattr(module, "create_expert_provider", None)
    if callable(factory):
        provider = factory()
    else:
        provider_cls = getattr(module, "ExpertProviderAdapter", None)
        provider = provider_cls() if provider_cls is not None else None
    if not isinstance(provider, ExpertProvider):
        raise TypeError(
            f"Expert provider file {path} must expose create_expert_provider() or ExpertProviderAdapter "
            "returning an ExpertProvider instance."
        )
    return provider


def _prod(values: tuple[int, ...]) -> int:
    total = 1
    for value in values:
        total *= int(value)
    return total


def _evaluate_mujoco_action(engine: Any, action: Any) -> float | None:
    return _evaluate_mujoco_action_sequence(engine, [action])


def _evaluate_mujoco_action_sequence(
    engine: Any,
    actions: list[Any],
    *,
    task: Any | None = None,
    pusher_shaping_scale: float = 0.0,
) -> float | None:
    env = getattr(engine, "env", None)
    if env is None:
        return None
    snapshot = _snapshot_mujoco_engine(engine)
    if snapshot is None:
        return None
    try:
        total_reward = 0.0
        for action in actions:
            _, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        shaping = _mujoco_terminal_shaping(engine, task=task, pusher_shaping_scale=pusher_shaping_scale)
        if shaping is not None:
            total_reward += shaping
        return total_reward
    except Exception:
        return None
    finally:
        _restore_mujoco_engine(engine, snapshot)


def _mujoco_terminal_shaping(engine: Any, *, task: Any | None, pusher_shaping_scale: float) -> float | None:
    if not _is_mujoco_pusher_task(task) or pusher_shaping_scale <= 0.0:
        return None
    try:
        import numpy as np

        object_pos = _first_body_com(engine, ("object", "obj", "puck", "object0"))
        goal_pos = _first_body_com(engine, ("goal", "target"))
        arm_pos = _first_body_com(engine, ("tips_arm", "fingertip", "tip", "end_effector"))
        if object_pos is None or goal_pos is None:
            return None
        goal_distance = float(np.linalg.norm(object_pos[:2] - goal_pos[:2]))
        arm_distance = float(np.linalg.norm(arm_pos[:2] - object_pos[:2])) if arm_pos is not None else 0.0
        return pusher_shaping_scale * (-(goal_distance + 0.1 * arm_distance))
    except Exception:
        return None


def _first_body_com(engine: Any, names: tuple[str, ...]) -> Any | None:
    env = getattr(engine, "env", None)
    unwrapped = getattr(env, "unwrapped", env)
    getter = getattr(unwrapped, "get_body_com", None)
    if not callable(getter):
        return None
    for name in names:
        try:
            return getter(name)
        except Exception:
            continue
    return None


def _is_mujoco_pusher_task(task: Any | None) -> bool:
    return str(getattr(task, "task_id", "")).lower() == "mujoco_pusher"


def _mujoco_rollout_score_kind(task: Any | None) -> str:
    return "task_shaped_return" if _is_mujoco_pusher_task(task) else "env_return"


def _snapshot_mujoco_engine(engine: Any) -> dict[str, Any] | None:
    try:
        import numpy as np

        env = getattr(engine, "env", None)
        unwrapped = getattr(env, "unwrapped", env)
        data = getattr(unwrapped, "data", None)
        if data is None:
            return None
        snapshot = {
            "qpos": np.array(data.qpos, copy=True),
            "qvel": np.array(data.qvel, copy=True),
            "time": float(getattr(data, "time", 0.0)),
            "engine_episode_return": getattr(engine, "episode_return", None),
            "engine_elapsed_steps": getattr(engine, "elapsed_steps", None),
            "elapsed_steps": _collect_wrapper_attr(env, "_elapsed_steps"),
        }
        return snapshot
    except Exception:
        return None


def _restore_mujoco_engine(engine: Any, snapshot: dict[str, Any]) -> None:
    try:
        env = getattr(engine, "env", None)
        unwrapped = getattr(env, "unwrapped", env)
        if hasattr(unwrapped, "set_state"):
            unwrapped.set_state(snapshot["qpos"], snapshot["qvel"])
        else:
            data = getattr(unwrapped, "data", None)
            if data is not None:
                data.qpos[:] = snapshot["qpos"]
                data.qvel[:] = snapshot["qvel"]
                if hasattr(data, "time"):
                    data.time = snapshot["time"]
        data = getattr(unwrapped, "data", None)
        if data is not None and hasattr(data, "time"):
            data.time = snapshot["time"]
        model = getattr(unwrapped, "model", None)
        if model is not None and data is not None:
            try:
                import mujoco

                mujoco.mj_forward(model, data)
            except Exception:
                pass
        _restore_wrapper_attr(env, "_elapsed_steps", snapshot.get("elapsed_steps", []))
        if snapshot.get("engine_episode_return") is not None:
            engine.episode_return = snapshot["engine_episode_return"]
        if snapshot.get("engine_elapsed_steps") is not None:
            engine.elapsed_steps = snapshot["engine_elapsed_steps"]
    except Exception:
        return None


def _collect_wrapper_attr(env: Any, attr: str) -> list[tuple[Any, Any]]:
    values = []
    current = env
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, attr):
            values.append((current, getattr(current, attr)))
        current = getattr(current, "env", None)
    return values


def _restore_wrapper_attr(env: Any, attr: str, values: list[tuple[Any, Any]]) -> None:
    del env
    for wrapper, value in values:
        try:
            setattr(wrapper, attr, value)
        except Exception:
            pass


def _unique_arrays(values: list[Any]) -> list[Any]:
    try:
        import numpy as np

        unique = []
        seen = set()
        for value in values:
            array = np.asarray(value, dtype=float)
            key = tuple(array.reshape(-1).round(8).tolist())
            if key not in seen:
                seen.add(key)
                unique.append(array)
        return unique
    except Exception:
        return values
