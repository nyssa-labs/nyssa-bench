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
        adaptive_margin: str | None = None,
        margin_fraction: float | None = None,
        margin_top_k: int | None = None,
        margin_top_fraction: float | None = None,
        min_margin: float | None = None,
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
        self.adaptive_margin = (
            adaptive_margin if adaptive_margin is not None else os.getenv("NYSSA_MUJOCO_ADAPTIVE_MARGIN", "auto")
        ).strip().lower()
        if self.adaptive_margin not in {"auto", "off", "on"}:
            raise ValueError("NYSSA_MUJOCO_ADAPTIVE_MARGIN must be one of: auto, on, off")
        self.margin_fraction = float(
            margin_fraction if margin_fraction is not None else os.getenv("NYSSA_MUJOCO_MARGIN_FRACTION", "0.25")
        )
        self.margin_top_k = max(
            0,
            int(margin_top_k if margin_top_k is not None else os.getenv("NYSSA_MUJOCO_MARGIN_TOP_K", "2")),
        )
        self.margin_top_fraction = min(
            1.0,
            max(
                0.01,
                float(
                    margin_top_fraction
                    if margin_top_fraction is not None
                    else os.getenv("NYSSA_MUJOCO_MARGIN_TOP_FRACTION", "0.10")
                ),
            ),
        )
        self.min_margin = float(min_margin if min_margin is not None else os.getenv("NYSSA_MUJOCO_MIN_MARGIN", "1e-6"))
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
            "capabilities": [
                "act",
                "recover",
                "score_action",
                "short_horizon_rollout_score",
                "pusher_guided_proposals",
            ],
            "rollout_margin": self.rollout_margin,
            "rollout_horizon": self.rollout_horizon,
            "candidate_count": self.candidate_count,
            "pusher_shaping_scale": self.pusher_shaping_scale,
            "adaptive_margin": self.adaptive_margin,
            "margin_fraction": self.margin_fraction,
            "margin_top_k": self.margin_top_k,
            "margin_top_fraction": self.margin_top_fraction,
            "min_margin": self.min_margin,
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
        candidate_returns = [proposed_return]
        candidates = self._candidate_action_sequences(observation, include_zero=True, task=task, engine=engine)
        for candidate_index, candidate_sequence in enumerate(candidates):
            candidate_return = _evaluate_mujoco_action_sequence(
                engine,
                candidate_sequence,
                task=task,
                pusher_shaping_scale=self.pusher_shaping_scale,
            )
            if candidate_return is not None:
                candidate_returns.append(candidate_return)
                if candidate_return > best_return:
                    best_return = candidate_return
                    best_index = candidate_index
        gap = best_return - proposed_return
        effective_margin, margin_details = self._effective_rollout_margin(task, candidate_returns)
        details = {
            "proposed_return": proposed_return,
            "best_candidate_return": best_return,
            "score_gap": gap,
            "rollout_margin": self.rollout_margin,
            "effective_rollout_margin": effective_margin,
            **margin_details,
            "rollout_horizon": self.rollout_horizon,
            "candidate_count": len(candidates),
            "best_candidate_index": best_index,
            "rollout_score_kind": _mujoco_rollout_score_kind(task),
            "pusher_shaping_scale": self.pusher_shaping_scale if _is_mujoco_pusher_task(task) else None,
        }
        if gap > effective_margin:
            confidence = min(1.0, max(0.0, gap / max(1.0, abs(best_return), abs(proposed_return))))
            return ExpertActionScore(
                accepted=False,
                confidence=confidence,
                reason="lower_than_candidate_reward",
                details=details,
            )
        return ExpertActionScore(accepted=True, confidence=0.75, reason=None, details=details)

    def _effective_rollout_margin(self, task: Any, returns: list[float]) -> tuple[float, dict[str, Any]]:
        adaptive_enabled = self.adaptive_margin == "on" or (
            self.adaptive_margin == "auto" and _is_mujoco_pusher_task(task)
        )
        if not adaptive_enabled or not returns:
            return self.rollout_margin, {
                "adaptive_margin_enabled": False,
                "candidate_return_spread": None,
                "candidate_top_return_spread": None,
                "margin_top_count": None,
                "margin_top_k": self.margin_top_k,
            }
        sorted_returns = sorted(returns, reverse=True)
        return_spread = sorted_returns[0] - sorted_returns[-1]
        if self.margin_top_k > 0:
            top_count = min(len(sorted_returns), max(2, self.margin_top_k))
        else:
            top_count = max(2, min(len(sorted_returns), int(len(sorted_returns) * self.margin_top_fraction)))
        top_returns = sorted_returns[:top_count]
        top_spread = top_returns[0] - top_returns[-1] if top_returns else 0.0
        reference_spread = top_spread if top_spread > 0.0 else return_spread
        if reference_spread <= 0.0:
            adaptive_margin = self.min_margin
        else:
            adaptive_margin = max(self.min_margin, reference_spread * self.margin_fraction)
        return min(self.rollout_margin, adaptive_margin), {
            "adaptive_margin_enabled": True,
            "candidate_return_spread": return_spread,
            "candidate_top_return_spread": top_spread,
            "margin_top_count": top_count,
            "margin_top_k": self.margin_top_k,
        }

    def _best_rollout_candidate(self, observation: dict[str, Any], *, task: Any, engine: Any) -> Any | None:
        best_action = None
        best_return = None
        for candidate_sequence in self._candidate_action_sequences(
            observation,
            include_zero=True,
            task=task,
            engine=engine,
        ):
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

    def _candidate_action_sequences(
        self,
        observation: dict[str, Any],
        *,
        include_zero: bool,
        task: Any | None = None,
        engine: Any | None = None,
    ) -> list[list[Any]]:
        candidates = [
            [candidate] * self.rollout_horizon
            for candidate in self._candidate_actions(observation, include_zero=include_zero)
        ]
        candidates.extend(self._pusher_guided_action_sequences(observation, task=task, engine=engine))
        candidates.extend(self._random_action_sequences(observation))
        return candidates

    def _pusher_guided_action_sequences(
        self,
        observation: dict[str, Any],
        *,
        task: Any | None,
        engine: Any | None,
    ) -> list[list[Any]]:
        if engine is None or not _is_mujoco_pusher_task(task):
            return []
        try:
            import numpy as np

            object_pos = _first_body_com(engine, ("object", "obj", "puck", "object0"))
            goal_pos = _first_body_com(engine, ("goal", "target"))
            tip_pos = _first_body_com(engine, ("tips_arm", "fingertip", "tip", "end_effector"))
            if object_pos is None or goal_pos is None or tip_pos is None:
                return []
            goal_direction = np.asarray(goal_pos[:2], dtype=float) - np.asarray(object_pos[:2], dtype=float)
            goal_norm = float(np.linalg.norm(goal_direction))
            if goal_norm <= 1e-8:
                return []
            goal_direction = goal_direction / goal_norm
            behind_object = np.asarray(object_pos[:2], dtype=float) - 0.18 * goal_direction
            tip_xy = np.asarray(tip_pos[:2], dtype=float)
            approach_delta = behind_object - tip_xy
            push_delta = goal_direction

            sequences: list[list[Any]] = []
            approach = self._pusher_action_for_tip_delta(observation, engine=engine, desired_delta=approach_delta)
            push = self._pusher_action_for_tip_delta(observation, engine=engine, desired_delta=push_delta)
            if approach is not None:
                sequences.append([approach] * self.rollout_horizon)
            if push is not None:
                sequences.append([push] * self.rollout_horizon)
            if approach is not None and push is not None:
                split = max(1, self.rollout_horizon // 2)
                sequences.append([approach] * split + [push] * max(1, self.rollout_horizon - split))
                alternating = [approach if index % 2 == 0 else push for index in range(self.rollout_horizon)]
                sequences.append(alternating)
            return [sequence[: self.rollout_horizon] for sequence in sequences if len(sequence) >= self.rollout_horizon]
        except Exception:
            return []

    def _pusher_action_for_tip_delta(
        self,
        observation: dict[str, Any],
        *,
        engine: Any,
        desired_delta: Any,
    ) -> Any | None:
        try:
            import numpy as np

            low, high, shape = action_bounds(observation)
            size = int(_prod(shape))
            desired = np.asarray(desired_delta, dtype=float).reshape(-1)[:2]
            desired_norm = float(np.linalg.norm(desired))
            if desired_norm <= 1e-8:
                return np.zeros(shape, dtype=float)
            desired = desired / desired_norm * min(0.08, desired_norm)
            base_tip = _first_body_com(engine, ("tips_arm", "fingertip", "tip", "end_effector"))
            if base_tip is None:
                return None
            columns = []
            eps = np.maximum(0.05, 0.25 * np.maximum(np.abs(high.reshape(-1)), 1.0))
            for index in range(size):
                action = np.zeros(size, dtype=float)
                action[index] = float(eps[index])
                moved_tip = _evaluate_mujoco_terminal_body_com(
                    engine,
                    action.reshape(shape),
                    ("tips_arm", "fingertip", "tip", "end_effector"),
                )
                if moved_tip is None:
                    columns.append(np.zeros(2, dtype=float))
                else:
                    columns.append((np.asarray(moved_tip[:2], dtype=float) - np.asarray(base_tip[:2], dtype=float)) / eps[index])
            jacobian = np.stack(columns, axis=1)
            if not np.any(np.abs(jacobian) > 1e-10):
                return None
            solution, *_ = np.linalg.lstsq(jacobian, desired, rcond=None)
            return np.clip(solution.reshape(shape), low, high)
        except Exception:
            return None

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


def _evaluate_mujoco_terminal_body_com(engine: Any, action: Any, body_names: tuple[str, ...]) -> Any | None:
    env = getattr(engine, "env", None)
    if env is None:
        return None
    snapshot = _snapshot_mujoco_engine(engine)
    if snapshot is None:
        return None
    try:
        env.step(action)
        return _first_body_com(engine, body_names)
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
