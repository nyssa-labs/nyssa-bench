from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nyssa_bench.baselines.features import action_bounds, fit_action_to_observation, flatten_observation


@dataclass(frozen=True)
class ExpertActionScore:
    accepted: bool = True
    confidence: float | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "confidence": self.confidence,
            "reason": self.reason,
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
    """Simple MuJoCo low-dimensional controller used as expert/recovery scaffold."""

    provider_id = "mujoco-heuristic"

    def __init__(self, gain: float = 0.5, reject_idle_threshold: float = 1e-6) -> None:
        self.gain = gain
        self.reject_idle_threshold = reject_idle_threshold
        self._bounds = BoundsVerifierExpertProvider()

    def act(self, observation: dict[str, Any], *, task: Any, engine: Any | None = None) -> Any | None:
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
        return {"provider_id": self.provider_id, "capabilities": ["act", "recover", "score_action"]}


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
