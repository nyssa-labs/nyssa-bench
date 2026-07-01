from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nyssa_bench.core.episode import EpisodeResult
from nyssa_bench.core.suite import Suite
from nyssa_bench.core.task import TaskSpec


PUBLIC_CLAIM_ENGINES = {"maniskill", "mujoco"}
EXPERIMENTAL_ENGINES = {"genesis", "robocasa"}
MIN_PUBLIC_EPISODES_PER_TASK = 100


@dataclass(frozen=True)
class RunClaimValidation:
    public_claim: bool
    benchmark_tier: str
    status: str
    checks: dict[str, bool]
    failures: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "public_claim": self.public_claim,
            "benchmark_tier": self.benchmark_tier,
            "checks": self.checks,
            "failures": self.failures,
            "warnings": self.warnings,
        }


class RunClaimValidator:
    """Conservative gate for publishing benchmark claims."""

    def validate(
        self,
        *,
        suite: Suite,
        engine_name: str,
        episodes_per_task: int,
        episodes: list[EpisodeResult],
        out_dir: str | Path | None,
        package_versions: dict[str, Any] | None = None,
        git_info: dict[str, Any] | None = None,
    ) -> RunClaimValidation:
        checks = {
            "supported_real_simulator_backend": engine_name in PUBLIC_CLAIM_ENGINES,
            "non_experimental_backend": engine_name not in EXPERIMENTAL_ENGINES,
            "explicit_task_mappings": all(_has_explicit_mapping(task, engine_name) for task in suite.tasks),
            "success_predicates_mapped": all(_has_success_predicate(task) for task in suite.tasks),
            "minimum_episodes_per_task": episodes_per_task >= MIN_PUBLIC_EPISODES_PER_TASK,
            "episode_evidence": _has_episode_evidence(episodes),
            "replay_or_episode_evidence": _has_replay_or_episode_evidence(episodes),
            "diagnosed_failure_labels": _has_diagnosed_failures(episodes),
            "package_versions_present": bool(package_versions),
            "git_info_present": bool(git_info),
            "artifact_directory_present": out_dir is not None,
        }
        failures = [name for name, passed in checks.items() if not passed]
        warnings = _warnings(suite.tasks, engine_name)
        public_claim = not failures
        if public_claim:
            tier = "real"
            status = "validated"
        elif engine_name in EXPERIMENTAL_ENGINES:
            tier = "experimental_contract_only"
            status = "not_public"
        else:
            tier = "prototype"
            status = "not_public"
        return RunClaimValidation(
            public_claim=public_claim,
            benchmark_tier=tier,
            status=status,
            checks=checks,
            failures=failures,
            warnings=warnings,
        )


def _has_explicit_mapping(task: TaskSpec, engine_name: str) -> bool:
    engine_env_ids = task.success.get("engine_env_ids", {})
    engine_factory = task.success.get("engine_factory", {})
    return (
        isinstance(engine_env_ids, dict)
        and bool(engine_env_ids.get(engine_name))
        or isinstance(engine_factory, dict)
        and bool(engine_factory.get(engine_name))
    )


def _has_success_predicate(task: TaskSpec) -> bool:
    success = task.success
    predicate_keys = {
        "success_info_keys",
        "success_metric",
        "reward_threshold",
        "return_threshold",
        "min_success_steps",
        "object_lifted",
        "object_inside",
        "object_on_top",
        "ee_at_target",
    }
    return any(key in success for key in predicate_keys)


def _has_episode_evidence(episodes: list[EpisodeResult]) -> bool:
    return bool(episodes) and all(episode.steps for episode in episodes)


def _has_replay_or_episode_evidence(episodes: list[EpisodeResult]) -> bool:
    return _has_episode_evidence(episodes) or any(bool(episode.replay_path) for episode in episodes)


def _has_diagnosed_failures(episodes: list[EpisodeResult]) -> bool:
    failures = [episode for episode in episodes if not episode.success]
    return all(
        bool(episode.failure_label)
        and episode.failure_label != "unknown_failure"
        and episode.failure_label_source in {"env", "mapper"}
        for episode in failures
    )


def _warnings(tasks: tuple[TaskSpec, ...], engine_name: str) -> list[str]:
    warnings: list[str] = []
    for task in tasks:
        if task.engine != engine_name:
            warnings.append(f"task {task.task_id} declares engine {task.engine}, run used {engine_name}")
    return warnings
