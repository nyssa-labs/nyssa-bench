from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from nyssa_bench.core.episode import EpisodeResult, StepRecord
from nyssa_bench.core.registry import make_engine, make_policy
from nyssa_bench.core.suite import Suite
from nyssa_bench.core.task import REPO_ROOT
from nyssa_bench.datasets.export_json import export_json
from nyssa_bench.datasets.export_jsonl import export_jsonl
from nyssa_bench.datasets.export_metrics_csv import export_metrics_csv
from nyssa_bench.metrics.robustness import robustness_metrics
from nyssa_bench.metrics.safety import safety_metrics
from nyssa_bench.metrics.sim_to_real import score_summary
from nyssa_bench.metrics.success import aggregate_episodes
from nyssa_bench.policies.base import Policy, PolicyLike, load_policy_from_path
from nyssa_bench.replay.video import write_episode_video, write_failure_clip, write_replay_manifest
from nyssa_bench.replay.viewer import replay_viewer_placeholder
from nyssa_bench.reports.html_report import Report
from nyssa_bench.utils.reproducibility import (
    environment_metadata,
    git_info,
    make_run_id,
    package_versions,
    utc_now,
    write_json,
)


class PolicyRunner:
    """Evaluation harness for NyssaBench suites."""

    def __init__(
        self,
        policy: str | PolicyLike,
        engine: str = "dummy",
        episodes: int = 10,
        seed: int = 0,
        out: str | Path | None = None,
        max_steps: int | None = None,
    ) -> None:
        self.policy_ref = policy
        self.engine_name = engine
        self.episodes = episodes
        self.seed = seed
        self.out = Path(out) if out else None
        self.max_steps = max_steps
        self.episode_results: list[EpisodeResult] = []
        self.run_metadata: dict[str, Any] = {}

    def evaluate(self, suite: Suite) -> Report:
        policy = self._load_policy()
        engine = make_engine(self.engine_name)
        results: list[EpisodeResult] = []
        started_at = utc_now()

        try:
            for task in suite.tasks:
                engine.load_task(task)
                for episode_index in range(self.episodes):
                    episode_seed = self.seed + len(results)
                    if hasattr(policy, "reset"):
                        policy.reset(task=task, seed=episode_seed)
                    results.append(self._run_episode(engine, policy, task.task_id, episode_index, episode_seed))
        finally:
            engine.close()
            if hasattr(policy, "close"):
                policy.close()

        self.episode_results = results
        summary = aggregate_episodes(results)
        summary["sim_to_real_score"] = score_summary(summary)
        self.run_metadata = {
            "run_id": make_run_id(suite.suite_id, self._policy_name()),
            "suite_id": suite.suite_id,
            "task_ids": [task.task_id for task in suite.tasks],
            "policy_name": self._policy_name(),
            "engine_name": self.engine_name,
            "episodes_per_task": self.episodes,
            "seed": self.seed,
            "started_at": started_at,
            "finished_at": utc_now(),
        }
        report = Report(
            suite_id=suite.suite_id,
            policy=self._policy_name(),
            engine=self.engine_name,
            summary=summary,
            run_dir=self.out,
        )
        if self.out:
            self._write_run_artifacts(suite, report)
        return report

    def _run_episode(self, engine: Any, policy: PolicyLike, task_id: str, episode_index: int, seed: int) -> EpisodeResult:
        observation, _ = engine.reset(seed=seed)
        steps: list[StepRecord] = []
        frames: list[Any] = []
        last_info: dict[str, Any] = {}
        step_limit = self.max_steps or getattr(engine, "max_steps", 1000)
        if self.out:
            frame = _safe_render(engine)
            if frame is not None:
                frames.append(frame)

        for _ in range(step_limit):
            action = policy.act(observation)
            next_observation, reward, terminated, truncated, info = engine.step(action)
            if self.out:
                frame = _safe_render(engine)
                if frame is not None:
                    frames.append(frame)
            steps.append(
                StepRecord(
                    observation=observation,
                    action=action,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                )
            )
            observation = next_observation
            last_info = info
            if terminated or truncated:
                break

        success = bool(last_info.get("success", False))
        metrics = {
            "completion_time": float(last_info.get("completion_time", len(steps))),
            "path_efficiency": float(last_info.get("path_efficiency", 0.0)),
            "grasp_success_rate": 1.0 if bool(last_info.get("grasp_success", False)) else 0.0,
            "recovery_success_rate": 1.0 if success and len(steps) > 1 else 0.0,
            "drop_rate": 1.0 if last_info.get("failure_label") == "object_slip" else 0.0,
            **safety_metrics(last_info),
            **robustness_metrics(last_info),
        }
        episode = EpisodeResult(
            task_id=task_id,
            episode_index=episode_index,
            seed=seed,
            success=success,
            failure_label=last_info.get("failure_label"),
            metrics=metrics,
            steps=steps,
        )
        if self.out:
            episode.replay_path = write_episode_video(frames, self.out, task_id, episode_index)
        return episode

    def _load_policy(self) -> PolicyLike:
        if isinstance(self.policy_ref, Policy):
            return self.policy_ref
        path = Path(str(self.policy_ref))
        if path.suffix == ".py" or path.exists():
            return load_policy_from_path(path)
        return make_policy(str(self.policy_ref))

    def _policy_name(self) -> str:
        if isinstance(self.policy_ref, str):
            return self.policy_ref
        return self.policy_ref.__class__.__name__

    def _write_run_artifacts(self, suite: Suite, report: Report) -> None:
        assert self.out is not None
        self.out.mkdir(parents=True, exist_ok=True)
        config = {
            "run_id": self.run_metadata.get("run_id"),
            "suite": suite.to_dict(),
            "policy": self._policy_name(),
            "engine": self.engine_name,
            "episodes_per_task": self.episodes,
            "seed": self.seed,
        }
        with (self.out / "run.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.run_metadata, handle, sort_keys=False)
        with (self.out / "config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        write_json(self.out / "environment.json", environment_metadata())
        write_json(self.out / "package_versions.json", package_versions())
        write_json(self.out / "git_info.json", git_info(REPO_ROOT))
        (self.out / "plots").mkdir(exist_ok=True)
        for episode in self.episode_results:
            write_failure_clip(self.out, episode)
        with (self.out / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(report.summary, handle, indent=2)
        export_metrics_csv(report.summary, self.out / "metrics.csv")
        export_json(self.episode_results, self.out / "episodes.json")
        export_jsonl(self.episode_results, self.out / "episodes.jsonl")
        write_replay_manifest(self.episode_results, self.out)
        replay_viewer_placeholder(self.out)
        report.save(self.out / "report.html")


def _safe_render(engine: Any) -> Any:
    try:
        return engine.render()
    except Exception:
        return None
