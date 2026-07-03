# NyssaBench Validation

A NyssaBench run can be described as a public benchmark result only when its
`public_claim_validation.status` is `validated`.

## Required Checks

- Real simulator backend: the run uses a supported real simulator adapter.
- Explicit task mapping: every task declares `success.engine_env_ids.<engine>` or an engine factory.
- Known success predicate: every task maps simulator success information or a task-specific success metric.
- Episode count: at least 100 episodes per task.
- Seed coverage: public result packs should include at least 3 seeds per policy.
- Evidence: every run includes episode artifacts. Replay videos are required for a replay-first public report, and their absence must be called out in `RESULTS.md`.
- Failure labels: failed episodes use environment or mapper labels, not silent placeholder defaults.
- Stressors: unsupported stressors are reported and not claimed as active perturbations.
- Reproducibility metadata: package versions, environment metadata, run config, and git metadata are present.

## Current Gate

The code-level gate is `nyssa_bench.metrics.run_claims.RunClaimValidator`.
It validates individual runs. Result-pack seed coverage is documented in the
generated `RESULTS.md` and `manifest.json`.

## What Not To Claim

- `prototype_reliability_score` is a simulator reliability heuristic, not a
  real-world sim-to-real score.
- A run without replay videos is still auditable from episode artifacts, but it
  should not be described as video-backed failure analysis.
- Adapter hooks for OpenVLA, diffusion, LeRobot, and RoboMimic are not public
  baseline results unless a concrete model/checkpoint and validated run artifacts
  are included.
- Genesis and RoboCasa are experimental contract adapters until concrete task
  mappings and upstream assets are configured.
