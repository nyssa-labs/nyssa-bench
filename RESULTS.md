# NyssaBench Results

NyssaBench can generate a ManiSkill result pack, but the current local Windows
numbers were weak. Treat local run outputs as reproducibility artifacts, not as
strong robotics benchmark claims.

Latest local candidate, not committed because the run artifacts are large:

- `benchmark_results/maniskill_manipulation_v0_state_obs`
- Suite: `maniskill_manipulation_v0`
- Engine: ManiSkill
- Tasks: `PickCube-v1`, `StackCube-v1`, `PushCube-v1`
- Policies: `random`, `scripted_oracle`, `bc_policy`
- Seeds: `0, 1, 2`
- Episodes: 100 per task per seed, 900 per policy
- Validation: not public under the current gate because MP4 replay videos are absent
- Replay videos: absent in this Windows/Sapien run; episode artifacts are present for debugging only

Current headline:

| Policy | Successes | Episodes | Success rate | Primary failure |
| --- | ---: | ---: | ---: | --- |
| random | 0 | 900 | 0.0000 | timeout |
| scripted_oracle | 6 | 900 | 0.0067 | timeout |
| bc_policy | 6 | 900 | 0.0067 | timeout |

Interpretation:

- `random` is a weak sanity check.
- `scripted_oracle` is currently a lightweight heuristic baseline, not a strong oracle.
- `bc_policy` is a learned baseline pipeline check, not a strong learned policy.
- The result demonstrates reproducible real-simulator execution and basic failure reporting.
- The result does not yet demonstrate strong manipulation performance.
- The result is not a public NyssaBench benchmark result because it is not video-backed.
- A stronger v0.1 should use a planner-backed oracle or a trained checkpoint that clearly beats random and the current heuristic.

## Target v0.1 Result Pack

```txt
benchmark_results/maniskill_manipulation_v0/
|-- random/
|-- scripted_oracle/
|-- bc_policy/
|-- comparison.html
|-- leaderboard.json
|-- scorecard.json
|-- manifest.json
`-- RESULTS.md
```

## Required Baselines

- `random`: weak sanity check.
- `scripted_oracle`: repo-local ManiSkill scripted heuristic by default, or custom controller from `NYSSA_SCRIPTED_ORACLE_POLICY`.
- `bc_policy`: repo-local linear BC checkpoint loaded from `NYSSA_BC_CHECKPOINT`, or custom model from `NYSSA_BC_POLICY`.

## Run Commands

```bash
uv run nyssa experiment \
  --suite maniskill_manipulation_v0 \
  --engine maniskill \
  --policies scripted_oracle \
  --seeds 0 1 2 \
  --episodes 100 \
  --out benchmark_results/maniskill_manipulation_v0_demos

uv run nyssa train-bc \
  benchmark_results/maniskill_manipulation_v0_demos/scripted_oracle/seed_0/episodes.json \
  benchmark_results/maniskill_manipulation_v0_demos/scripted_oracle/seed_1/episodes.json \
  benchmark_results/maniskill_manipulation_v0_demos/scripted_oracle/seed_2/episodes.json \
  --out checkpoints/bc_policy.json

uv run nyssa export \
  --run benchmark_results/maniskill_manipulation_v0_demos/scripted_oracle/seed_0 \
  --format robomimic \
  --out benchmark_results/maniskill_manipulation_v0_demos/scripted_oracle/seed_0/robomimic.hdf5

uv run nyssa train-robomimic \
  --config configs/policies/robomimic_bc_flat.json

NYSSA_BC_CHECKPOINT=checkpoints/bc_policy.json \
uv run nyssa experiment \
  --suite maniskill_manipulation_v0 \
  --engine maniskill \
  --policies random scripted_oracle bc_policy \
  --seeds 0 1 2 \
  --episodes 100 \
  --out benchmark_results/maniskill_manipulation_v0
```

Each run's `metrics.json` contains `public_claim_validation`. Only publish results whose validation passes. Under the current gate, that requires MP4 replay videos for every episode.

## Next Improvement Target

Install/use ManiSkill motion-planning dependencies in a suitable Linux or conda
environment, generate planner-backed demonstrations, retrain BC, and rerun the
same matrix with replay capture enabled.

Use `NYSSA_ROBOMIMIC_CHECKPOINT=path/to/model.pth --policies robomimic` to evaluate a robomimic checkpoint, or `NYSSA_LEROBOT_POLICY_PATH=path/to/policy --policies lerobot` to evaluate a LeRobot policy directory.
