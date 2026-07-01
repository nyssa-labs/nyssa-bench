# NyssaBench Results

No current committed artifact is a public benchmark claim.

The committed random baseline JSON and static leaderboard are sanity artifacts from earlier runs. Regenerate results with the current validator before publishing benchmark numbers.

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

Each run's `metrics.json` contains `public_claim_validation`. Only publish results whose validation passes.

Use `NYSSA_ROBOMIMIC_CHECKPOINT=path/to/model.pth --policies robomimic` to evaluate a robomimic checkpoint, or `NYSSA_LEROBOT_POLICY_PATH=path/to/policy --policies lerobot` to evaluate a LeRobot policy directory.
