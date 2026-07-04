# Learned Baselines

NyssaBench currently includes a repo-local linear behavior cloning baseline for
the focused ManiSkill result pack.

## Train BC

Generate or import demonstrations first. The repo-local `scripted_oracle` is a
lightweight heuristic and should not be used as a strong demo source unless it
clearly solves the target suite. For stronger ManiSkill demos, generate official
motion-planning trajectories and import them:

```bash
uv run nyssa import-maniskill-demos \
  --input demos/maniskill_motionplanning \
  --out benchmark_results/maniskill_manipulation_v0_planner_demos
```

For smoke testing only, you can generate repo-local scripted demonstrations:

```bash
uv run nyssa experiment \
  --suite maniskill_manipulation_v0 \
  --engine maniskill \
  --policies scripted_oracle \
  --seeds 0 1 2 \
  --episodes 100 \
  --out benchmark_results/maniskill_manipulation_v0_demos
```

Then train the checkpoint:

```bash
uv run nyssa train-bc \
  benchmark_results/maniskill_manipulation_v0_demos/scripted_oracle/seed_0/episodes.json \
  benchmark_results/maniskill_manipulation_v0_demos/scripted_oracle/seed_1/episodes.json \
  benchmark_results/maniskill_manipulation_v0_demos/scripted_oracle/seed_2/episodes.json \
  --out checkpoints/bc_policy.json
```

## Evaluate BC

```bash
NYSSA_BC_CHECKPOINT=checkpoints/bc_policy.json \
uv run nyssa experiment \
  --suite maniskill_manipulation_v0 \
  --engine maniskill \
  --policies random scripted_oracle bc_policy \
  --seeds 0 1 2 \
  --episodes 100 \
  --out benchmark_results/maniskill_manipulation_v0
```

## Interpretation

The linear BC baseline is intentionally simple. It is useful for checking that
NyssaBench can train and evaluate a learned policy from run artifacts. It should
not be described as a strong learned robot policy unless it clearly improves on
random and scripted baselines in validated result artifacts.

## Stronger Oracle Baselines

ManiSkill ships motion-planning examples for Panda tasks, but they require
native planning dependencies such as `mplib` and Pinocchio. On Windows/Python
3.13 these dependencies may need a separate Linux or conda environment. Use a
planner-backed oracle for publishable upper-bound numbers when those dependencies
are available; otherwise label the repo-local `scripted_oracle` as a lightweight
heuristic baseline.

After generating ManiSkill motion-planning HDF5 files, convert them into Nyssa
episode artifacts:

```bash
uv run nyssa import-maniskill-demos \
  --input demos/maniskill_motionplanning \
  --out benchmark_results/maniskill_manipulation_v0_planner_demos
```

This writes `episodes.json`, `episodes.jsonl`, `manifest.json`, and per-task
episode files under the output directory.
