# Learned Baselines

NyssaBench currently includes a repo-local linear behavior cloning baseline for
the focused ManiSkill result pack.

## Train BC

Generate scripted demonstrations first:

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
