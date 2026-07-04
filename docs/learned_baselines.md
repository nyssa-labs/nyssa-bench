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

## RoboMimic BC

RoboMimic is the next learned baseline after the linear smoke test. Export the
imported planner demonstrations to RoboMimic HDF5:

```bash
uv run nyssa export \
  --run benchmark_results/maniskill_manipulation_v0_planner_state_demos \
  --format robomimic \
  --out benchmark_results/maniskill_manipulation_v0_planner_state_demos/robomimic_flat.hdf5

uv run nyssa write-robomimic-config \
  --data benchmark_results/maniskill_manipulation_v0_planner_state_demos/robomimic_flat.hdf5 \
  --out configs/generated/robomimic_planner_bc.json \
  --output-dir checkpoints/robomimic_planner \
  --name nyssa_maniskill_planner_bc \
  --epochs 50 \
  --batch-size 64

uv run nyssa train-robomimic \
  --config configs/generated/robomimic_planner_bc.json
```

After training, point `NYSSA_ROBOMIMIC_CHECKPOINT` at the best saved
`model_epoch_*.pth` file and evaluate:

```bash
NYSSA_ROBOMIMIC_CHECKPOINT=checkpoints/robomimic_planner/nyssa_maniskill_planner_bc/<run>/models/model_epoch_50.pth \
uv run nyssa run \
  --suite maniskill_planner_bc_v0 \
  --engine maniskill \
  --policy robomimic \
  --episodes 30 \
  --seed 0 \
  --out runs/robomimic_planner_smoke \
  --capture-replay
```

For stronger per-task baselines, train one RoboMimic checkpoint per imported
task file and place/copy the best checkpoints here:

```txt
checkpoints/robomimic_by_task/maniskill_pick_cube.pth
checkpoints/robomimic_by_task/maniskill_stack_cube.pth
checkpoints/robomimic_by_task/maniskill_push_cube.pth
```

Then evaluate:

```bash
NYSSA_TASK_ROBOMIMIC_DIR=checkpoints/robomimic_by_task \
uv run nyssa run \
  --suite maniskill_planner_bc_v0 \
  --engine maniskill \
  --policy task_robomimic \
  --episodes 30 \
  --seed 0 \
  --out runs/task_robomimic_planner_smoke \
  --capture-replay
```

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

When evaluating BC trained from official ManiSkill Panda motion-planning demos,
use `maniskill_planner_bc_v0`. The official demo generator records actions in
`pd_joint_pos`; this suite uses the same control mode. Do not compare those
checkpoints against `maniskill_manipulation_v0`, which uses end-effector delta
control for the repo-local heuristic baseline.

For the repo-local linear BC baseline, train one checkpoint per task and use the
task-routed policy:

```bash
mkdir -p checkpoints/bc_by_task

uv run nyssa train-bc \
  benchmark_results/maniskill_manipulation_v0_planner_state_demos/maniskill_pick_cube/episodes.json \
  --out checkpoints/bc_by_task/maniskill_pick_cube.json

uv run nyssa train-bc \
  benchmark_results/maniskill_manipulation_v0_planner_state_demos/maniskill_stack_cube/episodes.json \
  --out checkpoints/bc_by_task/maniskill_stack_cube.json

uv run nyssa train-bc \
  benchmark_results/maniskill_manipulation_v0_planner_state_demos/maniskill_push_cube/episodes.json \
  --out checkpoints/bc_by_task/maniskill_push_cube.json

NYSSA_TASK_BC_DIR=checkpoints/bc_by_task \
uv run nyssa run \
  --suite maniskill_planner_bc_v0 \
  --engine maniskill \
  --policy task_bc_policy \
  --episodes 10 \
  --seed 0 \
  --out runs/task_bc_planner_smoke \
  --capture-replay
```
