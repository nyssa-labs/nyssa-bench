# NyssaBench

**NyssaBench: open-source evaluation infrastructure for embodied AI policies under real-world variation.**

Run reproducible robot-policy benchmarks, stress-test failures, and export simulation data for training.

NyssaBench is not a simulator. It is an evaluation and failure-analysis layer that sits on top of robotics simulators such as ManiSkill and MuJoCo, with experimental adapter boundaries for RoboCasa and Genesis. Public scorecards must pass the run claim validator; experimental adapters and placeholder task mappings are not public benchmark claims.

```python
from nyssa_bench import PolicyRunner, Suite

suite = Suite.load("mujoco_control_v0")

runner = PolicyRunner(
    policy="random",
    engine="mujoco",
    episodes=10,
    seed=42,
    out="runs/random_mujoco",
)

report = runner.evaluate(suite)
report.save("runs/random_mujoco/report.html")
```

## Install

```bash
uv sync --extra dev --extra mujoco --extra video --extra reports
```

Optional runnable simulator integrations:

```bash
uv sync --extra dev --extra maniskill --extra video --extra reports
uv sync --extra dev --extra mujoco --extra video --extra reports
```

Optional policy/report/export stacks:

```bash
uv sync --extra dev --extra dataset --extra reports --extra video
uv sync --extra dev --extra lerobot --extra robomimic --extra vla --extra diffusion
```

Plain `pip install -e ".[dev,mujoco]"` still works if you are not using `uv`.

Docker images are provided under `docker/` for core, ManiSkill, and MuJoCo environments.

Experimental adapters for RoboCasa and Genesis are included in the repo. Their integration contracts live in `configs/experiments/`; contract validation works without heavyweight assets, and full simulator runs require concrete task-to-scene mappings plus upstream setup. Use `.[full]` only when you intentionally want the heavy experimental/source dependencies.

## First Run

```bash
uv run nyssa list-suites

uv run nyssa run \
  --suite mujoco_control_v0 \
  --engine mujoco \
  --policy random \
  --episodes 10 \
  --out runs/random_mujoco

uv run nyssa report runs/random_mujoco
uv run nyssa export --run runs/random_mujoco --format lerobot
uv run nyssa export --run runs/random_mujoco --format jsonl
uv run nyssa compare runs/random_mujoco runs/other_policy --out reports/compare.html
uv run nyssa leaderboard runs/random_mujoco runs/other_policy --out reports/leaderboard.json
uv run nyssa scorecard runs/random_mujoco runs/other_policy --out benchmark_results/baselines_v0.json
```

Run the focused baseline matrix after collecting scripted demos and training the repo-local BC checkpoint:

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

NYSSA_BC_CHECKPOINT=checkpoints/bc_policy.json \
uv run nyssa experiment \
  --suite maniskill_manipulation_v0 \
  --engine maniskill \
  --policies random scripted_oracle bc_policy \
  --seeds 0 1 2 \
  --episodes 100 \
  --out benchmark_results/maniskill_manipulation_v0
```

The run folder contains:

```txt
runs/random_mujoco/
|-- config.yaml
|-- run.yaml
|-- environment.json
|-- package_versions.json
|-- git_info.json
|-- metrics.json
|-- metrics.csv
|-- episodes.json
|-- episodes.jsonl
|-- replay_manifest.json
|-- replay.html
|-- videos/
|-- failures/
|-- plots/
|-- lerobot/
`-- report.html
```

Validate optional simulator backends:

```bash
uv run python scripts/validate_backend.py maniskill
uv run python scripts/validate_backend.py mujoco
uv run python scripts/validate_backend.py robocasa
uv run python scripts/validate_backend.py genesis
```

`robocasa` and `genesis` validate their experiment contracts by default. They only run full simulator checks after concrete task-to-scene mappings are added.

Run the release check from a clean virtual environment:

```bash
uv run python scripts/release_smoke.py
```

## What Is Included In v0.1

- Core benchmark API: load suite, load policy, run episodes, collect metrics, save replay videos, export datasets, generate reports.
- Engine adapters: ManiSkill, MuJoCo, and experimental RoboCasa/Genesis adapters with explicit validation contracts.
- Task YAML DSL for tabletop, warehouse, articulated-object, and stress-test suites.
- Policy adapters: random, repo-local scripted heuristic, repo-local linear BC, robomimic checkpoint loading, LeRobot checkpoint loading, plus hook-only OpenVLA and diffusion adapters. External policy adapters accept `NYSSA_SCRIPTED_ORACLE_POLICY`, `NYSSA_BC_POLICY`, `NYSSA_LEROBOT_POLICY`, `NYSSA_OPENVLA_POLICY`, `NYSSA_ROBOMIMIC_POLICY`, or `NYSSA_DIFFUSION_POLICY` as `module:factory` entry points.
- Baseline experiment command for policy/seed matrices and result-pack generation.
- Failure taxonomy, mapper-based failure labels, and aggregate metrics.
- HTML reports, JSON metrics, public-claim validation, and unsupported-stressor reporting.
- Policy comparison reports, prototype reliability scores, and leaderboard export.
- Static leaderboard shell, protocol draft, scorecard structure, Docker files, and plugin API.
- CLI, docs, examples, and tests.

## Positioning

NyssaBench helps robotics teams evaluate embodied AI policies under realistic variation before deploying them to real robots. It focuses on policy-agnostic evaluation, failure analysis, stress testing, replay-first reports, and dataset export rather than owning a physics engine.
