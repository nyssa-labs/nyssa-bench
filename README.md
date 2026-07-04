# NyssaBench

**NyssaBench is open-source foundational infrastructure for failure-aware evaluation of frontier embodied AI policies.**

Run reproducible robot-policy benchmarks, stress-test failures, audit benchmark
claims, compare policies, and export simulation data for training.

NyssaBench is not a simulator. It is the measurement layer for frontier embodied
AI: a foundational benchmark and audit framework for evaluating how robot
policies fail under real-world variation. It sits on top of robotics simulators
such as ManiSkill and MuJoCo, with experimental adapter boundaries for RoboCasa
and Genesis. Public scorecards must pass the run claim validator; experimental
adapters and placeholder task mappings are not public benchmark claims.

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

Use Python 3.10 for ManiSkill/Linux runs. Some ManiSkill planning dependencies
publish wheels for CPython 3.10 but not newer Python ABIs, so Python 3.12 can
fail during dependency resolution.

```bash
uv python install 3.10
uv venv --python 3.10 .venv
source .venv/bin/activate
python --version
```

Then install the extras for the workflow you want to run. Use MuJoCo for the
quick smoke commands below:

```bash
uv sync --extra dev --extra mujoco --extra dataset --extra video --extra reports
```

Use ManiSkill for the focused manipulation result pack:

```bash
uv sync --extra dev --extra maniskill --extra dataset --extra video --extra reports
```

Optional policy/report/export stacks:

```bash
uv sync --extra dev --extra dataset --extra reports --extra video
uv sync --extra dev --extra lerobot --extra robomimic --extra vla --extra diffusion
```

Plain `python -m pip install -e ".[dev,mujoco,dataset,video,reports]"` still works if you are not using `uv`.

Simulator video capture also requires system rendering libraries. On
Ubuntu/Debian GPU machines, install and verify them before running public
benchmark commands:

```bash
bash scripts/setup_rendering_linux.sh
vulkaninfo --summary
nvidia-smi
```

If `vulkaninfo --summary` cannot see a Vulkan device, ManiSkill can still run
some CPU-side simulation paths but replay videos will not be produced. Public
NyssaBench benchmark claims require MP4 replay evidence for every episode.
If `vulkaninfo --summary` reports only `llvmpipe`, the machine is using CPU
Vulkan. Install the NVIDIA Vulkan ICD/GL packages that match the host driver,
for example `nvidia-utils-535` and `libnvidia-gl-535` on driver branch 535.

On macOS, MuJoCo smoke runs usually need the Python extras plus native GLFW:

```bash
brew install glfw
```

ManiSkill video-backed result packs are expected to run on Linux machines with
a working NVIDIA/Vulkan stack.

Do not use `pip install -e ".[full]"` for normal benchmark runs. The `full`
extra intentionally pulls heavy experimental stacks, including RoboCasa,
Genesis, RoboMimic, LeRobot, VLA, and diffusion dependencies, and native
packages in those stacks can fail on otherwise valid MuJoCo or ManiSkill
machines. Install only the extras for the workflow you are running.

Docker images are provided under `docker/` for core, ManiSkill, and MuJoCo environments.

Experimental adapters for RoboCasa and Genesis are included in the repo. Their integration contracts live in `configs/experiments/`; contract validation works without heavyweight assets, and full simulator runs require concrete task-to-scene mappings plus upstream setup. Use `.[full]` only when you intentionally want the heavy experimental/source dependencies.

## First Run

The MuJoCo commands require the MuJoCo install command above. They create two
real run directories before report, export, compare, leaderboard, and scorecard
commands are called.

```bash
uv run nyssa list-suites

uv run nyssa run \
  --suite mujoco_control_v0 \
  --engine mujoco \
  --policy random \
  --episodes 10 \
  --seed 0 \
  --out runs/random_mujoco_seed0

uv run nyssa run \
  --suite mujoco_control_v0 \
  --engine mujoco \
  --policy random \
  --episodes 10 \
  --seed 1 \
  --out runs/random_mujoco_seed1

uv run nyssa report runs/random_mujoco_seed0
uv run nyssa export --run runs/random_mujoco_seed0 --format lerobot
uv run nyssa export --run runs/random_mujoco_seed0 --format jsonl
uv run nyssa compare runs/random_mujoco_seed0 runs/random_mujoco_seed1 --out reports/compare.html
uv run nyssa leaderboard runs/random_mujoco_seed0 runs/random_mujoco_seed1 --out reports/leaderboard.json
uv run nyssa scorecard runs/random_mujoco_seed0 runs/random_mujoco_seed1 --out benchmark_results/baselines_v0.json
```

Run the focused ManiSkill baseline matrix after installing the ManiSkill extras,
collecting scripted demos, and training the repo-local BC checkpoint:

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
runs/random_mujoco_seed0/
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
