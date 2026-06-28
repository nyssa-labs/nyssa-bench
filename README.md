# NyssaBench

**NyssaBench: open-source evaluation infrastructure for embodied AI policies under real-world variation.**

Run reproducible robot-policy benchmarks, stress-test failures, and export simulation data for training.

NyssaBench is not a simulator. It is an evaluation and failure-analysis layer that sits on top of robotics simulators such as ManiSkill and MuJoCo, with experimental adapter boundaries for RoboCasa and Genesis. The v0.1 scaffold ships with a deterministic dummy engine so the full benchmark loop can run in minutes without a heavy simulator install. Dummy-engine runs are smoke tests only; public scorecards should use real simulator suites.

```python
from nyssa_bench import PolicyRunner, Suite

suite = Suite.load("warehouse_manipulation_v0")

runner = PolicyRunner(
    policy="random",
    engine="dummy",
    episodes=10,
    seed=42,
    out="runs/random_warehouse",
)

report = runner.evaluate(suite)
report.save("runs/random_warehouse/report.html")
```

## Install

```bash
pip install -e ".[dev]"
```

Optional runnable simulator integrations:

```bash
pip install -e ".[maniskill]"
pip install -e ".[mujoco]"
```

Optional policy/report/export stacks:

```bash
pip install -e ".[dataset,reports,video]"
pip install -e ".[lerobot,robomimic,vla,diffusion]"
```

Docker images are provided under `docker/` for core, ManiSkill, and MuJoCo smoke environments.

Experimental adapters for RoboCasa and Genesis are included in the repo. Their integration contracts live in `configs/experiments/`; contract validation works without heavyweight assets, and full simulator runs require concrete task-to-scene mappings plus upstream setup. Use `.[full]` only when you intentionally want the heavy experimental/source dependencies.

## First Run

```bash
nyssa list-suites

nyssa run \
  --suite warehouse_manipulation_v0 \
  --engine dummy \
  --policy random \
  --episodes 10 \
  --out runs/random_warehouse

nyssa report runs/random_warehouse
nyssa export --run runs/random_warehouse --format lerobot
nyssa export --run runs/random_warehouse --format jsonl
nyssa compare runs/random_warehouse runs/other_policy --out reports/compare.html
nyssa leaderboard runs/random_warehouse runs/other_policy --out reports/leaderboard.json
```

The run folder contains:

```txt
runs/random_warehouse/
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

Generate the local demo GIF:

```bash
python scripts/make_demo_gif.py
```

Validate optional simulator backends:

```bash
python scripts/validate_backend.py maniskill
python scripts/validate_backend.py mujoco
python scripts/validate_backend.py robocasa
python scripts/validate_backend.py genesis
```

`robocasa` and `genesis` validate their experiment contracts by default. They only run full simulator smoke tests after concrete task-to-scene mappings are added.

Run the release smoke test from a clean virtual environment:

```bash
python scripts/release_smoke.py
```

## What Is Included In v0.1

- Core benchmark API: load suite, load policy, run episodes, collect metrics, save replay videos, export datasets, generate reports.
- Engine adapters: dummy/local, ManiSkill, MuJoCo, and experimental RoboCasa/Genesis adapters with explicit validation contracts.
- Task YAML DSL for tabletop, warehouse, articulated-object, and stress-test suites.
- Policy adapters: random, scripted, LeRobot, OpenVLA, robomimic, and diffusion. External policy adapters accept `NYSSA_LEROBOT_POLICY`, `NYSSA_OPENVLA_POLICY`, `NYSSA_ROBOMIMIC_POLICY`, or `NYSSA_DIFFUSION_POLICY` as `module:factory` entry points.
- Failure taxonomy and aggregate metrics.
- HTML reports and JSON metrics.
- Policy comparison reports, sim-to-real readiness scores, and local leaderboard export.
- Static leaderboard shell, protocol draft, scorecard structure, Docker files, and plugin API.
- CLI, docs, examples, and tests.

## Positioning

NyssaBench helps robotics teams evaluate embodied AI policies under realistic variation before deploying them to real robots. It focuses on policy-agnostic evaluation, failure analysis, stress testing, replay-first reports, and dataset export rather than owning a physics engine.
