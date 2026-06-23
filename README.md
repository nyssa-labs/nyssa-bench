# NyssaBench

**NyssaBench: open-source evaluation infrastructure for embodied AI policies under real-world variation.**

Run reproducible robot-policy benchmarks, stress-test failures, and export simulation data for training.

NyssaBench is not a simulator. It is an evaluation and failure-analysis layer that sits on top of robotics simulators such as ManiSkill and MuJoCo, with experimental adapter boundaries for RoboCasa and Genesis. The v0.1 scaffold ships with a deterministic dummy engine so the full benchmark loop can run in minutes without a heavy simulator install.

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

Experimental adapter boundaries for RoboCasa and Genesis are included in the repo. Their integration contracts live in `configs/experiments/`, and the adapters fail with explicit messages until those backends are wired to real task mappings. Use `.[full]` only when you intentionally want the heavy experimental/source dependencies.

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

## What Is Included In v0.1

- Core benchmark API: load suite, load policy, run episodes, collect metrics, save replay placeholders, export datasets, generate reports.
- Engine adapters: dummy/local, ManiSkill boundary, MuJoCo boundary, and experimental RoboCasa/Genesis boundaries.
- Task YAML DSL for tabletop, warehouse, articulated-object, and stress-test suites.
- Policy adapters: random, scripted, LeRobot, OpenVLA, robomimic, and diffusion policy interfaces.
- Failure taxonomy and aggregate metrics.
- HTML reports and JSON metrics.
- Policy comparison reports, sim-to-real readiness scores, and local leaderboard export.
- CLI, docs, examples, and tests.

## Positioning

NyssaBench helps robotics teams evaluate embodied AI policies under realistic variation before deploying them to real robots. It focuses on policy-agnostic evaluation, failure analysis, stress testing, replay-first reports, and dataset export rather than owning a physics engine.
