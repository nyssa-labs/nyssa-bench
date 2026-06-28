# Scorecards

Public scorecards should be generated from committed task specs, fixed seeds, and published run artifacts.

`benchmark_results/baselines_v0.json` is a real-baseline manifest until simulator runs are generated. Dummy-engine runs are smoke tests only and must not be published as benchmark scorecards.

Generate public baseline artifacts from real simulator suites:

```bash
nyssa run --suite maniskill_smoke_v0 --engine maniskill --policy random --episodes 100 --seed 42 --out runs/maniskill_random_v0
nyssa run --suite mujoco_control_v0 --engine mujoco --policy random --episodes 100 --seed 42 --out runs/mujoco_random_v0
nyssa leaderboard runs/maniskill_random_v0 runs/mujoco_random_v0 --out site/leaderboard/leaderboard.json
```

Every scorecard should publish `metrics.json`, `episodes.jsonl`, `report.html`, environment metadata, package versions, git info, per-task success rates, confidence intervals, and failure counts.
