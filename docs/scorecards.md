# Scorecards

Public scorecards should be generated from committed task specs, fixed seeds, and published run artifacts.

`benchmark_results/baselines_v0.json` is generated from real simulator run directories. Do not hand-edit score values.

Generate public baseline artifacts from real simulator suites:

```bash
nyssa run --suite maniskill_smoke_v0 --engine maniskill --policy random --episodes 100 --seed 42 --out runs/maniskill_random_v0
nyssa run --suite mujoco_control_v0 --engine mujoco --policy random --episodes 100 --seed 42 --out runs/mujoco_random_v0
nyssa scorecard \
  runs/maniskill_random_v0 \
  runs/mujoco_random_v0 \
  --out benchmark_results/baselines_v0.json \
  --comparison-out reports/real_baselines_v0.html \
  --leaderboard-out site/leaderboard/leaderboard.json
```

Every scorecard should publish `metrics.json`, `episodes.jsonl`, `report.html`, environment metadata, package versions, git info, per-task success rates, confidence intervals, and failure counts.

For a learned baseline, run a real checkpoint through the adapter factory and include that run directory in the same command:

```bash
NYSSA_ROBOMIMIC_POLICY=my_project.policies:create_robomimic_policy \
nyssa run --suite maniskill_smoke_v0 --engine maniskill --policy robomimic --episodes 100 --seed 42 --out runs/maniskill_robomimic_v0 --no-replay

nyssa scorecard \
  runs/maniskill_random_v0 \
  runs/mujoco_random_v0 \
  runs/maniskill_robomimic_v0 \
  --out benchmark_results/baselines_v0.json \
  --comparison-out reports/real_baselines_v0.html \
  --leaderboard-out site/leaderboard/leaderboard.json
```
