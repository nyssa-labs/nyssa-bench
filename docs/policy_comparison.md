# Policy Comparison Reports

Use `nyssa compare` to rank multiple run directories by success rate and sim-to-real score.

```bash
nyssa compare runs/maniskill_random_v0 runs/mujoco_random_v0 --out reports/compare.html
```

Use `nyssa leaderboard` to write a JSON ranking that can be published by a static site or uploaded to a future hosted leaderboard.

```bash
nyssa leaderboard runs/maniskill_random_v0 runs/mujoco_random_v0 --out reports/leaderboard.json
```
