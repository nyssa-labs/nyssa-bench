# Policy Comparison Reports

Use `nyssa compare` to rank multiple run directories by success rate and sim-to-real score.

```bash
nyssa compare runs/random_warehouse runs/scripted_warehouse --out reports/warehouse_compare.html
```

Use `nyssa leaderboard` to write a JSON ranking that can be published by a static site or uploaded to a future hosted leaderboard.

```bash
nyssa leaderboard runs/random_warehouse runs/scripted_warehouse --out reports/leaderboard.json
```
