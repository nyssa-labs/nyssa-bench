# Scorecards

Public scorecards should be generated from committed task specs, fixed seeds, and published run artifacts.

The seed result file in `benchmark_results/baselines_v0.json` is a structure example, not a claim of real simulator performance. Replace it with generated metrics from:

```bash
nyssa run --suite tabletop_manipulation_v0 --engine dummy --policy scripted --episodes 10 --seed 42 --out runs/tabletop_scripted
nyssa run --suite tabletop_manipulation_v0 --engine dummy --policy random --episodes 10 --seed 42 --out runs/tabletop_random
nyssa leaderboard runs/tabletop_scripted runs/tabletop_random --out site/leaderboard/leaderboard.json
```
