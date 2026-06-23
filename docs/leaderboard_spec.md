# Leaderboard Spec

The local leaderboard export is a JSON array sorted by success rate and sim-to-real score.

Each entry contains:

- `rank`
- `run_dir`
- `success_rate`
- `sim_to_real_score`
- `primary_failure_mode`

This file can be published by a static site today or uploaded to a hosted Nyssa Cloud leaderboard later.
