# v0.1 Launch Checklist

NyssaBench v0.1 is ready to launch when these checks pass from a clean checkout:

```bash
python scripts/release_smoke.py
python scripts/validate_backend.py maniskill
python scripts/validate_backend.py robocasa
python scripts/validate_backend.py genesis
python scripts/make_demo_gif.py
```

Release artifacts:

- `runs/release_smoke/report.html`
- `runs/maniskill_validation/report.html`
- `docs/assets/nyssabench-demo.gif`
- `benchmark_results/baselines_v0.json`
- `docs/paper/nyssabench_v0_protocol.md`

Tag and publish:

```bash
git tag v0.1.0
git push origin v0.1.0
```

Announcement draft:

```text
We are open-sourcing NyssaBench v0.1, an evaluation and failure-analysis framework for embodied AI policies under real-world variation.

It lets robotics teams run reproducible manipulation benchmarks, stress-test policies, generate replay-first HTML reports, classify failure modes, compare policy versions, and export trajectories for training.

The v0.1 release includes a lightweight dummy engine for first-run demos, ManiSkill and MuJoCo adapter paths, task YAML specs, policy adapter entry points, LeRobot-style export, local reports, and release smoke tests.

Repo: https://github.com/nyssa-labs/nyssa-bench
```
