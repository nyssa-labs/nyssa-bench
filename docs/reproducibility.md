# Reproducibility

Every run directory records:

- `run.yaml`
- `config.yaml`
- `environment.json`
- `package_versions.json`
- `git_info.json`
- `metrics.json`
- `episodes.json`
- `report.html`

Use fixed seeds for comparisons and compare policies only when suite, task specs, engine, and randomization settings match.
