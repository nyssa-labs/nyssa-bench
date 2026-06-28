# Adding New Tasks

1. Add a YAML file under `nyssa_bench/tasks/<domain>/<task_id>.yaml`.
2. Include the task ID in a suite under `configs/suites`.
3. Add simulator-specific metadata under `success`.
4. Run the suite with the target real backend.
