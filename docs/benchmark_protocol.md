# Benchmark Protocol

Report the suite ID, task IDs, policy adapter, engine adapter, simulator version, seed range, number of episodes, task YAML revision, aggregate metrics, per-task metrics, 95% confidence intervals, and failure counts. Do not compare policies across different task specs, engines, success criteria, or randomization ranges unless the report states the difference explicitly.

Only runs that pass `public_claim_validation` should be published as benchmark claims. A public claim requires a supported real simulator adapter, explicit task-to-environment mappings, mapped success predicates, enough episodes, replay or episode evidence, diagnosed failure labels, package versions, and git metadata.
