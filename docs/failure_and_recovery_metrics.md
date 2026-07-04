# Failure and Recovery Metrics

NyssaBench should explain how policies fail, whether they recover, and which
failures are sensitive to stressors.

## Core Failure Fields

Each episode should record:

- `success`
- `failure_label`
- `failure_label_source`
- `steps`
- `replay_path`
- `failure_clip_path`
- task id
- seed
- stressor settings
- policy id and checkpoint metadata

## Recovery Fields

When available, tasks should add:

- `recovery_attempted`
- `recovery_success`
- `recovery_steps`
- `failure_step`
- `first_recovery_step`
- `post_recovery_success`

These fields should be optional because not every simulator or task exposes
enough state to classify recovery cleanly.

## Aggregate Metrics

Reports should include:

- failure mode distribution
- primary failure mode
- recovery success rate
- mean steps before failure
- mean steps to recovery
- drop rate
- collision rate
- timeout rate
- failure by stressor
- failure by seed
- failure by task

## Interpretation

Failure/recovery metrics should not hide low task success. A policy with low
success and high recovery attempts may still be unreliable. Reports should show
success, failure, and recovery together rather than replacing success rate with
a single composite score.

