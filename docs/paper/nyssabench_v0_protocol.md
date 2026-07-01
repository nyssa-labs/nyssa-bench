# NyssaBench v0 Protocol Draft

## Abstract

NyssaBench is an evaluation and failure-analysis layer for embodied AI policies under realistic simulation variation. It is designed to run on top of simulators rather than replacing them.

## Benchmark Scope

The v0 protocol covers tabletop manipulation, warehouse manipulation, articulated objects, stress tests, ManiSkill smoke mappings, and MuJoCo control smoke mappings.

## Reporting Requirements

Every published result must include:

- suite ID and task IDs
- task spec revision
- engine and engine version
- policy adapter and checkpoint
- seed range
- number of episodes
- aggregate metrics
- per-task success rates and 95% confidence intervals
- failure taxonomy distribution
- replay or video availability
- run artifact archive

## Comparison Rule

Compare policies only when task specs, engine, randomization settings, and seed protocol match.

## Limitations

The v0 prototype reliability score is not a calibrated sim-to-real predictor. Treat it as a simulator-readiness proxy until validated against real robot outcomes.

The protocol excludes synthetic backends. Published scorecards require real simulator adapters, explicit task mappings, reproducibility metadata, episode evidence, and passing public-claim validation.
