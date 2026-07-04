# Pairwise Evaluation Protocol

Pairwise evaluation is a planned NyssaBench mode inspired by real-world
preference-style robot-policy evaluation. It is not the v0.1 priority, but the
protocol should shape future APIs.

## Goal

Evaluate two policies under matched conditions:

- same suite
- same tasks
- same seeds
- same stressors
- same episode budget
- same success predicates
- same replay evidence requirements

The question is not only "which policy has higher success rate?" but:

- Which policy wins on the same initial conditions?
- Which failures are unique to one policy?
- Which stressors flip the winner?
- Are differences statistically meaningful?

## Planned Command Shape

```bash
nyssa arena-run \
  --suite maniskill_manipulation_v0 \
  --policy-a bc_policy \
  --policy-b scripted_oracle \
  --episodes 100 \
  --seeds 0 1 2 \
  --blind \
  --out arena_results/bc_vs_scripted
```

## Required Artifacts

- paired manifest
- per-policy run artifacts
- per-seed paired outcomes
- replay videos for both policies
- failure-delta table
- preference or win-rate summary
- significance note

## Blinding

When `--blind` is used, generated reports should hide policy names behind
stable labels such as `policy_a` and `policy_b` until the comparison is frozen.
This does not make simulation evaluation identical to real-world blinded
human-evaluator studies, but it reduces report bias and prepares the data model
for later real-world evaluation.

