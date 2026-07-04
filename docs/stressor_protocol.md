# Stressor Protocol

NyssaBench stressors are controlled variations used to expose policy brittleness.
They must be declared, applied by the backend, and reported honestly.

## Core Stressor Dimensions

Recommended dimensions:

- object pose
- object identity
- distractor objects
- lighting
- camera viewpoint
- camera noise
- background texture
- language perturbation
- robot initial state
- friction
- mass
- latency
- sensor dropout
- joint fault
- actuator degradation

## Support States

Every stressor has one of three states:

| State | Meaning |
| --- | --- |
| `supported` | The adapter applies the stressor and reports its settings. |
| `unsupported` | The task declares the stressor, but the adapter cannot apply it. |
| `not_declared` | The task does not request this stressor. |

Unsupported stressors must appear in `metrics.json` and reports. They must not
be described as active perturbations in public results.

## Minimum Reporting

Each run should report:

- enabled stressors
- supported stressors
- unsupported stressors
- stressor values per episode when available
- success by stressor
- failure mode by stressor
- policy comparison under identical stressor distributions

## Public Claim Rule

A public stress-test claim requires:

- identical stressor distributions for compared policies
- MP4 replay evidence for every episode
- declared randomization ranges
- unsupported stressors called out explicitly
- confidence intervals for success and important failure rates

## v0.1 Scope

For the first credible NyssaBench result, `seed` is the only stressor that is
fully supported across the initial ManiSkill result pack. Do not claim lighting,
friction, camera, distractor, or object-identity stress tests until the adapter
actually applies and records them.

