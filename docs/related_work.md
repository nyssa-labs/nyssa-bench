# Related Work

NyssaBench is not intended to replace robot simulators or task benchmarks. Its
role is to sit above them as a failure-aware evaluation, validation, reporting,
and comparison layer for embodied AI policies.

## Evaluation Systems

RoboArena introduces distributed real-world evaluation with double-blind
pairwise comparisons across evaluators and sites. Its core lesson for
NyssaBench is that future robot-policy evaluation should support pairwise,
blind, multi-site comparison rather than only single-policy success-rate tables.
Source: https://arxiv.org/abs/2506.18123

SIMPLER evaluates real-world manipulation policies in simulation and reports
paired sim-and-real evidence. Its lesson for NyssaBench is that simulator
results become stronger when they are tested for correlation with real robot
behavior, not merely reported as standalone numbers.
Source: https://arxiv.org/abs/2405.05941

PolaRiS reconstructs real scenes from short video scans into interactive
simulation environments for policy evaluation. It points toward a future
NyssaBench real-to-sim path where real robot logs or scene captures become
repeatable evaluation cases.
Source: https://arxiv.org/abs/2512.16881

RobotArena Infinity converts robot videos into simulated counterparts and uses
automated and human preference scoring. It supports NyssaBench's long-term
direction toward real-to-sim, perturbation-based, preference-compatible
evaluation.
Source: https://arxiv.org/abs/2510.23571

## Benchmark Audit Work

What Are We Actually Benchmarking in Robot Manipulation? identifies shortcut
solvability, weak statistics, creeping overfitting, and data-source dependence
as ways benchmark scores can stop measuring real capability. NyssaBench should
turn these concerns into validation checks, audit notes, and scorecard caveats.
Source: https://arxiv.org/abs/2606.04233

LIBERO-PRO shows that high standard benchmark performance can collapse under
reasonable perturbations. This supports NyssaBench's emphasis on stressors,
generalization checks, and anti-overclaim result validation.
Source: https://arxiv.org/abs/2510.03827

ROBOGATE studies adaptive failure discovery through large simulation sweeps and
boundary-focused sampling. This is closely aligned with NyssaBench's planned
failure-region discovery workflows.
Source: https://arxiv.org/abs/2603.22126

The SO-101 failure and recovery benchmark emphasizes failure taxonomy and
recovery-aware metrics on low-cost real robots. This supports NyssaBench's
decision to report failure labels, failure sources, and recovery metrics instead
of only binary success.
Source: https://arxiv.org/abs/2606.08881

## Simulator Benchmarks

ManiSkill, RoboCasa, RoboTwin, RoboVerse, RLBench, Meta-World, robosuite,
BEHAVIOR-1K, AI2-THOR, ProcTHOR, Habitat, iGibson, CALVIN, LIBERO, and VIMA are
best treated as backends, datasets, or task ecosystems. NyssaBench should not
compete with them as a simulator. It should provide:

- result validation
- replay-first evidence requirements
- failure taxonomy
- stress-condition reporting
- policy comparison
- dataset export from evaluated episodes
- result tiers and publication caveats

## Policy and Dataset Ecosystem

Open X-Embodiment, DROID, BridgeData V2, LeRobot, OpenVLA, Octo, SmolVLA, pi0,
Diffusion Policy, and MimicGen define the policy/data landscape NyssaBench
should evaluate against later. The immediate baseline order should stay simple:

1. random policy as a weak sanity check
2. scripted or planner-backed policy as a task sanity check
3. behavior cloning, robomimic, or diffusion policy as the first learned result
4. VLA/generalist policies only after the harness has credible video-backed
   result packs

## Positioning

NyssaBench's lane is:

> Open-source infrastructure for failure-aware, stress-test-based evaluation of
> embodied AI policies.

The project should complement simulator benchmarks such as ManiSkill and
RoboCasa, real-world evaluation systems such as RoboArena, and real-to-sim
frameworks such as SIMPLER, PolaRiS, and RobotArena Infinity.

