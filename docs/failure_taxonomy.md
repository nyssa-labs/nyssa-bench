# Failure Taxonomy

NyssaBench reports failure modes instead of reducing evaluation to pass or fail.

Core labels:

- bad_grasp
- object_slip
- collision
- missed_target
- wrong_object
- occlusion_failure
- planner_stuck
- joint_limit_failure
- timeout
- latency_failure
- unstable_contact
- unknown_failure
- out_of_distribution_layout

Simulator adapters should map their native events into these labels first, then add engine-specific labels only when the shared taxonomy is insufficient. If no diagnostic event matches, use `unknown_failure`; do not default to the first configured failure label.
