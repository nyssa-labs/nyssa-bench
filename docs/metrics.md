# Metrics

NyssaBench reports more than success rate. v0.1 includes:

- success_rate
- completion_time
- collision_count
- path_efficiency
- grasp_success_rate
- drop_rate
- object_slip_rate
- wrong_object_rate
- recovery_success_rate
- safety_violation_rate
- out_of_distribution_failure_rate
- prototype_reliability_score

`prototype_reliability_score` is a heuristic over simulator success, safety, and robustness. It is not a calibrated sim-to-real score.

Failure labels include bad grasp, object slip, collision, missed target, wrong object, occlusion failure, planner stuck, joint limit failure, timeout, latency failure, unstable contact, unknown failure, and out-of-distribution layout.
