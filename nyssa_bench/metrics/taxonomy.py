FAILURE_LABELS = [
    "bad_grasp",
    "object_slip",
    "collision",
    "missed_target",
    "wrong_object",
    "occlusion_failure",
    "planner_stuck",
    "joint_limit_failure",
    "timeout",
    "latency_failure",
    "unstable_contact",
    "unknown_failure",
    "out_of_distribution_layout",
]

DEFAULT_METRICS = [
    "success_rate",
    "completion_time",
    "collision_count",
    "path_efficiency",
    "grasp_success_rate",
    "drop_rate",
    "object_slip_rate",
    "wrong_object_rate",
    "recovery_success_rate",
    "safety_violation_rate",
    "out_of_distribution_failure_rate",
]
