# Task Spec

Task YAML files define the benchmark contract: robot, scene, objects, success criteria, randomization, metrics, and failure labels.

```yaml
task_id: warehouse_pick_place_v0
engine: maniskill
robot: panda
scene: warehouse_table
description: Pick the target object and place it into the green bin.
objects:
  - name: target_box
    type: cube
    randomized_pose: true
success:
  object_inside: green_bin
  max_time_seconds: 60
  max_collisions: 2
randomization:
  lighting: true
  object_pose: true
metrics:
  - success_rate
  - completion_time
failure_labels:
  - bad_grasp
  - object_slip
```
