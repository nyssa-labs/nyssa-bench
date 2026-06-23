from nyssa_bench.core.task import TaskSpec


task = TaskSpec.load("pick_cube")
print(task.to_dict())
