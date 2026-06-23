from nyssa_bench import PolicyRunner, Suite


suite = Suite.load("warehouse_manipulation_v0")
runner = PolicyRunner(policy="random", engine="dummy", episodes=3, seed=42, out="runs/random_warehouse")
report = runner.evaluate(suite)
print(report.summary)
