from nyssa_bench import PolicyRunner, Suite


suite = Suite.load("maniskill_smoke_v0")
runner = PolicyRunner(policy="random", engine="maniskill", episodes=3, seed=42, out="runs/maniskill_random")
report = runner.evaluate(suite)
print(report.summary)
