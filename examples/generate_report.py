from nyssa_bench import PolicyRunner, Suite


suite = Suite.load("maniskill_smoke_v0")
runner = PolicyRunner(policy="random", engine="maniskill", episodes=2, seed=5)
report = runner.evaluate(suite)
report.save("runs/maniskill_report.html")
