from nyssa_bench import PolicyRunner, Suite


suite = Suite.load("tabletop_manipulation_v0")
runner = PolicyRunner(policy="scripted", engine="dummy", episodes=3, seed=7, out="runs/scripted_tabletop")
report = runner.evaluate(suite)
print(report.summary)
