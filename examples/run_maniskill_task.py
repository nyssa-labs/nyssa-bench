from nyssa_bench import PolicyRunner, Suite


suite = Suite.load("tabletop_manipulation_v0")
runner = PolicyRunner(policy="scripted", engine="maniskill", episodes=5, seed=42, out="runs/maniskill_tabletop")
runner.evaluate(suite)
