from nyssa_bench import PolicyRunner, Suite


suite = Suite.load("stress_tests_v0")
runner = PolicyRunner(policy="random", engine="dummy", episodes=2, seed=5)
report = runner.evaluate(suite)
report.save("runs/stress_report.html")
