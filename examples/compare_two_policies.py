from nyssa_bench import PolicyRunner, Suite
from nyssa_bench.reports.comparison import compare_runs, save_comparison_report


suite = Suite.load("tabletop_manipulation_v0")
PolicyRunner(policy="random", engine="maniskill", episodes=2, seed=1, out="runs/maniskill_random_a").evaluate(suite)
PolicyRunner(policy="random", engine="maniskill", episodes=2, seed=2, out="runs/maniskill_random_b").evaluate(suite)

comparison = compare_runs(["runs/maniskill_random_a", "runs/maniskill_random_b"])
save_comparison_report(comparison, "reports/tabletop_compare.html")
