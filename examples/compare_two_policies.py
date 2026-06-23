from nyssa_bench import PolicyRunner, Suite
from nyssa_bench.reports.comparison import compare_runs, save_comparison_report


suite = Suite.load("tabletop_manipulation_v0")
PolicyRunner(policy="random", engine="dummy", episodes=2, seed=1, out="runs/random_tabletop").evaluate(suite)
PolicyRunner(policy="scripted", engine="dummy", episodes=2, seed=1, out="runs/scripted_tabletop").evaluate(suite)

comparison = compare_runs(["runs/random_tabletop", "runs/scripted_tabletop"])
save_comparison_report(comparison, "reports/tabletop_compare.html")
