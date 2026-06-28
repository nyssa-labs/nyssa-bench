from nyssa_bench import PolicyRunner, Suite
from nyssa_bench.datasets.export_lerobot import export_lerobot


suite = Suite.load("tabletop_manipulation_v0")
runner = PolicyRunner(policy="random", engine="maniskill", episodes=2, seed=1)
runner.evaluate(suite)
export_lerobot(runner.episode_results, "runs/maniskill_tabletop/lerobot")
