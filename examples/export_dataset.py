from nyssa_bench import PolicyRunner, Suite
from nyssa_bench.datasets.export_lerobot import export_lerobot


suite = Suite.load("tabletop_manipulation_v0")
runner = PolicyRunner(policy="scripted", engine="dummy", episodes=2, seed=1)
runner.evaluate(suite)
export_lerobot(runner.episode_results, "runs/scripted_tabletop/lerobot")
