from nyssa_bench import PolicyRunner, Suite
from nyssa_bench.policies.lerobot_adapter import LeRobotPolicy


def load_lerobot_policy(path: str):
    try:
        from lerobot.common.policies.factory import make_policy
    except ImportError as exc:
        raise RuntimeError("Install with pip install -e '.[lerobot]'") from exc
    return make_policy(policy_path=path)


model = load_lerobot_policy("checkpoints/lerobot_policy")
suite = Suite.load("tabletop_manipulation_v0")
runner = PolicyRunner(policy=LeRobotPolicy(model), engine="dummy", episodes=5, out="runs/lerobot_example")
runner.evaluate(suite)
