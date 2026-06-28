from nyssa_bench import PolicyRunner, Suite
from nyssa_bench.policies.robomimic_adapter import RoboMimicPolicy


def load_robomimic_checkpoint(path: str):
    try:
        from robomimic.utils.file_utils import policy_from_checkpoint
    except ImportError as exc:
        raise RuntimeError("Install with pip install -e '.[robomimic]'") from exc
    policy, _ = policy_from_checkpoint(ckpt_path=path)
    return policy


model = load_robomimic_checkpoint("checkpoints/robomimic_policy.pth")
suite = Suite.load("maniskill_smoke_v0")
runner = PolicyRunner(policy=RoboMimicPolicy(model), engine="maniskill", episodes=5, out="runs/robomimic_example")
runner.evaluate(suite)
