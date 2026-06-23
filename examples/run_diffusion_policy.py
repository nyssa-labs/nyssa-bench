from nyssa_bench import PolicyRunner, Suite
from nyssa_bench.policies.diffusion_policy_adapter import DiffusionPolicyAdapter


def load_diffusion_policy(path: str):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install with pip install -e '.[diffusion]'") from exc
    return torch.load(path, map_location="cpu")


model = load_diffusion_policy("checkpoints/diffusion_policy.pt")
suite = Suite.load("tabletop_manipulation_v0")
runner = PolicyRunner(policy=DiffusionPolicyAdapter(model), engine="dummy", episodes=5, out="runs/diffusion_example")
runner.evaluate(suite)
