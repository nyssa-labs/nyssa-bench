from nyssa_bench import PolicyRunner, Suite
from nyssa_bench.policies.openvla_adapter import OpenVLAPolicy


def load_openvla_model(model_id: str):
    try:
        from transformers import AutoModelForVision2Seq, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("Install with pip install -e '.[vla]' and follow OpenVLA upstream setup") from exc
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(model_id, trust_remote_code=True)
    return {"model": model, "processor": processor}


model = load_openvla_model("openvla/openvla-7b")
suite = Suite.load("tabletop_manipulation_v0")
runner = PolicyRunner(policy=OpenVLAPolicy(model), engine="dummy", episodes=5, out="runs/openvla_example")
runner.evaluate(suite)
