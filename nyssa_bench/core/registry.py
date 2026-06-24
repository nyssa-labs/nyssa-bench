from __future__ import annotations

from nyssa_bench.engines.base import NyssaEngine
from nyssa_bench.engines.dummy_adapter import DummyEngine
from nyssa_bench.engines.genesis_adapter import GenesisEngine
from nyssa_bench.engines.maniskill_adapter import ManiSkillEngine
from nyssa_bench.engines.mujoco_adapter import MuJoCoEngine
from nyssa_bench.engines.robocasa_adapter import RoboCasaEngine
from nyssa_bench.policies.base import Policy
from nyssa_bench.policies.diffusion_policy_adapter import DiffusionPolicyAdapter
from nyssa_bench.policies.lerobot_adapter import LeRobotPolicy
from nyssa_bench.policies.openvla_adapter import OpenVLAPolicy
from nyssa_bench.policies.random_policy import RandomPolicy
from nyssa_bench.policies.robomimic_adapter import RoboMimicPolicy
from nyssa_bench.policies.scripted_policy import ScriptedPolicy
from nyssa_bench.plugins import get_plugin_registry


ENGINE_REGISTRY: dict[str, type[NyssaEngine]] = {
    "dummy": DummyEngine,
    "local": DummyEngine,
    "maniskill": ManiSkillEngine,
    "mujoco": MuJoCoEngine,
    "genesis": GenesisEngine,
    "robocasa": RoboCasaEngine,
}

POLICY_REGISTRY: dict[str, type[Policy]] = {
    "random": RandomPolicy,
    "scripted": ScriptedPolicy,
    "oracle": ScriptedPolicy,
    "lerobot": LeRobotPolicy,
    "robomimic": RoboMimicPolicy,
    "diffusion": DiffusionPolicyAdapter,
    "openvla": OpenVLAPolicy,
}


def make_engine(name: str) -> NyssaEngine:
    plugin_engine = get_plugin_registry().engines.get(name)
    if plugin_engine is not None:
        return plugin_engine()
    try:
        engine_cls = ENGINE_REGISTRY[name]
    except KeyError as exc:
        options = ", ".join(sorted(ENGINE_REGISTRY))
        raise ValueError(f"Unknown engine '{name}'. Available engines: {options}") from exc
    return engine_cls()


def make_policy(name: str) -> Policy:
    plugin_policy = get_plugin_registry().policies.get(name)
    if plugin_policy is not None:
        return plugin_policy()
    try:
        policy_cls = POLICY_REGISTRY[name]
    except KeyError as exc:
        options = ", ".join(sorted(POLICY_REGISTRY))
        raise ValueError(f"Unknown policy '{name}'. Available policies: {options}") from exc
    return policy_cls()
