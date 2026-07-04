from __future__ import annotations

from nyssa_bench.engines.base import NyssaEngine
from nyssa_bench.engines.genesis_adapter import GenesisEngine
from nyssa_bench.engines.maniskill_adapter import ManiSkillEngine
from nyssa_bench.engines.mujoco_adapter import MuJoCoEngine
from nyssa_bench.engines.robocasa_adapter import RoboCasaEngine
from nyssa_bench.policies.bc_policy import BCPolicy
from nyssa_bench.policies.base import Policy
from nyssa_bench.policies.diffusion_policy_adapter import DiffusionPolicyAdapter
from nyssa_bench.policies.lerobot_adapter import LeRobotPolicy
from nyssa_bench.policies.openvla_adapter import OpenVLAPolicy
from nyssa_bench.policies.random_policy import RandomPolicy
from nyssa_bench.policies.robomimic_adapter import RoboMimicPolicy, TaskRoboMimicPolicy
from nyssa_bench.policies.scripted_oracle_policy import ScriptedOraclePolicy
from nyssa_bench.policies.task_bc_policy import TaskBCPolicy
from nyssa_bench.plugins import get_plugin_registry


ENGINE_REGISTRY: dict[str, type[NyssaEngine]] = {
    "maniskill": ManiSkillEngine,
    "mujoco": MuJoCoEngine,
    "genesis": GenesisEngine,
    "robocasa": RoboCasaEngine,
}

ENGINE_SUPPORT_TIER: dict[str, str] = {
    "maniskill": "supported_real_simulator",
    "mujoco": "supported_real_simulator",
    "genesis": "experimental_contract_only",
    "robocasa": "experimental_contract_only",
}

POLICY_REGISTRY: dict[str, type[Policy]] = {
    "random": RandomPolicy,
    "scripted_oracle": ScriptedOraclePolicy,
    "bc_policy": BCPolicy,
    "task_bc_policy": TaskBCPolicy,
    "lerobot": LeRobotPolicy,
    "robomimic": RoboMimicPolicy,
    "task_robomimic": TaskRoboMimicPolicy,
    "diffusion": DiffusionPolicyAdapter,
    "openvla": OpenVLAPolicy,
}

POLICY_SUPPORT_TIER: dict[str, str] = {
    "random": "sanity_baseline",
    "scripted_oracle": "oracle_baseline_adapter",
    "bc_policy": "learned_baseline_adapter",
    "task_bc_policy": "task_routed_learned_baseline_adapter",
    "lerobot": "adapter_hook",
    "robomimic": "adapter_hook",
    "task_robomimic": "task_routed_robomimic_adapter",
    "diffusion": "adapter_hook",
    "openvla": "adapter_hook",
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
