from typing import Any

from nyssa_bench.core.registry import make_policy
from nyssa_bench.plugins import NyssaPlugin, get_plugin_registry, register_plugin


class ConstantPolicy:
    def act(self, observation: dict[str, Any]) -> float:
        return 0.1


class ExamplePlugin:
    name = "example"

    def register(self, registry):
        registry.policies["constant"] = ConstantPolicy


def test_plugin_policy_registration():
    plugin: NyssaPlugin = ExamplePlugin()
    register_plugin(plugin)

    assert get_plugin_registry().policies["constant"] is ConstantPolicy
    assert isinstance(make_policy("constant"), ConstantPolicy)


def test_builtin_external_policy_adapters_are_registered():
    for policy_name in ["lerobot", "openvla", "robomimic", "diffusion"]:
        policy = make_policy(policy_name)
        action = policy.act({"state": {"distance": 0.4}})
        assert isinstance(action, float)
