from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class NyssaPlugin(Protocol):
    name: str

    def register(self, registry: "PluginRegistry") -> None:
        raise NotImplementedError


@dataclass
class PluginRegistry:
    engines: dict[str, object] = field(default_factory=dict)
    policies: dict[str, object] = field(default_factory=dict)
    suites: dict[str, object] = field(default_factory=dict)
    reports: dict[str, object] = field(default_factory=dict)


_REGISTRY = PluginRegistry()


def get_plugin_registry() -> PluginRegistry:
    return _REGISTRY


def register_plugin(plugin: NyssaPlugin) -> PluginRegistry:
    plugin.register(_REGISTRY)
    return _REGISTRY
