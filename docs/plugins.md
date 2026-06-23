# Plugins

Plugins let external packages register engines, policies, suites, or report extensions without editing NyssaBench core.

Minimal plugin:

```python
from nyssa_bench.plugins import NyssaPlugin, register_plugin

class MyPlugin(NyssaPlugin):
    name = "my_plugin"

    def register(self, registry):
        registry.policies["my_policy"] = MyPolicy

register_plugin(MyPlugin())
```
