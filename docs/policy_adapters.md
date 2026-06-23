# Policy Adapters

Policies implement:

```python
class Policy:
    def reset(self, task=None, seed=None): ...
    def act(self, observation): ...
    def close(self): ...
```

Built-ins include `random`, `scripted`, `robomimic`, `diffusion`, and `openvla`.

The robomimic, diffusion, and OpenVLA adapters can wrap real loaded models when used from Python. When invoked by name from the CLI, they run deterministic dummy-engine baselines so comparison reports can be smoke-tested without heavyweight model installs.

Install optional policy stacks as needed:

```bash
pip install -e ".[lerobot]"
pip install -e ".[robomimic]"
pip install -e ".[vla]"
pip install -e ".[diffusion]"
```

The `vla` extra covers common PyTorch/Transformers dependencies. Install OpenVLA model code and checkpoints from the upstream project when running a real OpenVLA policy.
