# Policy Adapters

Policies implement:

```python
class Policy:
    def reset(self, task=None, seed=None): ...
    def act(self, observation): ...
    def close(self): ...
```

Built-ins include `random`, `scripted`, `lerobot`, `robomimic`, `diffusion`, and `openvla`.

The LeRobot, robomimic, diffusion, and OpenVLA adapters can wrap real loaded models when used from Python. When invoked by name from the CLI, they read an optional model factory from the environment and otherwise run deterministic dummy-engine baselines so comparison reports can be smoke-tested without heavyweight model installs.

```bash
NYSSA_LEROBOT_POLICY=my_project.policies:create_lerobot_policy nyssa run --policy lerobot ...
NYSSA_ROBOMIMIC_POLICY=my_project.policies:create_robomimic_policy nyssa run --policy robomimic ...
NYSSA_DIFFUSION_POLICY=my_project.policies:create_diffusion_policy nyssa run --policy diffusion ...
NYSSA_OPENVLA_POLICY=my_project.policies:create_openvla_policy nyssa run --policy openvla ...
```

Install optional policy stacks as needed:

```bash
pip install -e ".[lerobot]"
pip install -e ".[robomimic]"
pip install -e ".[vla]"
pip install -e ".[diffusion]"
```

The `vla` extra covers common PyTorch/Transformers dependencies. Install OpenVLA model code and checkpoints from the upstream project when running a real OpenVLA policy.
