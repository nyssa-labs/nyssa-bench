# Getting Started

NyssaBench evaluates robot policies through a simulator adapter. Install ManiSkill or MuJoCo before running benchmark episodes.

```bash
pip install -e ".[dev,mujoco]"
nyssa run --suite mujoco_control_v0 --engine mujoco --policy random --episodes 5 --out runs/quickstart
nyssa report runs/quickstart
```
