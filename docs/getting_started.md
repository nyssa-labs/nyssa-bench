# Getting Started

NyssaBench evaluates robot policies through a simulator adapter. Use the dummy engine for installation checks, then switch to ManiSkill or MuJoCo once those simulators are installed.

```bash
pip install -e ".[dev]"
nyssa run --suite tabletop_manipulation_v0 --engine dummy --policy scripted --episodes 5 --out runs/quickstart
nyssa report runs/quickstart
```
