# Installation

Install the lightweight local package first:

```bash
pip install -e ".[dev]"
```

Optional simulator and dataset extras are installed only when needed:

```bash
pip install -e ".[maniskill]"
pip install -e ".[mujoco]"
pip install -e ".[dataset]"
pip install -e ".[reports]"
pip install -e ".[video]"
```

The dummy engine requires no robotics simulator and is the recommended first validation path.

## Extras

| Extra | Purpose |
| --- | --- |
| `cli` | Rich/Typer dependencies for future polished terminal UI. |
| `dataset` | HDF5 and Parquet export. |
| `reports` | Template and plotting dependencies. |
| `video` | MP4/frame export dependencies. |
| `maniskill` | ManiSkill adapter runtime dependencies. |
| `mujoco` | MuJoCo adapter runtime dependencies. |
| `lerobot` | LeRobot policy and dataset integration dependencies. |
| `robomimic` | robomimic baseline dependencies. |
| `robocasa` | Best-effort source install for RoboCasa and its robosuite dependency. |
| `vla` | Shared PyTorch/Transformers dependencies for VLA adapters such as OpenVLA. |
| `diffusion` | Diffusion policy baseline dependencies. |
| `experimental` | Experimental Genesis dependency. RoboCasa may still require a source install depending on upstream packaging. |
| `all` | Everything except experimental backends. |
| `full` | All declared extras, including experimental/source backend dependencies. |

OpenVLA and some robotics diffusion-policy codebases are commonly installed from their upstream GitHub repositories rather than as stable PyPI packages. NyssaBench declares their common runtime stack in `vla` and `diffusion`, while the model code/checkpoints should be installed according to the upstream project instructions.

RoboCasa requires additional setup beyond Python package installation. Follow the upstream RoboCasa docs to set up macros and download kitchen assets after installing the source packages:

```bash
python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets
```
