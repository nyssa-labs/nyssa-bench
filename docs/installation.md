# Installation

Use `uv` for the fastest local setup:

```bash
uv sync --extra dev --extra mujoco --extra video --extra reports
uv run nyssa list-suites
uv run pytest -q
uv run ruff check .
```

Optional simulator and dataset extras are installed only when needed:

```bash
uv sync --extra dev --extra maniskill --extra video --extra reports
uv sync --extra dev --extra mujoco --extra video --extra reports
uv sync --extra dev --extra dataset --extra reports --extra video
```

Use `mujoco` for the lightest real backend path and `maniskill` for manipulation tasks.

## Rendering system packages

Python packages are not enough for video-backed robotics benchmarks. Public
NyssaBench benchmark claims require MP4 replay evidence, so install simulator
rendering libraries before running result packs.

On Ubuntu/Debian GPU machines:

```bash
bash scripts/setup_rendering_linux.sh
vulkaninfo --summary
nvidia-smi
```

The helper installs common GL/Vulkan/X11 runtime libraries:

```txt
libvulkan1 vulkan-tools mesa-vulkan-drivers libglvnd0 libgl1 libegl1
libglfw3 libx11-6 libxext6 libxrender1 libxrandr2 libxinerama1
libxcursor1 libxi6
```

If `vulkaninfo --summary` cannot see a Vulkan device, fix the host NVIDIA
driver/ICD setup before running ManiSkill result packs. Video-less runs are
allowed only for local smoke tests with `--no-replay`; they are not public
benchmark results.

On macOS, MuJoCo smoke runs usually need native GLFW:

```bash
brew install glfw
```

ManiSkill video-backed result packs are expected to run on Linux with a working
NVIDIA/Vulkan stack.

If you are not using `uv`, the equivalent pip command is:

```bash
python -m venv .venv
python -m pip install -e ".[dev,mujoco,video,reports]"
```

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
