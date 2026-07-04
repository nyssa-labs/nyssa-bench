#!/usr/bin/env bash
set -euo pipefail

if ! command -v apt-get >/dev/null 2>&1; then
  echo "This helper currently supports Debian/Ubuntu systems with apt-get." >&2
  exit 2
fi

sudo apt-get update
sudo apt-get install -y \
  libvulkan1 \
  vulkan-tools \
  mesa-vulkan-drivers \
  libglvnd0 \
  libgl1 \
  libegl1 \
  libglfw3 \
  libx11-6 \
  libxext6 \
  libxrender1 \
  libxrandr2 \
  libxinerama1 \
  libxcursor1 \
  libxi6

echo "Installed common MuJoCo/ManiSkill rendering libraries."
echo "Checking Vulkan visibility:"
if command -v vulkaninfo >/dev/null 2>&1; then
  vulkaninfo --summary || true
else
  echo "vulkaninfo is not available."
fi

echo "Checking NVIDIA visibility:"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi is not available. CPU/Mesa rendering may still work for MuJoCo, but ManiSkill video capture usually needs a working GPU/Vulkan stack."
fi
