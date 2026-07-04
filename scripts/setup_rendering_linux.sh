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

if command -v nvidia-smi >/dev/null 2>&1; then
  driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 || true)"
  driver_branch="${driver_version%%.*}"
  if [[ -n "${driver_branch}" && "${driver_branch}" =~ ^[0-9]+$ ]]; then
    echo "Detected NVIDIA driver branch ${driver_branch}; installing matching Vulkan/GL packages if available."
    sudo apt-get install -y "nvidia-utils-${driver_branch}" "libnvidia-gl-${driver_branch}" || {
      echo "Could not install nvidia-utils-${driver_branch}/libnvidia-gl-${driver_branch} from apt." >&2
      echo "If Vulkan only shows llvmpipe, install the NVIDIA Vulkan ICD package that matches your host driver." >&2
    }
  fi
fi

echo "Installed common MuJoCo/ManiSkill rendering libraries."
echo "Checking Vulkan visibility:"
if command -v vulkaninfo >/dev/null 2>&1; then
  vulkan_summary="$(vulkaninfo --summary 2>&1 || true)"
  echo "${vulkan_summary}"
  if echo "${vulkan_summary}" | grep -qi "llvmpipe"; then
    echo "WARNING: Vulkan is only seeing llvmpipe CPU rendering." >&2
    echo "ManiSkill video-backed benchmark runs need an NVIDIA Vulkan device, not only llvmpipe." >&2
  fi
else
  echo "vulkaninfo is not available."
fi

echo "Checking NVIDIA visibility:"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi is not available. CPU/Mesa rendering may still work for MuJoCo, but ManiSkill video capture usually needs a working GPU/Vulkan stack."
fi
