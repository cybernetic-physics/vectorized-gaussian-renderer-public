#!/usr/bin/env bash
set -euo pipefail

remote_root="${1:?usage: provision_lidar_node.sh REMOTE_ROOT}"
run_id="${2:-gaussian-lidar}"
src_root=/workspace/src
tools_root=/workspace/tools

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y \
  build-essential ca-certificates cmake curl git git-lfs htop jq libcap2-bin \
  libx11-xcb1 libxcb-dri2-0 libxcb-dri3-0 libxcb-glx0 libxcb-present0 libxcb-sync1 \
  libxt6 ninja-build nodejs npm python3-venv ripgrep rsync tmux wget \
  mesa-utils vulkan-tools vulkan-validationlayers xvfb ffmpeg imagemagick

if ! test -x /usr/local/cuda-12.8/bin/nvcc; then
  keyring=/tmp/cuda-keyring_1.1-1_all.deb
  curl --fail --location --silent --show-error \
    --output "$keyring" \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
  dpkg -i "$keyring"
  apt-get update
  apt-get install -y cuda-toolkit-12-8
fi

mkdir -p "$src_root" "$tools_root/ovrtx-0.3.0/python" "$remote_root/outputs/logs"

nsight_graphics_home=/opt/nvidia/nsight-graphics-for-linux/nsight-graphics-for-linux-2026.2.0.0/NVIDIA-Nsight-Graphics-2026.2/host/linux-desktop-nomad-x64
if ! test -x "$nsight_graphics_home/ngfx"; then
  nsight_graphics_installer="$tools_root/NVIDIA_Nsight_Graphics_2026.2.0.26134-linux_x64.run"
  curl --fail --location --silent --show-error \
    --output "$nsight_graphics_installer" \
    https://developer.nvidia.com/downloads/assets/tools/secure/nsight-graphics/2026_2_0/linux_x64/NVIDIA_Nsight_Graphics_2026.2.0.26134-linux_x64.run
  echo '817c24494a71a7389d77613b659eff9b72a9747a9e9dd703510f49dbad10a089  '"$nsight_graphics_installer" | sha256sum --check
  chmod +x "$nsight_graphics_installer"
  "$nsight_graphics_installer" -- -noprompt \
    -targetpath=/opt/nvidia/nsight-graphics-for-linux/nsight-graphics-for-linux-2026.2.0.0
fi

if ! test -d "$src_root/ovrtx/.git"; then
  git clone --filter=blob:none https://github.com/NVIDIA-Omniverse/ovrtx.git "$src_root/ovrtx"
fi
git -C "$src_root/ovrtx" fetch origin 29d11037fbcaed0f0f53e7f32d17bd0486fd453b
git -C "$src_root/ovrtx" checkout --detach 29d11037fbcaed0f0f53e7f32d17bd0486fd453b

if ! test -d "$src_root/kit-app-template/.git"; then
  git clone --branch main --single-branch \
    https://github.com/NVIDIA-Omniverse/kit-app-template.git \
    "$src_root/kit-app-template"
fi

if ! test -d "$src_root/gsplat/.git"; then
  git clone https://github.com/nerfstudio-project/gsplat.git "$src_root/gsplat"
fi
git -C "$src_root/gsplat" fetch origin 77ab983ffe43420b2131669cb35776b883ca4c3c
git -C "$src_root/gsplat" checkout --detach 77ab983ffe43420b2131669cb35776b883ca4c3c
git -C "$src_root/gsplat" submodule update --init --recursive

cd "$remote_root"
source scripts/remote_env.sh
/isaac-sim/python.sh -m pip install --no-deps \
  'lazy_loader>=0.4' 'numpy==1.26.4' 'pillow>=10' 'plyfile==1.1.3'
/isaac-sim/python.sh -m pip install setuptools wheel pytest
/isaac-sim/python.sh -m pip install --no-deps --target "$OVRTX_ROOT/python" \
  ovrtx==0.3.0.312915
/isaac-sim/python.sh -m pip install --no-deps -e "$remote_root"

git -C "$src_root/ovrtx" rev-parse HEAD
git -C "$src_root/gsplat" rev-parse HEAD
git -C "$src_root/kit-app-template" rev-parse HEAD
