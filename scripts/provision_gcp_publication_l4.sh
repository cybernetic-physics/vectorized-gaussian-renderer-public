#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${VGR_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUNTIME_ROOT="${VGR_RUNTIME_ROOT:-/workspace/venvs/vgr-publication-l4}"
SOURCE_ROOT="${VGR_SOURCE_ROOT:-/workspace/src}"
DATASET_ROOT="${VGR_DATASET_ROOT:-/workspace/datasets}"
TOOLS_ROOT="${VGR_TOOLS_ROOT:-/workspace/tools}"
PROVENANCE_ROOT="${VGR_PROVENANCE_ROOT:-/workspace/provenance/vgr-publication-l4}"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"

if [[ -x /usr/local/cuda/bin/nvcc ]]; then
  export PATH="/usr/local/cuda/bin:$PATH"
fi

GSPLAT_COMMIT=77ab983ffe43420b2131669cb35776b883ca4c3c
FLASHGS_COMMIT=cdfc4e4002318423eda356eed02df8e01fa32cb6
OVRTX_COMMIT=29d11037fbcaed0f0f53e7f32d17bd0486fd453b
KIT_TEMPLATE_COMMIT=483e364a4176f102f2d3c3aaf9f301a103d61d69
OPTIX_COMMIT=f1f6dd803f3159992d248178f6e09421c6eb8b6d

HOME_SHA256=29cee159465406d94f2b24954eefb9da76ba80cab827b558a6e75676b8809267
HOME_URL=https://pub-243008be935848b6accaf262f04a7b82.r2.dev/datasets/home-scan-lod0/29cee1594654/home-scan-lod0.ply
GARAGE_SHA256=cf8d65138dbc31e3619ce697a875b1a005fe88fa126d5737e66294b9064a171a
GARAGE_URL=https://pub-243008be935848b6accaf262f04a7b82.r2.dev/datasets/garage-high-detail/cf8d65138dbc/garage-high-detail.ply

if [[ ! -f "$PROJECT_ROOT/pyproject.toml" ]]; then
  echo "VGR project root is invalid: $PROJECT_ROOT" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "An NVIDIA driver is required." >&2
  exit 2
fi

NVIDIA_QUERY="$(nvidia-smi -q)"
if ! grep -Eq \
  'Product Name[[:space:]]+:[[:space:]]+NVIDIA RTX Virtual Workstation' \
  <<< "$NVIDIA_QUERY" || \
   ! grep -Eq 'License Status[[:space:]]+:[[:space:]]+Licensed' \
  <<< "$NVIDIA_QUERY"; then
  echo \
    "Publication graphics gates require an nvidia-l4-vws VM and a licensed NVIDIA GRID driver." \
    >&2
  exit 2
fi
if [[ ! -r /proc/driver/nvidia/params ]] || \
   [[ "$(awk '/^EnableGpuFirmware:/{print $2}' /proc/driver/nvidia/params)" != 0 ]]; then
  echo "Linux L4 vWS publication requires NVIDIA GSP firmware to be disabled." >&2
  exit 2
fi

export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo apt-get install -y \
  build-essential ca-certificates cmake curl ffmpeg git git-lfs jq \
  libcap2-bin libgl1 libglib2.0-0 libglu1-mesa libsm6 libvulkan1 \
  libx11-6 libx11-xcb1 libxcb-dri2-0 libxcb-dri3-0 libxcb-glx0 \
  libxcb-present0 libxcb-sync1 libxext6 libxrender1 libxt6 ninja-build \
  pkg-config python3-pip ripgrep rsync tmux unzip vulkan-tools wget xvfb

sudo mkdir -p /workspace
sudo chown "$(id -un):$(id -gn)" /workspace
mkdir -p \
  "$HOME/.local/bin" \
  "$RUNTIME_ROOT" \
  "$SOURCE_ROOT" \
  "$DATASET_ROOT" \
  "$TOOLS_ROOT/ovrtx-0.3.0/python" \
  "$PROVENANCE_ROOT"

VULKAN_ICD=""
for candidate in \
  /etc/vulkan/icd.d/nvidia_icd.json \
  /usr/share/vulkan/icd.d/nvidia_icd.json; do
  if [[ -f "$candidate" ]]; then
    resolved="$(readlink -f "$candidate")"
    if [[ -n "$VULKAN_ICD" && "$VULKAN_ICD" != "$resolved" ]]; then
      echo "Multiple distinct NVIDIA Vulkan ICD manifests are installed." >&2
      exit 2
    fi
    VULKAN_ICD="$resolved"
  fi
done
if [[ -z "$VULKAN_ICD" ]]; then
  echo "No NVIDIA Vulkan ICD manifest is installed." >&2
  exit 2
fi
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/xdg-runtime-vgr-publication}"
install -d -m 700 "$XDG_RUNTIME_DIR"
VK_ICD_FILENAMES="$VULKAN_ICD" timeout 60s xvfb-run -a vulkaninfo --summary \
  > "$PROVENANCE_ROOT/vulkaninfo-summary.txt" 2>&1
if ! grep -Eq 'deviceName[[:space:]]*=[[:space:]]*NVIDIA L4' \
  "$PROVENANCE_ROOT/vulkaninfo-summary.txt"; then
  echo "Vulkan did not expose the NVIDIA L4." >&2
  exit 2
fi

if [[ ! -x "$UV_BIN" ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | \
    env UV_INSTALL_DIR="$HOME/.local/bin" sh
fi

"$UV_BIN" python install 3.12
if [[ ! -x "$RUNTIME_ROOT/bin/python" ]]; then
  "$UV_BIN" venv --python 3.12 "$RUNTIME_ROOT"
fi

"$UV_BIN" pip install --python "$RUNTIME_ROOT/bin/python" \
  --index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.11.0' 'torchvision==0.26.0'
"$UV_BIN" pip install --python "$RUNTIME_ROOT/bin/python" \
  'ninja>=1.11' 'pytest==8.4.1' 'plyfile==1.1.3' \
  'pillow>=10' 'scipy>=1.14,<2' 'lpips==0.1.4' \
  'setuptools>=77' wheel \
  'jaxtyping==0.3.11' 'nvtx==0.2.15' 'rich==15.0.0'
"$UV_BIN" pip install --python "$RUNTIME_ROOT/bin/python" \
  --extra-index-url https://pypi.nvidia.com \
  --index-strategy unsafe-best-match \
  --prerelease=allow \
  'isaacsim[all,extscache]==6.0.1'
"$UV_BIN" pip install --python "$RUNTIME_ROOT/bin/python" --no-deps \
  --target "$TOOLS_ROOT/ovrtx-0.3.0/python" \
  --extra-index-url https://pypi.nvidia.com \
  --index-strategy unsafe-best-match \
  'ovrtx==0.3.0.312915'
"$UV_BIN" pip install --python "$RUNTIME_ROOT/bin/python" --no-deps \
  -e "$PROJECT_ROOT"
"$UV_BIN" pip check --python "$RUNTIME_ROOT/bin/python"
"$RUNTIME_ROOT/bin/python" - <<'PY'
import jaxtyping  # noqa: F401
import nvtx  # noqa: F401
import rich  # noqa: F401
PY

clone_pinned() {
  local url="$1"
  local destination="$2"
  local commit="$3"
  if [[ ! -d "$destination/.git" ]]; then
    git clone --filter=blob:none "$url" "$destination"
  fi
  if [[ -n "$(git -C "$destination" status --short --untracked-files=all)" ]]; then
    echo "Refusing to alter dirty dependency checkout: $destination" >&2
    exit 3
  fi
  git -C "$destination" fetch origin "$commit"
  git -C "$destination" checkout --detach "$commit"
  test "$(git -C "$destination" rev-parse HEAD)" = "$commit"
}

clone_pinned \
  https://github.com/InternLandMark/FlashGS.git \
  "$SOURCE_ROOT/FlashGS" \
  "$FLASHGS_COMMIT"
clone_pinned \
  https://github.com/NVIDIA-Omniverse/ovrtx.git \
  "$SOURCE_ROOT/ovrtx" \
  "$OVRTX_COMMIT"
clone_pinned \
  https://github.com/NVIDIA-Omniverse/kit-app-template.git \
  "$SOURCE_ROOT/kit-app-template" \
  "$KIT_TEMPLATE_COMMIT"
clone_pinned \
  https://github.com/NVIDIA/optix-dev.git \
  "$SOURCE_ROOT/optix-dev" \
  "$OPTIX_COMMIT"

if [[ ! -d "$SOURCE_ROOT/gsplat/.git" ]]; then
  git clone --filter=blob:none \
    https://github.com/nerfstudio-project/gsplat.git \
    "$SOURCE_ROOT/gsplat"
fi
git -C "$SOURCE_ROOT/gsplat" fetch origin "$GSPLAT_COMMIT"
git -C "$SOURCE_ROOT/gsplat" checkout --detach "$GSPLAT_COMMIT"
git -C "$SOURCE_ROOT/gsplat" submodule update --init --recursive
GSPLAT_PATCH="$PROJECT_ROOT/patches/gsplat-cuda-event-flags.patch"
if git -C "$SOURCE_ROOT/gsplat" apply --reverse --check "$GSPLAT_PATCH" 2>/dev/null; then
  :
elif [[ -z "$(git -C "$SOURCE_ROOT/gsplat" status --short --untracked-files=all)" ]]; then
  git -C "$SOURCE_ROOT/gsplat" apply "$GSPLAT_PATCH"
else
  echo "gsplat is dirty but does not contain the recorded compatibility patch." >&2
  exit 3
fi
test "$(git -C "$SOURCE_ROOT/gsplat" rev-parse HEAD)" = "$GSPLAT_COMMIT"
test "$(git -C "$SOURCE_ROOT/gsplat" diff --name-only)" = \
  gsplat/cuda/csrc/Utils.cpp

download_dataset() {
  local url="$1"
  local destination="$2"
  local sha256="$3"
  if [[ ! -f "$destination" ]]; then
    curl --fail --location --retry 5 --retry-delay 2 \
      --output "$destination.part" "$url"
    mv "$destination.part" "$destination"
  fi
  printf '%s  %s\n' "$sha256" "$destination" | sha256sum --check
}

download_dataset "$HOME_URL" "$DATASET_ROOT/home-scan-lod0.ply" "$HOME_SHA256"
download_dataset "$GARAGE_URL" "$DATASET_ROOT/garage-high-detail.ply" "$GARAGE_SHA256"

ISAAC_SHIM="$HOME/isaac-sim-shim"
ISAAC_PACKAGE="$RUNTIME_ROOT/lib/python3.12/site-packages/isaacsim"
mkdir -p "$ISAAC_SHIM"
ln -sfn "$ISAAC_PACKAGE/exts" "$ISAAC_SHIM/exts"
ln -sfn "$ISAAC_PACKAGE/extsDeprecated" "$ISAAC_SHIM/extsDeprecated"
ln -sfn "$ISAAC_PACKAGE/extscache" "$ISAAC_SHIM/extscache"
ln -sfn "$ISAAC_PACKAGE/apps" "$ISAAC_SHIM/apps"
printf '#!/usr/bin/env bash\nexec %q "$@"\n' "$RUNTIME_ROOT/bin/python" \
  > "$ISAAC_SHIM/python.sh"
chmod +x "$ISAAC_SHIM/python.sh"
if [[ -e /isaac-sim && ! -L /isaac-sim ]]; then
  echo "/isaac-sim exists and is not a managed symlink." >&2
  exit 3
fi
sudo ln -sfn "$ISAAC_SHIM" /isaac-sim

"$UV_BIN" pip freeze --python "$RUNTIME_ROOT/bin/python" | \
  LC_ALL=C sort > "$PROVENANCE_ROOT/python-freeze.txt"

{
  echo PROVISION_OK
  date -u +%FT%TZ
  printf 'project_root=%s\n' "$PROJECT_ROOT"
  printf 'runtime_root=%s\n' "$RUNTIME_ROOT"
  printf 'vulkan_icd=%s\n' "$VULKAN_ICD"
  printf 'vulkan_summary_sha256=%s\n' \
    "$(sha256sum "$PROVENANCE_ROOT/vulkaninfo-summary.txt" | cut -d' ' -f1)"
  nvidia-smi \
    --query-gpu=name,uuid,driver_version,memory.total \
    --format=csv,noheader
  nvcc --version
  nsys --version
  ncu --version
  compute-sanitizer --version
  ffmpeg -version | head -1
  ffprobe -version | head -1
  "$RUNTIME_ROOT/bin/python" - <<'PY'
import importlib.metadata as metadata
import torch

print("torch", torch.__version__, "cuda_runtime", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
for name in (
    "isaacsim",
    "numpy",
    "pillow",
    "plyfile",
    "lpips",
    "jaxtyping",
    "nvtx",
    "rich",
):
    print(name, metadata.version(name))
PY
  printf 'python_freeze_sha256=%s\n' "$(sha256sum "$PROVENANCE_ROOT/python-freeze.txt" | cut -d' ' -f1)"
  printf 'FlashGS=%s\n' "$(git -C "$SOURCE_ROOT/FlashGS" rev-parse HEAD)"
  printf 'gsplat=%s\n' "$(git -C "$SOURCE_ROOT/gsplat" rev-parse HEAD)"
  printf 'ovrtx=%s\n' "$(git -C "$SOURCE_ROOT/ovrtx" rev-parse HEAD)"
  printf 'kit_app_template=%s\n' "$(git -C "$SOURCE_ROOT/kit-app-template" rev-parse HEAD)"
  printf 'optix_dev=%s\n' "$(git -C "$SOURCE_ROOT/optix-dev" rev-parse HEAD)"
} | tee "$PROVENANCE_ROOT/provision-summary.txt"
