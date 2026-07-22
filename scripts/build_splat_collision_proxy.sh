#!/usr/bin/env bash
# Build a deliberately coarse PhysX-ready collision proxy from a canonical PLY.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 6 ]]; then
  echo "usage: $0 INPUT.ply OUTPUT.voxel.json [seed-x,y,z] [voxel-size] [opacity] [interior|exterior]" >&2
  exit 2
fi

input="$1"
output="$2"
seed="${3:-0,0,0}"
voxel_size="${4:-0.08}"
opacity="${5:-0.10}"
mode="${6:-interior}"

command -v splat-transform >/dev/null
test -f "$input"
case "$output" in
  *.voxel.json) ;;
  *) echo "OUTPUT must end in .voxel.json" >&2; exit 2 ;;
esac
case "$mode" in
  interior)
    fill_args=(--voxel-carve "1.6,0.2" --seed-pos "$seed")
    ;;
  exterior)
    fill_args=(--voxel-external-fill 1.6)
    ;;
  *)
    echo "mode must be interior or exterior" >&2
    exit 2
    ;;
esac

mkdir -p "$(dirname "$output")"
splat-transform \
  --gpu 0 \
  "$input" \
  --voxel-params "$voxel_size,$opacity" \
  "${fill_args[@]}" \
  --collision-mesh faces \
  "$output"

collision_mesh="${output%.voxel.json}.collision.glb"
test -f "$output"
test -f "$collision_mesh"
printf 'SPLAT_COLLISION_PROXY_OK input=%s voxel=%s mesh=%s mode=%s\n' \
  "$input" "$output" "$collision_mesh" "$mode"
