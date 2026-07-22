"""Mechanical source audit for the pinned upstream-faithful FlashGS lane.

The control lane compiles upstream ``sort.cu`` and ``render.cu`` byte-for-byte.
Only ``preprocess.cu`` needs a generated glue variant so camera tensors can stay
on the GPU and key emission can be bounded without a host counter readback.
Every accepted upstream file is content-addressed here; a source change is a
hard failure rather than an implicit baseline update.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FLASHGS_UPSTREAM_COMMIT = "cdfc4e4002318423eda356eed02df8e01fa32cb6"
PINNED_SOURCE_SHA256 = {
    "setup.py": "16262dbb94ab1a0c5c2d1abda00272a12806ec555178f3c149a75b29432d10d1",
    "csrc/ops.h": "a4fbe9ea602d58cb93c83100dcd904bc0f198c9f62753e903468bc9686e3af09",
    "csrc/pybind.cpp": "0e83621907ec27a20833f4e99969703852d765d05e52ea0520dc2cbeff4d232f",
    "csrc/cuda_rasterizer/preprocess.cu": "a571d8288523207e7beeff60e67ad2db509fdb902cdf7619fe97e6f9c56a734d",
    "csrc/cuda_rasterizer/render.cu": "acacc5b973bd53e27d245518a89da307d339e53f293ddca60b5f41b7ed64dd74",
    "csrc/cuda_rasterizer/sort.cu": "6850b8ae3a3d69b1cd53d98ca1b73112bc0580c3b6952291249591b6a2a13439",
}

# Independently reviewed output of ``transform_preprocess_source`` for the
# pinned source above.  The source-input hash alone is insufficient: without a
# frozen generated hash, an accidental change to this repository's transform
# could still emit an audit record whose self-computed ``pass`` field is true.
PINNED_GENERATED_PREPROCESS_SHA256 = (
    "d5a81199325d9673f065d84c90decdf56471e48e139841e49d2f39a5ad0bb82f"
)
PINNED_PREPROCESS_DIFF = {
    "added_lines": 54,
    "removed_lines": 29,
    "hunks": 9,
    "unified_diff_sha256": (
        "52467008dd6419460976ad9b0243704337144112a2de864e34ce2babfffc31b4"
    ),
}

PRESERVED_PREPROCESS_FUNCTIONS = (
    "fast_max_f32",
    "fast_sqrt_f32",
    "fast_rsqrt_f32",
    "fast_lg2_f32",
    "ndc2Pix",
    "transformPoint4x3",
    "transformPoint4x4",
    "getRect",
    "computeCov2D",
    "computeColorFromSH",
    "segment_intersect_ellipse",
    "block_intersect_ellipse",
    "block_contains_center",
    "getViewMatrix",
    "getProjectionMatrix",
)


@dataclass(frozen=True)
class SourceTransform:
    source: str
    audit: dict[str, Any]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _replace_once(source: str, old: str, new: str, category: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"Pinned FlashGS transform {category!r} expected one exact source "
            f"match, found {count}."
        )
    return source.replace(old, new, 1)


def _function_source(source: str, name: str) -> str:
    match = re.search(rf"\b{re.escape(name)}\s*\([^;]*?\)\s*\{{", source, re.DOTALL)
    if match is None:
        raise RuntimeError(f"Cannot locate pinned FlashGS function {name!r}.")
    start = match.start()
    brace = source.find("{", match.start())
    depth = 0
    for index in range(brace, len(source)):
        token = source[index]
        if token == "{":
            depth += 1
        elif token == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise RuntimeError(f"Pinned FlashGS function {name!r} has no closing brace.")


def function_hashes(source: str) -> dict[str, str]:
    return {
        name: sha256_bytes(_function_source(source, name).encode("utf-8"))
        for name in PRESERVED_PREPROCESS_FUNCTIONS
    }


def verify_pinned_source_files(source_root: Path) -> dict[str, Any]:
    """Fail closed unless every equation-bearing upstream file is exact."""
    files: dict[str, Any] = {}
    for relative, expected in PINNED_SOURCE_SHA256.items():
        path = source_root / relative
        if not path.is_file():
            raise RuntimeError(f"Pinned FlashGS source is missing: {path}.")
        actual = sha256_bytes(path.read_bytes())
        if actual != expected:
            raise RuntimeError(
                f"Pinned FlashGS source drift for {relative}: {actual} != {expected}."
            )
        files[relative] = {
            "path": str(path.resolve()),
            "sha256": actual,
            "bytes": path.stat().st_size,
        }
    return {
        "schema_version": "upstream-faithful-flashgs-source-audit-v1",
        "upstream_commit": FLASHGS_UPSTREAM_COMMIT,
        "files": files,
        "pass": True,
    }


def transform_preprocess_source(source: str) -> SourceTransform:
    """Generate the bounded GPU-camera entrypoint from exact upstream text.

    Projection, covariance projection, support/tile tests, SH evaluation, and
    the upstream camera matrix helpers stay byte-identical. The only modified
    portions are input plumbing, canonical degree-zero color selection, and
    capacity-checked key writes. The compositor is not generated here at all;
    the exact upstream ``render.cu`` is linked directly.
    """
    source_hash = sha256_bytes(source.encode("utf-8"))
    expected_hash = PINNED_SOURCE_SHA256["csrc/cuda_rasterizer/preprocess.cu"]
    if source_hash != expected_hash:
        raise RuntimeError(
            "Refusing to transform unpinned FlashGS preprocess source: "
            f"{source_hash} != {expected_hash}."
        )
    before_functions = function_hashes(source)
    transformed = source
    categories: dict[str, int] = {}

    transformed = _replace_once(
        transformed,
        '#include "../ops.h"',
        '#include <ops.h>\n#include "faithful_glue.h"',
        "include_path",
    )
    categories["include_path"] = 1
    transformed = _replace_once(
        transformed,
        '#include "../glm/glm.hpp"',
        "#include <glm/glm.hpp>",
        "include_path",
    )
    categories["include_path"] += 1

    color_anchor = """__forceinline__ __device__ bool segment_intersect_ellipse(float a, float b, float c, float d, float l, float r)
{"""
    degree_zero_helper = """__forceinline__ __device__ glm::vec3 computeColorFromDegreeZero(
	int idx, const glm::vec3* colors)
{
	glm::vec3 result = colors[idx];
	result.x = fast_max_f32(result.x, 0.0f);
	result.y = fast_max_f32(result.y, 0.0f);
	result.z = fast_max_f32(result.z, 0.0f);
	return result;
}

"""
    transformed = _replace_once(
        transformed,
        color_anchor,
        degree_zero_helper + color_anchor,
        "degree_zero_color",
    )
    categories["degree_zero_color"] = 1

    transformed = _replace_once(
        transformed,
        "\tconst shs_deg3_t* __restrict__ shs,\n\tglm::mat4 viewmatrix,\n\tglm::mat4 projmatrix,\n\tglm::vec3 cam_position,",
        "\tconst glm::vec3* __restrict__ colors,\n"
        "\tconst upstream_camera_state_t* __restrict__ camera_state,",
        "camera_state",
    )
    categories["camera_state"] = 1
    transformed = _replace_once(
        transformed,
        "\tconst float tan_fovx, float tan_fovy,\n\tconst float focal_x, float focal_y,",
        "",
        "camera_state",
    )
    categories["camera_state"] += 1
    transformed = _replace_once(
        transformed,
        "\tint* __restrict__ curr_offset,\n\tuint64_t* __restrict__ gaussian_keys_unsorted,\n\tuint32_t* __restrict__ gaussian_values_unsorted,\n\tconst dim3 grid)",
        "\tint* __restrict__ curr_offset,\n"
        "\tuint64_t* __restrict__ gaussian_keys_unsorted,\n"
        "\tuint32_t* __restrict__ gaussian_values_unsorted,\n"
        "\tconst int capacity,\n"
        "\tint* __restrict__ overflow,\n"
        "\tconst dim3 grid)",
        "bounded_emission",
    )
    categories["bounded_emission"] = 1
    transformed = _replace_once(
        transformed,
        "\tint idx_vec = warp_id * FLASHGS_WARP_SIZE + lane;\n",
        "\tint idx_vec = warp_id * FLASHGS_WARP_SIZE + lane;\n"
        "\tconst glm::mat4 viewmatrix = camera_state->viewmatrix;\n"
        "\tconst glm::mat4 projmatrix = camera_state->projmatrix;\n"
        "\tconst glm::vec3 cam_position = camera_state->cam_position;\n"
        "\tconst float tan_fovx = camera_state->tan_fovx;\n"
        "\tconst float tan_fovy = camera_state->tan_fovy;\n"
        "\tconst float focal_x = camera_state->focal_x;\n"
        "\tconst float focal_y = camera_state->focal_y;\n",
        "camera_state",
    )
    categories["camera_state"] += 1

    transformed = _replace_once(
        transformed,
        """\t\t\tint offset = atomicAdd(curr_offset, 1);
\t\t\tgaussian_keys_unsorted[offset] = key;
\t\t\tgaussian_values_unsorted[offset] = idx_vec;""",
        """\t\t\tint offset = atomicAdd(curr_offset, 1);
\t\t\tif (offset < capacity)
\t\t\t{
\t\t\t\tgaussian_keys_unsorted[offset] = key;
\t\t\t\tgaussian_values_unsorted[offset] = idx_vec;
\t\t\t}
\t\t\telse
\t\t\t{
\t\t\t\tatomicAdd(overflow, 1);
\t\t\t}""",
        "bounded_emission",
    )
    categories["bounded_emission"] += 1
    transformed = _replace_once(
        transformed,
        """\t\t\t\tif (valid)
\t\t\t\t{
\t\t\t\t\tgaussian_keys_unsorted[my_offset + count] = key;
\t\t\t\t\tgaussian_values_unsorted[my_offset + count] = idx;
\t\t\t\t}""",
        """\t\t\t\tif (valid)
\t\t\t\t{
\t\t\t\t\tint output_offset = my_offset + count;
\t\t\t\t\tif (output_offset < capacity)
\t\t\t\t\t{
\t\t\t\t\t\tgaussian_keys_unsorted[output_offset] = key;
\t\t\t\t\t\tgaussian_values_unsorted[output_offset] = idx;
\t\t\t\t\t}
\t\t\t\t\telse
\t\t\t\t\t{
\t\t\t\t\t\tatomicAdd(overflow, 1);
\t\t\t\t\t}
\t\t\t\t}""",
        "bounded_emission",
    )
    categories["bounded_emission"] += 1
    transformed = _replace_once(
        transformed,
        "\t\tauto color = computeColorFromSH(idx_vec, p_orig, cam_position, (const shs_deg3_t*)shs);",
        "\t\tauto color = computeColorFromDegreeZero(idx_vec, colors);",
        "degree_zero_color",
    )
    categories["degree_zero_color"] += 1

    entry_start = transformed.find("void preprocess(int P,")
    entry_end = transformed.find("\n} // namepace flashgs", entry_start)
    if entry_start < 0 or entry_end < 0:
        raise RuntimeError("Cannot locate the exact upstream preprocess host entrypoint.")
    upstream_entry = transformed[entry_start:entry_end]
    if "preprocessCUDA<<<" not in upstream_entry or "getProjectionMatrix" not in upstream_entry:
        raise RuntimeError("Unexpected pinned FlashGS preprocess host entrypoint shape.")
    generated_entry = """void preprocess_degree0_fixed(int P,
\tconst glm::vec3* positions, const glm::vec3* colors,
\tconst float* opacities, cov3d_t* cov3Ds,
\tint width, int height, int block_x, int block_y,
\tconst upstream_camera_state_t* camera_state,
\tfloat2* points_xy, float4* rgb_depth, float4* conic_opacity,
\tuint64_t* gaussian_keys_unsorted, uint32_t* gaussian_values_unsorted,
\tint capacity, int* curr_offset, int* overflow, cudaStream_t stream)
{
\tdim3 grid((width + block_x - 1) / block_x, (height + block_y - 1) / block_y, 1);

\tpreprocessCUDA<<<(P + 127) / 128, dim3(8, 4, 4), 0, stream>>>(
\t\tP,
\t\tpositions,
\t\topacities,
\t\tcolors,
\t\tcamera_state,
\t\twidth, height,
\t\tblock_x, block_y,
\t\tpoints_xy,
\t\tcov3Ds,
\t\trgb_depth,
\t\tconic_opacity,
\t\tcurr_offset,
\t\tgaussian_keys_unsorted,
\t\tgaussian_values_unsorted,
\t\tcapacity,
\t\toverflow,
\t\tgrid);
}
"""
    transformed = transformed[:entry_start] + generated_entry + transformed[entry_end:]
    categories["host_entrypoint"] = 1

    after_functions = function_hashes(transformed)
    if after_functions != before_functions:
        changed = sorted(
            name
            for name in before_functions
            if before_functions[name] != after_functions.get(name)
        )
        raise RuntimeError(
            "Equation-bearing upstream preprocess functions changed during glue "
            f"generation: {changed}."
        )
    unified_diff = list(
        difflib.unified_diff(
            source.splitlines(keepends=True),
            transformed.splitlines(keepends=True),
            fromfile="upstream/csrc/cuda_rasterizer/preprocess.cu",
            tofile="generated/csrc/cuda_rasterizer/preprocess.cu",
        )
    )
    added_lines = sum(
        line.startswith("+") and not line.startswith("+++")
        for line in unified_diff
    )
    removed_lines = sum(
        line.startswith("-") and not line.startswith("---")
        for line in unified_diff
    )
    hunk_count = sum(line.startswith("@@") for line in unified_diff)
    generated_hash = sha256_bytes(transformed.encode("utf-8"))
    diff_audit = {
        "added_lines": added_lines,
        "removed_lines": removed_lines,
        "hunks": hunk_count,
        "unified_diff_sha256": sha256_bytes(
            "".join(unified_diff).encode("utf-8")
        ),
    }
    if generated_hash != PINNED_GENERATED_PREPROCESS_SHA256:
        raise RuntimeError(
            "Generated upstream-faithful preprocess source differs from the "
            "independently reviewed hash: "
            f"{generated_hash} != {PINNED_GENERATED_PREPROCESS_SHA256}."
        )
    if diff_audit != PINNED_PREPROCESS_DIFF:
        raise RuntimeError(
            "Generated upstream-faithful preprocess diff differs from the "
            f"independently reviewed contract: {diff_audit} != "
            f"{PINNED_PREPROCESS_DIFF}."
        )
    audit = {
        "schema_version": "upstream-faithful-flashgs-transform-audit-v1",
        "upstream_commit": FLASHGS_UPSTREAM_COMMIT,
        "source_sha256": source_hash,
        "generated_sha256": generated_hash,
        "categories": categories,
        "diff": diff_audit,
        "preserved_function_sha256": before_functions,
        "exact_upstream_translation_units": [
            "csrc/cuda_rasterizer/sort.cu",
            "csrc/cuda_rasterizer/render.cu",
        ],
        "generated_translation_units": [
            "csrc/cuda_rasterizer/preprocess.cu",
        ],
        "pass": True,
    }
    return SourceTransform(source=transformed, audit=audit)


def write_generated_preprocess(
    source_root: Path,
    output_path: Path,
    audit_path: Path,
) -> dict[str, Any]:
    source = (source_root / "csrc/cuda_rasterizer/preprocess.cu").read_text(
        encoding="utf-8"
    )
    result = transform_preprocess_source(source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.source, encoding="utf-8")
    unified_diff = "".join(
        difflib.unified_diff(
            source.splitlines(keepends=True),
            result.source.splitlines(keepends=True),
            fromfile="upstream/csrc/cuda_rasterizer/preprocess.cu",
            tofile="generated/csrc/cuda_rasterizer/preprocess.cu",
        )
    )
    diff_path = output_path.with_name("preprocess.patch")
    diff_path.write_text(unified_diff, encoding="utf-8")
    audit = {
        **result.audit,
        "unified_diff_artifact": {
            "path": str(diff_path.resolve()),
            "sha256": sha256_bytes(diff_path.read_bytes()),
            "bytes": diff_path.stat().st_size,
        },
    }
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit
