from __future__ import annotations

import json

import pytest

from isaacsim_gaussian_renderer.benchmark_manifest import sha256_json
from isaacsim_gaussian_renderer.evaluation import matched_artifacts
from isaacsim_gaussian_renderer.evaluation.matched_artifacts import (
    GSPLAT_BOUNDARY_PROBE_EXPECTED_RADII,
    GSPLAT_BUILD_ATTESTATION_SCHEMA,
    GSPLAT_SUPPORT_PROBE_EXPECTED_RADII,
    MATCHED_GAUSSIAN_SUPPORT_SIGMA,
    MATCHED_PROJECTION_RULES,
    PINNED_GSPLAT_COMMIT,
    artifact_record,
    load_verified_gsplat_build_attestation,
)


def test_build_attestation_binds_source_build_graph_and_native_binary(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "gsplat"
    header = source / "gsplat/cuda/include/Common.h"
    header.parent.mkdir(parents=True)
    header.write_text("#define GAUSSIAN_EXTEND 3.33f\n", encoding="utf-8")
    relative = str(header.relative_to(source))
    source_inputs = {relative: artifact_record(header)}
    normalized_inputs = {
        relative: {
            "bytes": source_inputs[relative]["bytes"],
            "sha256": source_inputs[relative]["sha256"],
        }
    }

    build = tmp_path / "torch-extensions/gsplat_cuda"
    build.mkdir(parents=True)
    build_ninja = build / "build.ninja"
    build_parameters = build / "build_params.json"
    native = build / "gsplat_cuda.so"
    build_ninja.write_text("rule compile\n", encoding="utf-8")
    build_parameters.write_text("{}\n", encoding="utf-8")
    native.write_bytes(b"fresh-native-binary")
    probe = {
        "pass": True,
        "support_sigma": MATCHED_GAUSSIAN_SUPPORT_SIGMA,
        "expected_radii": GSPLAT_SUPPORT_PROBE_EXPECTED_RADII,
        "observed_radii": GSPLAT_SUPPORT_PROBE_EXPECTED_RADII,
        "boundary_expected_radii": GSPLAT_BOUNDARY_PROBE_EXPECTED_RADII,
        "boundary_observed_radii": GSPLAT_BOUNDARY_PROBE_EXPECTED_RADII,
    }
    payload = {
        "schema_version": GSPLAT_BUILD_ATTESTATION_SCHEMA,
        "gsplat_commit": PINNED_GSPLAT_COMMIT,
        "gsplat_source_root": str(source),
        "source_inputs": source_inputs,
        "source_input_tree_sha256": sha256_json(normalized_inputs),
        "support_contract": {
            "macro": "GAUSSIAN_EXTEND",
            "value": MATCHED_GAUSSIAN_SUPPORT_SIGMA,
            "header_artifact": artifact_record(header),
        },
        "projection_contract": MATCHED_PROJECTION_RULES,
        "build_directory": str(build),
        "build_ninja": artifact_record(build_ninja),
        "build_parameters": artifact_record(build_parameters),
        "native_extension": artifact_record(native),
        "behavior_probe": probe,
    }
    attestation = tmp_path / "gsplat-build-attestation.json"
    attestation.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        matched_artifacts.subprocess,
        "check_output",
        lambda *args, **kwargs: f"{PINNED_GSPLAT_COMMIT}\n",
    )

    verified = load_verified_gsplat_build_attestation(
        attestation,
        gsplat_source=source,
    )
    assert verified["native_extension"]["sha256"] == artifact_record(native)[
        "sha256"
    ]

    native.write_bytes(b"stale-or-tampered-native-binary")
    with pytest.raises(ValueError, match="native_extension content differs"):
        load_verified_gsplat_build_attestation(
            attestation,
            gsplat_source=source,
        )
