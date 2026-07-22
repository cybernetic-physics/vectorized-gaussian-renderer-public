import json
import math
from pathlib import Path

import pytest

from benchmarks.audit_flashgs_paper_reproduction import (
    CASES,
    EXPECTED_INPUT_SHA256,
    FLASHGS_COMMIT,
    FREIZA_CONTROL_GPU_UUID,
    VANILLA_COMMIT,
    VANILLA_RASTER_COMMIT,
    audit,
    compare_ppm_samples,
    sha256,
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def refresh_checksums(root: Path) -> None:
    results = sorted((root / "results").glob("*.json"))
    (root / "provenance" / "result-sha256.txt").write_text(
        "".join(f"{sha256(path)}  {path}\n" for path in results),
        encoding="utf-8",
    )


def create_fixture(root: Path) -> None:
    results = root / "results"
    provenance = root / "provenance"
    results.mkdir(parents=True)
    provenance.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    for case, contract in CASES.items():
        backend_fps: dict[str, float] = {}
        for backend, milliseconds in (("vanilla", 10.0), ("flash", 2.0)):
            fps = 1000.0 / milliseconds
            backend_fps[backend] = fps
            if contract["exact_public_checkpoint"]:
                scene_directory = contract["scene_directory"]
                model = (
                    f"/fixture/standard/{scene_directory}/point_cloud/"
                    "iteration_30000/point_cloud.ply"
                )
                cameras = f"/fixture/standard/{scene_directory}/cameras.json"
            else:
                model = f"/fixture/{contract['model_suffix']}"
                cameras = f"/fixture/{contract['cameras_suffix']}"
            write_json(
                results / f"{case}-{backend}.json",
                {
                    "schema_version": 1,
                    "scene": case,
                    "backend": backend,
                    "resolution": contract["resolution"],
                    "camera_count": 1,
                    "camera_stride": 1,
                    "warmup_per_camera": 1,
                    "repeats_per_camera": 10,
                    "vertex_count": 100,
                    "mean_frame_fps": fps,
                    "aggregate_throughput_fps": fps,
                    "minimum_frame_fps": fps,
                    "median_frame_fps": fps,
                    "gpu": "NVIDIA GeForce RTX 3090",
                    "gpu_capability": [8, 6],
                    "hostname": "freiza-1",
                    "cuda_visible_devices": "1",
                    "model": model,
                    "cameras": cameras,
                    "frames": [
                        {
                            "index": 0,
                            "camera_id": 0,
                            "image_name": "fixture",
                            "width": contract["resolution"][0],
                            "height": contract["resolution"][1],
                            "milliseconds": milliseconds,
                            "fps": fps,
                        }
                    ],
                },
            )
        write_json(
            results / f"{case}-pixel-comparison.json",
            {
                "reference": f"/fixture/samples/{case}-vanilla.ppm",
                "candidate": f"/fixture/samples/{case}-flash.ppm",
                "shape": [
                    contract["resolution"][1],
                    contract["resolution"][0],
                    3,
                ],
                "mean_absolute_error_u8": 1.0,
                "root_mean_square_error_u8": 1.0,
                "maximum_absolute_error_u8": 1,
                "psnr_db": 20.0 * math.log10(255.0),
                "exact_sample_fraction": 0.0,
            },
        )
        rows.append(
            {
                "case": case,
                "resolution": contract["resolution"],
                "camera_count": 1,
                "vanilla_fps": backend_fps["vanilla"],
                "flash_fps": backend_fps["flash"],
                "speedup": backend_fps["flash"] / backend_fps["vanilla"],
            }
        )
    exact_rows = [
        row for row in rows if CASES[str(row["case"])]["exact_public_checkpoint"]
    ]
    for name, selected in (
        ("exact-four-summary.json", exact_rows),
        ("public-seven-summary.json", rows),
    ):
        write_json(
            results / name,
            {
                "case_count": len(selected),
                "arithmetic_mean_speedup": sum(
                    float(row["speedup"]) for row in selected
                )
                / len(selected),
                "cases": selected,
            },
        )
    (provenance / "source-commits.tsv").write_text(
        "".join(
            (
                f"FlashGS\t{FLASHGS_COMMIT}\n",
                f"gaussian-splatting\t{VANILLA_COMMIT}\n",
                f"diff-gaussian-rasterization\t{VANILLA_RASTER_COMMIT}\n",
            )
        ),
        encoding="utf-8",
    )
    (provenance / "machine-and-toolchain.txt").write_text(
        f"1, NVIDIA GeForce RTX 3090, {FREIZA_CONTROL_GPU_UUID}\n",
        encoding="utf-8",
    )
    (provenance / "input-sha256.txt").write_text(
        "".join(
            f"{digest}  /fixture/{suffix}\n"
            for suffix, digest in EXPECTED_INPUT_SHA256.items()
        ),
        encoding="utf-8",
    )
    refresh_checksums(root)


def test_audit_recomputes_complete_fixture(tmp_path: Path) -> None:
    create_fixture(tmp_path)
    result = audit(tmp_path, verify_samples=False)
    assert result["pass"] is True
    assert result["failures"] == []
    assert result["exact_public_four"]["case_count"] == 4
    assert result["exact_public_four"]["arithmetic_mean_speedup"] == 5.0
    assert result["public_substitute_seven"]["case_count"] == 7
    assert result["hardware_scope"]["gpu_uuid"] == FREIZA_CONTROL_GPU_UUID


@pytest.mark.parametrize(
    ("mutation", "failure"),
    (
        ("frame-fps", "frame timing is invalid"),
        ("stored-mean", "timing aggregates differ from raw frames"),
        ("summary-row", "summary aggregate differs"),
        ("gpu-uuid", "control GPU UUID"),
        ("frame-sequence", "backend inputs or frame sequence differ"),
    ),
)
def test_audit_fails_closed_on_derived_evidence_mutation(
    tmp_path: Path,
    mutation: str,
    failure: str,
) -> None:
    create_fixture(tmp_path)
    if mutation in {"frame-fps", "stored-mean", "frame-sequence"}:
        path = tmp_path / "results" / "truck_4k-flash.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if mutation == "frame-fps":
            payload["frames"][0]["fps"] = 499.0
        elif mutation == "frame-sequence":
            payload["frames"][0]["image_name"] = "different"
        else:
            payload["mean_frame_fps"] = 499.0
        write_json(path, payload)
        refresh_checksums(tmp_path)
    elif mutation == "summary-row":
        path = tmp_path / "results" / "exact-four-summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["cases"][0]["speedup"] = 4.0
        write_json(path, payload)
        refresh_checksums(tmp_path)
    else:
        (tmp_path / "provenance" / "machine-and-toolchain.txt").write_text(
            "1, NVIDIA GeForce RTX 3090, GPU-wrong\n",
            encoding="utf-8",
        )
    result = audit(tmp_path, verify_samples=False)
    assert result["pass"] is False
    assert any(failure in item for item in result["failures"])


def write_ppm(path: Path, width: int, height: int, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode() + payload)


def test_compare_ppm_samples_recomputes_metrics_and_hashes(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.ppm"
    candidate = tmp_path / "candidate.ppm"
    write_ppm(reference, 2, 1, bytes((0, 1, 2, 3, 4, 5)))
    write_ppm(candidate, 2, 1, bytes((1, 1, 2, 3, 6, 5)))
    result = compare_ppm_samples(
        reference, candidate, expected_resolution=[2, 1]
    )
    assert result["shape"] == [1, 2, 3]
    assert result["mean_absolute_error_u8"] == pytest.approx(0.5)
    assert result["root_mean_square_error_u8"] == pytest.approx(
        math.sqrt(5.0 / 6.0)
    )
    assert result["maximum_absolute_error_u8"] == 2
    assert result["exact_sample_fraction"] == pytest.approx(4.0 / 6.0)
    assert len(result["reference_payload_sha256"]) == 64
    assert len(result["candidate_payload_sha256"]) == 64
    assert len(result["reference_file_sha256"]) == 64
    assert len(result["candidate_file_sha256"]) == 64


def test_audit_rejects_nonfinite_fidelity(tmp_path: Path) -> None:
    create_fixture(tmp_path)
    path = tmp_path / "results" / "truck_4k-pixel-comparison.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["psnr_db"] = float("nan")
    write_json(path, payload)
    refresh_checksums(tmp_path)
    result = audit(tmp_path, verify_samples=False)
    assert result["pass"] is False
    assert any("pixel parity failed" in item for item in result["failures"])


def test_audit_rejects_ppm_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for contract in CASES.values():
        monkeypatch.setitem(contract, "resolution", [1, 1])
    create_fixture(tmp_path)
    for case in CASES:
        write_ppm(
            tmp_path / "samples" / f"{case}-vanilla.ppm",
            1,
            1,
            bytes((0, 0, 0)),
        )
        write_ppm(
            tmp_path / "samples" / f"{case}-flash.ppm",
            1,
            1,
            bytes((1, 1, 1)),
        )
    assert audit(tmp_path)["pass"] is True

    write_ppm(
        tmp_path / "samples" / "truck_4k-flash.ppm",
        1,
        1,
        bytes((2, 2, 2)),
    )
    result = audit(tmp_path)
    assert result["pass"] is False
    assert any("sample fidelity differs" in item for item in result["failures"])
