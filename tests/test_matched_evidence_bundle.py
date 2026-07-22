import json
import os
import shutil
from pathlib import Path

import pytest

from isaacsim_gaussian_renderer.evaluation import evidence_bundle
from isaacsim_gaussian_renderer.benchmark_manifest import file_sha256, sha256_json
from isaacsim_gaussian_renderer.evaluation.evidence_bundle import (
    CAPACITY_AMENDMENT_ARTIFACTS,
    CAPACITY_AMENDMENT_SCHEMA,
    CLAIM_LEDGER_SCHEMA,
    EVIDENCE_BUNDLE_SCHEMA,
    FLASHGS_ADAPTER_ATTESTATION_SCHEMA,
    FLASHGS_MATCHED_PORT_CLASSIFICATION,
    MATCHED_BATCHES,
    PUBLICATION_INVENTORY_SCHEMA,
    SEMANTIC_VALIDATION_SCHEMA,
    _matched_required_paths,
    build_evidence_bundle,
    resolve_artifact_record,
    verify_deterministic_archive,
    verify_evidence_bundle,
    write_deterministic_archive,
    write_publication_inventory,
)
from isaacsim_gaussian_renderer.evaluation.matched_artifacts import artifact_record


SCENE_SHA256 = "e" * 64
SCENE_BYTES = 1_203_883_212
CAPACITY_AMENDMENT_NAMES = tuple(CAPACITY_AMENDMENT_ARTIFACTS)


@pytest.fixture(autouse=True)
def _fixture_capacity_amendment_identities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evidence_bundle,
        "CAPACITY_AMENDMENT_ARTIFACTS",
        {name: {"bytes": 0, "sha256": "0" * 64} for name in CAPACITY_AMENDMENT_NAMES},
    )


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def semantic_result(source_root: Path, _summary_path: Path) -> dict[str, object]:
    paths = {
        "source_manifest": source_root / "provenance/source-manifest.json",
        "summary": source_root / "summary.json",
        "validator": (source_root / "publication/validator/summarize_flashgs_matched.py"),
    }
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    return {
        "batches": list(MATCHED_BATCHES),
        "pass": True,
        "recomputed_summary_sha256": sha256_json(summary),
        "schema_version": SEMANTIC_VALIDATION_SCHEMA,
        **{
            field: {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for field, path in paths.items()
        },
    }


def _json_payload(relative: str, roles: set[str]) -> dict[str, object]:
    if "canonical-summary" in roles:
        return {
            "schema_version": "flashgs-matched-summary-v4",
            "pass": True,
            "scientific_pass": True,
            "headline_eligible": True,
        }
    if "source-manifest" in roles:
        return {
            "schema_version": "renderer-source-manifest-v2",
            "dirty": False,
        }
    if "adapter-attestation" in roles:
        render_sha256 = "f" * 64
        return {
            "schema_version": FLASHGS_ADAPTER_ATTESTATION_SCHEMA,
            "upstream_commit": "cdfc4e4002318423eda356eed02df8e01fa32cb6",
            "upstream_clean": True,
            "baseline_classification": FLASHGS_MATCHED_PORT_CLASSIFICATION,
            "source_pairs": [
                {
                    "adapter_path": "src/isaacsim_gaussian_renderer/native/flashgs/render.cu",
                    "adapter_bytes": 1,
                    "adapter_sha256": render_sha256,
                }
            ],
            "correctness_repair_audit": {
                "render_source": {"bytes": 1, "sha256": render_sha256},
                "checks": {
                    "repaired_slot_0_predicate_exactly_once": True,
                    "repaired_slot_1_predicate_exactly_once": True,
                    "buggy_slot_0_predicate_absent": True,
                    "buggy_slot_1_predicate_absent": True,
                    "slot_0_load_guard_present": True,
                    "slot_1_load_guard_present": True,
                },
                "pass": True,
            },
            "pass": True,
        }
    if "build-attestation" in roles:
        return {"schema_version": "flashgs-matched-gsplat-build-v1"}
    if "code-freeze" in roles:
        return {"schema_version": "publication-code-freeze-v1", "pass": True}
    if "machine-provenance" in roles:
        return {
            "cuda_runtime": "12.1",
            "driver": "fixture",
            "gpu_name": "NVIDIA GeForce RTX 3090",
            "gpu_uuid": "GPU-fixture",
            "torch": "2.5.1",
        }
    if "verification-record" in roles:
        return {"schema_version": "publication-verification-v1", "pass": True}
    if "camera-contract" in roles:
        return {
            "schema_version": "camera-trajectory-v1",
            "trajectory_id": relative,
        }
    if "capacity-calibration" in roles:
        return {
            "schema_version": "flashgs-matched-capacity-calibration-v1",
            "mode": "capacity-calibration-only",
            "output_validation": {"pass": True},
            "pass": True,
        }
    if "flashgs-demand-survey" in roles:
        return {
            "schema_version": "flashgs-matched-flashgs-demand-survey-v1",
            "mode": "flashgs-demand-survey-only",
            "renderer": "flashgs",
            "timing_valid": False,
            "render_outputs_valid": False,
            "pass": True,
        }
    if "matrix-invocation" in roles:
        return {
            "schema_version": "flashgs-matched-matrix-invocation-v1",
            "argv": ["python", "benchmarks/run_flashgs_matched_matrix.py"],
        }
    if "occupancy-evidence" in roles:
        return {
            "schema_version": "flashgs-matched-node-occupancy-v2",
            "pass": True,
        }
    if "oracle-manifest" in roles:
        return {"schema_version": "flashgs-matched-gsplat-oracle-v4", "pass": True}
    if "fidelity-summary" in roles:
        return {"schema_version": "flashgs-matched-fidelity-v4", "pass": True}
    if "fidelity-json" in roles:
        return {"schema_version": "fidelity-report-v1", "pass": True}
    if "profile-control" in roles:
        return {
            "schema_version": "flashgs-matched-renderer-run-v4",
            "profile_control": True,
            "pass": True,
        }
    if "profile-summary" in roles:
        return {"schema_version": "flashgs-matched-profile-v3", "pass": True}
    if "profile-wrapper" in roles:
        return {
            "schema_version": "flashgs-profile-wrapper-evidence-v1",
            "command": ["nsys", "profile"],
            "pass": True,
        }
    if "raw-renderer-run" in roles:
        result: dict[str, object] = {
            "schema_version": "flashgs-matched-renderer-run-v4",
            "pass": True,
            "timing": {"gpu_batch_ms": {"mean": 1.0}},
        }
        if relative == "runs/custom/full/b1.json":
            result["scene"] = {
                "bytes": SCENE_BYTES,
                "path": "/home/freiza/datasets/home-scan-lod0.ply",
                "sha256": SCENE_SHA256,
            }
        return result
    if roles.intersection({"command-record", "summary-command"}):
        return {"command": ["python", relative.removesuffix(".command.json")]}
    return {"fixture": relative}


def create_publication_root(
    root: Path,
) -> tuple[dict[str, list[str]], list[dict[str, object]]]:
    roles = {relative: sorted(required_roles) for relative, required_roles in _matched_required_paths().items()}
    for relative, required_roles in roles.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        role_set = set(required_roles)
        if path.suffix == ".json":
            if not role_set.intersection({"capacity-amendment", "claim-ledger"}):
                write_json(path, _json_payload(relative, role_set))
        elif path.suffix == ".py":
            path.write_text("# frozen semantic validator fixture\n", encoding="utf-8")
        elif path.suffix in {".md", ".log", ".txt", ".csv"}:
            path.write_text(f"fixture,{relative}\n", encoding="utf-8")
        else:
            path.write_bytes(f"fixture:{relative}\n".encode())

    amendment_evidence: dict[str, dict[str, object]] = {}
    for name in CAPACITY_AMENDMENT_NAMES:
        relative = evidence_bundle.capacity_amendment_relative_path(name)
        record = artifact_record(root / relative)
        evidence_bundle.CAPACITY_AMENDMENT_ARTIFACTS[name] = {
            "bytes": record["bytes"],
            "sha256": record["sha256"],
        }
        amendment_evidence[name] = record
    write_json(
        root / "publication/capacity-amendment.json",
        {
            "schema_version": CAPACITY_AMENDMENT_SCHEMA,
            "pass": True,
            "failed_source_head": "025bdaf2258894f477074dbd955653abcbfb3f62",
            "failed_direct_protocol": {
                "logical_batch": 512,
                "physical_batch": 512,
                "capacity_headroom": 1.05,
                "requested_backend_workspace_bytes": 21_445_449_872,
                "timed_rows_completed": 0,
                "failure_stage": "custom-capacity-calibration-workspace-allocation",
            },
            "replacement_protocol": {
                "capacity_headroom": 1.05,
                "alternative_chunk_sizes_swept": False,
                "timing_observed_before_selection": False,
                "custom_physical_batches": {
                    "b1-b256": "direct",
                    "b512": 128,
                    "b1024": 128,
                },
                "custom_native_submissions": {
                    "b1-b256": 1,
                    "b512": 4,
                    "b1024": 8,
                },
            },
            "evidence": amendment_evidence,
        },
    )

    run_path = root / "runs/custom/full/b1.json"
    ledger_path = root / "publication/claim-ledger.json"
    write_json(
        ledger_path,
        {
            "schema_version": CLAIM_LEDGER_SCHEMA,
            "pass": True,
            "claims": [
                {
                    "artifact": artifact_record(run_path),
                    "article_location": "matched-results-table",
                    "claim_id": "full-b1-custom-ms",
                    "display_value": "1.00 ms",
                    "equation_contract": "matched EWA v4",
                    "hardware": "RTX 3090 GPU-fixture",
                    "json_fields": ["/timing/gpu_batch_ms/mean"],
                    "output_contract": "full",
                    "scene": "home-scan-lod0",
                }
            ],
        },
    )
    external = [
        {
            "bytes": SCENE_BYTES,
            "content_addressed_key": (f"datasets/home-scan-lod0/{SCENE_SHA256[:12]}/home-scan-lod0.ply"),
            "dependency_id": "home-scan-lod0",
            "license": "CC-BY-NC-4.0",
            "media_type": "application/octet-stream",
            "retrieval_manifest_url": ("https://example.test/home-scan-lod0.manifest.json"),
            "roles": ["scene-dataset"],
            "sha256": SCENE_SHA256,
        }
    ]
    return roles, external


def test_required_paths_match_real_capacity_protocol() -> None:
    required = _matched_required_paths()
    capacity_paths = [path for path, roles in required.items() if "capacity-calibration" in roles]
    survey_paths = [path for path, roles in required.items() if "flashgs-demand-survey" in roles]

    assert set(capacity_paths) == {f"capacity/custom/b{batch}.json" for batch in (1, 8, 32, 64, 128, 256, 512, 1024)}
    assert survey_paths == ["capacity/flashgs/b1024-demand-survey.json"]
    assert "logs/flashgs-b1024-demand-survey.log" in required
    assert "logs/flashgs-b1024-demand-survey.command.json" in required


def create_inventory(
    tmp_path: Path,
    root: Path,
    *,
    roles: dict[str, list[str]] | None = None,
    external: list[dict[str, object]] | None = None,
) -> Path:
    default_roles, default_external = create_publication_root(root)
    path = tmp_path / "publication-inventory.json"
    write_publication_inventory(
        input_root=root,
        bundle_name="freiza-rtx3090-matched-v1",
        roles_by_path=default_roles if roles is None else roles,
        external_dependencies=default_external if external is None else external,
        output_path=path,
    )
    return path


def build_fixture_bundle(
    *,
    source: Path,
    inventory: Path,
    output: Path,
) -> dict[str, object]:
    return build_evidence_bundle(
        input_root=source,
        inventory_path=inventory,
        output_root=output,
        semantic_validator=semantic_result,
    )


def test_bundle_is_relocatable_and_archive_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "source"
    inventory = create_inventory(tmp_path, source)
    first = tmp_path / "bundle-a"
    manifest = build_fixture_bundle(
        source=source,
        inventory=inventory,
        output=first,
    )
    assert manifest["schema_version"] == EVIDENCE_BUNDLE_SCHEMA
    assert all(not Path(item["object_path"]).is_absolute() for item in manifest["objects"])
    assert not any(item["sha256"] == SCENE_SHA256 for item in manifest["objects"])

    second = tmp_path / "bundle-b"
    second_manifest = build_fixture_bundle(
        source=source,
        inventory=inventory,
        output=second,
    )
    assert second_manifest["manifest_id"] == manifest["manifest_id"]

    ledger = json.loads((source / "publication/claim-ledger.json").read_text(encoding="utf-8"))
    recorded_run = ledger["claims"][0]["artifact"]
    shutil.rmtree(source)
    resolved = resolve_artifact_record(recorded_run, bundle_root=first)
    assert resolved["sha256"] == recorded_run["sha256"]
    assert Path(resolved["path"]).is_file()
    verify_evidence_bundle(first)

    archive_a = tmp_path / "bundle-a.tar"
    archive_b = tmp_path / "bundle-b.tar"
    record_a = write_deterministic_archive(first, archive_a)
    record_b = write_deterministic_archive(second, archive_b)
    assert record_a["sha256"] == record_b["sha256"]
    assert archive_a.read_bytes() == archive_b.read_bytes()
    assert verify_deterministic_archive(first, archive_a) == record_a


@pytest.mark.parametrize("mutation", ["tamper", "missing", "extra", "symlink"])
def test_verifier_fails_closed_on_object_tree_mutation(
    tmp_path: Path,
    mutation: str,
) -> None:
    source = tmp_path / "source"
    inventory = create_inventory(tmp_path, source)
    bundle = tmp_path / "bundle"
    manifest = build_fixture_bundle(
        source=source,
        inventory=inventory,
        output=bundle,
    )
    object_path = bundle / manifest["objects"][0]["object_path"]
    object_path.chmod(0o644)
    if mutation == "tamper":
        object_path.write_bytes(object_path.read_bytes() + b"tamper")
    elif mutation == "missing":
        object_path.unlink()
    elif mutation == "extra":
        (bundle / "extra.txt").write_text("extra", encoding="utf-8")
    else:
        replacement = tmp_path / "replacement"
        replacement.write_bytes(object_path.read_bytes())
        object_path.unlink()
        os.symlink(replacement, object_path)
    with pytest.raises((FileNotFoundError, ValueError)):
        verify_evidence_bundle(bundle)


@pytest.mark.parametrize(
    "missing_path",
    [
        "publication/capacity-amendment.json",
        "capacity/custom/b1024.json",
        "capacity/flashgs/b1024-demand-survey.json",
        "runs/flashgs/rgb/b512.json",
        "repeats/custom/full/b128-trial3.json",
        ("profiles/flashgs-matched-flashgs-full-b1024/profile-summary.json"),
        "oracle/b64.manifest.json",
        "runs/custom/rgb/b32.node-occupancy.json",
        "logs/fidelity-flashgs-full-b8.command.json",
    ],
)
def test_inventory_rejects_any_missing_frozen_matrix_evidence(
    tmp_path: Path,
    missing_path: str,
) -> None:
    source = tmp_path / "source"
    roles, external = create_publication_root(source)
    (source / missing_path).unlink()
    roles.pop(missing_path)
    with pytest.raises(ValueError):
        write_publication_inventory(
            input_root=source,
            bundle_name="freiza-rtx3090-matched-v1",
            roles_by_path=roles,
            external_dependencies=external,
            output_path=tmp_path / "incomplete.json",
        )


@pytest.mark.parametrize("mutation", ["schema", "pass", "timing-valid"])
def test_builder_rejects_invalid_flashgs_demand_survey(
    tmp_path: Path,
    mutation: str,
) -> None:
    source = tmp_path / "source"
    roles, external = create_publication_root(source)
    survey_path = source / "capacity/flashgs/b1024-demand-survey.json"
    survey = json.loads(survey_path.read_text(encoding="utf-8"))
    if mutation == "schema":
        survey["schema_version"] = "wrong"
    elif mutation == "pass":
        survey["pass"] = False
    else:
        survey["timing_valid"] = True
    write_json(survey_path, survey)
    inventory = tmp_path / "inventory.json"
    write_publication_inventory(
        input_root=source,
        bundle_name="gcp-l4-matched-v1",
        roles_by_path=roles,
        external_dependencies=external,
        output_path=inventory,
    )

    with pytest.raises(ValueError, match="FlashGS demand survey"):
        build_fixture_bundle(
            source=source,
            inventory=inventory,
            output=tmp_path / "bundle",
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("requested_backend_workspace_bytes", 1),
        ("failure_stage", "different"),
    ],
)
def test_builder_rejects_changed_capacity_amendment(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    source = tmp_path / "source"
    roles, external = create_publication_root(source)
    amendment_path = source / "publication/capacity-amendment.json"
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    amendment["failed_direct_protocol"][field] = replacement
    write_json(amendment_path, amendment)
    inventory = tmp_path / "inventory.json"
    write_publication_inventory(
        input_root=source,
        bundle_name="gcp-l4-matched-v1",
        roles_by_path=roles,
        external_dependencies=external,
        output_path=inventory,
    )

    with pytest.raises(ValueError, match="capacity amendment"):
        build_fixture_bundle(
            source=source,
            inventory=inventory,
            output=tmp_path / "bundle",
        )


@pytest.mark.parametrize("semantic_failure", ["exception", "false", "digest"])
def test_builder_requires_successful_semantic_revalidation(
    tmp_path: Path,
    semantic_failure: str,
) -> None:
    source = tmp_path / "source"
    inventory = create_inventory(tmp_path, source)
    output = tmp_path / "bundle"

    def failing_validator(root: Path, summary_path: Path) -> dict[str, object]:
        if semantic_failure == "exception":
            raise RuntimeError("canonical validator failed")
        result = semantic_result(root, summary_path)
        if semantic_failure == "false":
            result["pass"] = False
        else:
            result["recomputed_summary_sha256"] = "0" * 64
        return result

    with pytest.raises((RuntimeError, ValueError)):
        build_evidence_bundle(
            input_root=source,
            inventory_path=inventory,
            output_root=output,
            semantic_validator=failing_validator,
        )
    assert not output.exists()


@pytest.mark.parametrize("adapter_mutation", ["v1", "missing-classification"])
def test_builder_rejects_unreviewed_flashgs_baseline_label(
    tmp_path: Path,
    adapter_mutation: str,
) -> None:
    source = tmp_path / "source"
    roles, external = create_publication_root(source)
    adapter_path = source / "provenance/flashgs-adapter-attestation.json"
    adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
    if adapter_mutation == "v1":
        adapter["schema_version"] = "flashgs-adapter-attestation-v1"
    else:
        adapter.pop("baseline_classification")
    write_json(adapter_path, adapter)
    inventory = tmp_path / "inventory.json"
    write_publication_inventory(
        input_root=source,
        bundle_name="freiza-rtx3090-matched-v1",
        roles_by_path=roles,
        external_dependencies=external,
        output_path=inventory,
    )
    with pytest.raises(ValueError, match="adapter attestation"):
        build_fixture_bundle(
            source=source,
            inventory=inventory,
            output=tmp_path / "bundle",
        )


def test_builder_rejects_dangling_reference_and_unlisted_extra(tmp_path: Path) -> None:
    source = tmp_path / "source"
    roles, external = create_publication_root(source)
    run_path = source / "runs/custom/full/b1.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["dangling"] = {
        "bytes": 10,
        "path": str(tmp_path / "absent.bin"),
        "sha256": "d" * 64,
    }
    write_json(run_path, run)
    ledger_path = source / "publication/claim-ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["claims"][0]["artifact"] = artifact_record(run_path)
    write_json(ledger_path, ledger)
    inventory = tmp_path / "inventory.json"
    write_publication_inventory(
        input_root=source,
        bundle_name="freiza-rtx3090-matched-v1",
        roles_by_path=roles,
        external_dependencies=external,
        output_path=inventory,
    )
    with pytest.raises(FileNotFoundError, match="Missing artifact"):
        build_fixture_bundle(
            source=source,
            inventory=inventory,
            output=tmp_path / "bundle-dangling",
        )

    (source / "late.log").write_text("late", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from inventory"):
        build_fixture_bundle(
            source=source,
            inventory=inventory,
            output=tmp_path / "bundle-extra",
        )


def test_inventory_rejects_unsafe_secret_symlink_and_dataset_paths(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    roles, external = create_publication_root(source)

    unsafe_roles = dict(roles)
    unsafe_roles["../escape.json"] = unsafe_roles.pop("publication/machine-provenance.json")
    with pytest.raises(ValueError, match="Every publication file needs explicit roles"):
        write_publication_inventory(
            input_root=source,
            bundle_name="freiza-rtx3090-matched-v1",
            roles_by_path=unsafe_roles,
            external_dependencies=external,
            output_path=tmp_path / "unsafe.json",
        )

    secret = source / ".env"
    secret.write_text("API_KEY=abcdefghijklmnopqrstuvwxyz123456\n", encoding="utf-8")
    roles[".env"] = ["supporting-artifact"]
    with pytest.raises(ValueError, match="secret-bearing path"):
        write_publication_inventory(
            input_root=source,
            bundle_name="freiza-rtx3090-matched-v1",
            roles_by_path=roles,
            external_dependencies=external,
            output_path=tmp_path / "secret.json",
        )
    secret.unlink()
    roles.pop(".env")

    dataset = source / "scene.ply"
    dataset.write_bytes(b"ply\n")
    roles["scene.ply"] = ["supporting-artifact"]
    with pytest.raises(ValueError, match="Dataset PLY"):
        write_publication_inventory(
            input_root=source,
            bundle_name="freiza-rtx3090-matched-v1",
            roles_by_path=roles,
            external_dependencies=external,
            output_path=tmp_path / "dataset.json",
        )
    dataset.unlink()
    roles.pop("scene.ply")

    target = tmp_path / "target.log"
    target.write_text("target", encoding="utf-8")
    os.symlink(target, source / "linked.log")
    with pytest.raises(ValueError, match="Symlink"):
        write_publication_inventory(
            input_root=source,
            bundle_name="freiza-rtx3090-matched-v1",
            roles_by_path=roles,
            external_dependencies=external,
            output_path=tmp_path / "symlink.json",
        )


def test_inventory_and_manifest_are_hash_pinned(tmp_path: Path) -> None:
    source = tmp_path / "source"
    inventory = create_inventory(tmp_path, source)
    inventory_payload = json.loads(inventory.read_text(encoding="utf-8"))
    assert inventory_payload["schema_version"] == PUBLICATION_INVENTORY_SCHEMA
    assert all(item["sha256"] == file_sha256(source / item["path"]) for item in inventory_payload["artifacts"])

    bundle = tmp_path / "bundle"
    build_fixture_bundle(
        source=source,
        inventory=inventory,
        output=bundle,
    )
    manifest_path = bundle / "manifest.json"
    manifest_path.chmod(0o644)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["bundle_name"] = "tampered"
    write_json(manifest_path, payload)
    with pytest.raises(ValueError, match="manifest ID"):
        verify_evidence_bundle(bundle)


def test_secret_content_disguised_ply_empty_directories_and_root_symlinks_fail(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    roles, external = create_publication_root(source)

    (source / "innocent.log").write_text(
        "Authorization: Bearer abcdefghijklmnopqrstuvwxyz123456\n",
        encoding="utf-8",
    )
    roles["innocent.log"] = ["supporting-artifact"]
    with pytest.raises(ValueError, match="Secret material"):
        write_publication_inventory(
            input_root=source,
            bundle_name="freiza-rtx3090-matched-v1",
            roles_by_path=roles,
            external_dependencies=external,
            output_path=tmp_path / "secret-content.json",
        )
    (source / "innocent.log").unlink()
    roles.pop("innocent.log")

    (source / "scene.bin").write_bytes(b"ply\nformat binary_little_endian 1.0\n")
    roles["scene.bin"] = ["supporting-artifact"]
    with pytest.raises(ValueError, match="PLY payload"):
        write_publication_inventory(
            input_root=source,
            bundle_name="freiza-rtx3090-matched-v1",
            roles_by_path=roles,
            external_dependencies=external,
            output_path=tmp_path / "disguised-ply.json",
        )
    (source / "scene.bin").unlink()
    roles.pop("scene.bin")

    (source / "empty").mkdir()
    with pytest.raises(ValueError, match="empty or unexpected directories"):
        write_publication_inventory(
            input_root=source,
            bundle_name="freiza-rtx3090-matched-v1",
            roles_by_path=roles,
            external_dependencies=external,
            output_path=tmp_path / "empty-dir.json",
        )
    (source / "empty").rmdir()

    linked_root = tmp_path / "linked-root"
    os.symlink(source, linked_root)
    with pytest.raises(ValueError, match="root may not be a symlink"):
        write_publication_inventory(
            input_root=linked_root,
            bundle_name="freiza-rtx3090-matched-v1",
            roles_by_path=roles,
            external_dependencies=external,
            output_path=tmp_path / "linked-root.json",
        )


def test_archive_rejects_trailing_bytes_and_bundle_rejects_extra_directory(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    inventory = create_inventory(tmp_path, source)
    bundle = tmp_path / "bundle"
    build_fixture_bundle(
        source=source,
        inventory=inventory,
        output=bundle,
    )
    archive = tmp_path / "bundle.tar"
    write_deterministic_archive(bundle, archive)
    with archive.open("ab") as handle:
        handle.write(b"\0" * 10_240)
    with pytest.raises(ValueError, match="trailing or missing bytes"):
        verify_deterministic_archive(bundle, archive)

    (bundle / "empty-extra").mkdir()
    with pytest.raises(ValueError, match="empty or unexpected directories"):
        verify_evidence_bundle(bundle)


def test_builder_rejects_unsafe_path_even_for_external_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    roles, external = create_publication_root(source)
    run_path = source / "runs/custom/full/b1.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["scene"]["path"] = "../../secret/home-scan-lod0.ply"
    write_json(run_path, run)
    ledger_path = source / "publication/claim-ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["claims"][0]["artifact"] = artifact_record(run_path)
    write_json(ledger_path, ledger)
    inventory = tmp_path / "inventory.json"
    write_publication_inventory(
        input_root=source,
        bundle_name="freiza-rtx3090-matched-v1",
        roles_by_path=roles,
        external_dependencies=external,
        output_path=inventory,
    )
    with pytest.raises(ValueError, match="Unsafe recorded artifact path"):
        build_fixture_bundle(
            source=source,
            inventory=inventory,
            output=tmp_path / "bundle",
        )
