from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_custom_vectorization_ablation as ablation


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def timing(value: float) -> dict:
    return {
        "count": 100,
        "mean": value,
        "samples": [value] * 100,
    }


class DesignTests(unittest.TestCase):
    def test_counterbalanced_order_is_exact(self) -> None:
        self.assertEqual(
            ablation.run_order(),
            [
                ("full", 1, 1),
                ("full", 1, 128),
                ("rgb", 1, 128),
                ("rgb", 1, 1),
                ("full", 2, 128),
                ("full", 2, 1),
                ("rgb", 2, 1),
                ("rgb", 2, 128),
                ("full", 3, 1),
                ("full", 3, 128),
                ("rgb", 3, 128),
                ("rgb", 3, 1),
            ],
        )

    def test_commands_pin_twelve_fresh_process_cells(self) -> None:
        root = Path("/frozen/source")
        output = Path("/evidence/ablation")
        _capacity, timed, fidelity = ablation.build_commands(
            python=Path("/runtime/bin/python"),
            benchmark_root=root,
            output_root=output,
            scene_path=Path("/data/home.ply"),
            source_manifest=Path("/evidence/source.json"),
            trajectory=Path("/evidence/contracts/b128.json"),
            p128_capacity=Path("/evidence/matrix/capacity/b128.json"),
            oracle=Path("/evidence/matrix/oracle/b128.npz"),
            camera_bundle=Path("/evidence/matrix/oracle/b128.camera-bundle.json"),
            adapter_attestation=Path("/evidence/matrix/adapter.json"),
            expected_gpu_uuid="GPU-test",
            semantic_topology="spatial-octants-8",
            allow_nonheadline_gpu=True,
        )
        self.assertEqual(len(timed), 12)
        self.assertEqual(len(fidelity), 4)
        self.assertEqual(
            [item["run_id"] for item in timed],
            [ablation.run_id(*cell) for cell in ablation.run_order()],
        )
        self.assertEqual(len({item["result"] for item in timed}), 12)
        for item in timed:
            command = item["command"]
            self.assertIn("--measured-frames", command)
            self.assertEqual(command[command.index("--measured-frames") + 1], "100")
            self.assertEqual(
                command[command.index("--custom-max-physical-views") + 1],
                str(item["physical_batch"]),
            )
            capture_flag = (
                "--capture-last-output"
                if item["trial"] == 1
                else "--no-capture-last-output"
            )
            self.assertIn(capture_flag, command)


class ImmutableResumeTests(unittest.TestCase):
    def test_plan_resume_requires_identical_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "plan.json"
            plan = {"schema_version": ablation.PLAN_SCHEMA, "value": 1}
            first = ablation.write_or_verify_plan(path, plan)
            second = ablation.write_or_verify_plan(path, plan)
            self.assertTrue(ablation.same_artifact(first, second))
            with self.assertRaisesRegex(ablation.AblationError, "plan differs"):
                ablation.write_or_verify_plan(path, {**plan, "value": 2})

    def test_stage_resume_rejects_mutated_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "result.json"
            output.write_text("first\n")
            command = ["python", "runner.py", "--cell", "1"]
            receipt_path = root / "receipt.json"
            write_json(
                receipt_path,
                {
                    "schema_version": ablation.STAGE_RECEIPT_SCHEMA,
                    "stage_id": "cell-1",
                    "plan_sha256": "a" * 64,
                    "command_sha256": ablation.command_sha256(command),
                    "exit_code": 0,
                    "artifacts": {
                        "result": ablation.artifact_record(output),
                    },
                },
            )
            ablation.verify_stage_receipt(
                receipt_path,
                stage_id="cell-1",
                plan_sha256="a" * 64,
                command=command,
            )
            output.write_text("changed\n")
            with self.assertRaisesRegex(ablation.AblationError, "content differs"):
                ablation.verify_stage_receipt(
                    receipt_path,
                    stage_id="cell-1",
                    plan_sha256="a" * 64,
                    command=command,
                )

    def test_subcommand_file_is_the_bare_wrapper_input_list(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "timed-subcommands.json"
            commands = [["python", "run_flashgs_matched.py", "--cell", "1"]]
            ablation.write_or_verify_json(
                path,
                commands,
                label="timed subcommand plan",
            )
            self.assertEqual(json.loads(path.read_text()), commands)


class EvidenceTests(unittest.TestCase):
    def test_timing_mean_recomputes_all_samples(self) -> None:
        payload = {"timing": {"gpu_batch_ms": timing(3.0)}}
        self.assertEqual(
            ablation.timing_mean(payload, "gpu_batch_ms", label="test"),
            3.0,
        )
        payload["timing"]["gpu_batch_ms"]["mean"] = 2.0
        with self.assertRaisesRegex(ablation.AblationError, "does not match"):
            ablation.timing_mean(payload, "gpu_batch_ms", label="test")

    def _assembled_fixture(self, root: Path) -> tuple[dict, dict, dict, dict, dict]:
        p1_path = root / "capacity-p1.json"
        p128_path = root / "capacity-p128.json"
        p1_path.write_text("p1\n")
        p128_path.write_text("p128\n")
        p1_record = ablation.artifact_record(p1_path)
        p128_record = ablation.artifact_record(p128_path)
        timed = []
        run_results = {}
        run_receipts = {}
        fidelity_receipts = {}
        for contract, trial, physical in ablation.run_order():
            identifier = ablation.run_id(contract, trial, physical)
            run_path = root / "runs" / f"{identifier}.json"
            occupancy_path = root / "occupancy" / f"{identifier}.json"
            occupancy_path.parent.mkdir(parents=True, exist_ok=True)
            occupancy_path.write_text(identifier + "\n")
            occupancy = ablation.artifact_record(occupancy_path)
            gpu = 5.0 if physical == 1 else 10.0
            wall = 6.0 if physical == 1 else 12.0
            equation = {
                "precision": "float32",
                "support": 3.33,
                "output_dtypes": (
                    {"rgb": "torch.float32"}
                    if contract == "rgb"
                    else {
                        "rgb": "torch.float32",
                        "depth": "torch.float32",
                    }
                ),
            }
            result = {
                "equation_contract": equation,
                "environment": {"node_occupancy": occupancy},
                "timing": {
                    "gpu_batch_ms": timing(gpu),
                    "synchronized_wall_batch_ms": timing(wall),
                },
            }
            write_json(run_path, {"run_id": identifier})
            run_record = ablation.artifact_record(run_path)
            run_results[identifier] = result
            run_receipts[identifier] = {
                "artifacts": {"result": run_record}
            }
            timed.append(
                {
                    "run_id": identifier,
                    "contract": contract,
                    "trial": trial,
                    "physical_batch": physical,
                }
            )
            if trial == 1:
                fidelity_path = root / "fidelity" / f"{identifier}.json"
                write_json(fidelity_path, {"run_id": identifier, "pass": True})
                fidelity_receipts[identifier] = {
                    "artifacts": {
                        "result": ablation.artifact_record(fidelity_path)
                    }
                }
        plan = {
            "timed_runs": timed,
            "run_order": [item["run_id"] for item in timed],
            "inputs": {"p128_capacity": p128_record},
        }
        return plan, run_results, run_receipts, fidelity_receipts, p1_record

    def test_assembly_accepts_an_honest_slowdown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, results, receipts, fidelity, p1 = self._assembled_fixture(root)
            output = ablation.assemble_output(
                plan=plan,
                plan_record={"sha256": "a" * 64},
                context={},
                run_results=results,
                run_receipts=receipts,
                fidelity_receipts=fidelity,
                p1_capacity_payload={},
                p1_capacity_record=p1,
            )
            self.assertTrue(output["pass"])
            self.assertEqual(set(output), {"schema_version", "pass", "batch", "runs", "run_order", "ratios"})
            for contract in ablation.CONTRACTS:
                self.assertEqual(
                    [row["cuda_speedup_p128_over_p1"] for row in output["ratios"][contract]],
                    [0.5, 0.5, 0.5],
                )

    def test_assembly_rejects_reused_process_occupancy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, results, receipts, fidelity, p1 = self._assembled_fixture(root)
            first, second = plan["timed_runs"][:2]
            results[second["run_id"]]["environment"]["node_occupancy"] = results[
                first["run_id"]
            ]["environment"]["node_occupancy"]
            with self.assertRaisesRegex(ablation.AblationError, "Occupancy evidence was reused"):
                ablation.assemble_output(
                    plan=plan,
                    plan_record={"sha256": "a" * 64},
                    context={},
                    run_results=results,
                    run_receipts=receipts,
                    fidelity_receipts=fidelity,
                    p1_capacity_payload={},
                    p1_capacity_record=p1,
                )

    def test_matrix_binding_requires_reused_capacity_and_oracle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capacity_path = root / "capacity.json"
            oracle_path = root / "oracle-manifest.json"
            capacity_path.write_text("capacity\n")
            oracle_path.write_text("oracle\n")
            capacity = ablation.artifact_record(capacity_path)
            oracle = ablation.artifact_record(oracle_path)
            tables = {}
            for table_name, contract in (
                ("primary_full_sensor_dynamic_table", "full"),
                ("rgb_only_dynamic_table", "rgb"),
            ):
                run_path = root / f"{contract}-run.json"
                fidelity_path = root / f"{contract}-fidelity.json"
                write_json(
                    run_path,
                    {
                        "renderer": "custom",
                        "output_contract": contract,
                        "pass": True,
                        "capacity": {"calibration_artifact": capacity},
                    },
                )
                write_json(
                    fidelity_path,
                    {
                        "schema_version": ablation.FIDELITY_SCHEMA,
                        "pass": True,
                        "input_artifacts": {"oracle_manifest": oracle},
                    },
                )
                tables[table_name] = [
                    {
                        "batch": 128,
                        "artifacts": {
                            "custom_run": ablation.artifact_record(run_path),
                            "custom_fidelity": ablation.artifact_record(
                                fidelity_path
                            ),
                        },
                    }
                ]
            summary = root / "summary.json"
            write_json(
                summary,
                {
                    "schema_version": "flashgs-matched-summary-v4",
                    "pass": True,
                    "scientific_pass": True,
                    **tables,
                },
            )
            ablation.verify_matrix_bindings(
                summary_path=summary,
                p128_capacity_record=capacity,
                oracle_manifest_record=oracle,
            )
            wrong_path = root / "wrong-capacity.json"
            wrong_path.write_text("wrong\n")
            with self.assertRaisesRegex(ablation.AblationError, "not the matrix"):
                ablation.verify_matrix_bindings(
                    summary_path=summary,
                    p128_capacity_record=ablation.artifact_record(wrong_path),
                    oracle_manifest_record=oracle,
                )


if __name__ == "__main__":
    unittest.main()
