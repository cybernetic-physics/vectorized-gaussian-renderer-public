from __future__ import annotations

import pytest
import torch

from benchmarks.run_home_scan import require_passing_result as require_home_pass
from benchmarks.run_multi_scene import require_passing_result as require_multi_pass
from benchmarks.run_custom import output_contract, require_passing_result


def _valid_cpu_outputs() -> dict[str, torch.Tensor]:
    outputs = {
        "rgb": torch.zeros((1, 2, 3, 3), dtype=torch.float32),
        "depth": torch.full(
            (1, 2, 3, 1),
            float("inf"),
            dtype=torch.float32,
        ),
        "alpha": torch.zeros((1, 2, 3, 1), dtype=torch.float32),
        "semantic_id": torch.full(
            (1, 2, 3, 1),
            -1,
            dtype=torch.int64,
        ),
    }
    outputs["rgb"][0, 0, 0] = torch.tensor([0.2, 0.3, 0.4])
    outputs["depth"][0, 0, 0, 0] = 2.0
    outputs["alpha"][0, 0, 0, 0] = 0.5
    outputs["semantic_id"][0, 0, 0, 0] = 7
    return outputs


def _contract(outputs: dict[str, torch.Tensor]) -> dict[str, object]:
    return output_contract(
        outputs,
        expected_shape=(1, 2, 3),
        semantic_min_alpha=0.01,
        require_cuda=False,
    )


def test_run_custom_output_contract_records_complete_layout() -> None:
    contract = _contract(_valid_cpu_outputs())

    assert contract["valid"]
    assert contract["output_names_match"]
    assert contract["output_shapes_match"]
    assert contract["output_dtypes_match"]
    assert contract["outputs_contiguous"]
    assert contract["alpha_in_range"]
    assert contract["output_dtypes"] == {
        "rgb": "torch.float32",
        "depth": "torch.float32",
        "alpha": "torch.float32",
        "semantic_id": "torch.int64",
    }


def test_run_custom_output_contract_rejects_out_of_range_alpha() -> None:
    outputs = _valid_cpu_outputs()
    outputs["alpha"][0, 0, 0, 0] = 1.01

    contract = _contract(outputs)

    assert not contract["valid"]
    assert not contract["alpha_in_range"]


def test_run_custom_output_contract_rejects_wrong_dtype_and_layout() -> None:
    outputs = _valid_cpu_outputs()
    outputs["semantic_id"] = outputs["semantic_id"].to(torch.int32)
    outputs["rgb"] = torch.zeros(
        (1, 3, 2, 3),
        dtype=torch.float32,
    ).transpose(1, 2)

    contract = _contract(outputs)

    assert not contract["valid"]
    assert not contract["output_dtypes_match"]
    assert not contract["outputs_contiguous"]


def test_run_custom_output_contract_rejects_extra_aov() -> None:
    outputs = _valid_cpu_outputs()
    outputs["unexpected"] = torch.zeros((1,), dtype=torch.float32)

    contract = _contract(outputs)

    assert not contract["valid"]
    assert not contract["output_names_match"]


def test_run_custom_failed_result_exits_nonzero_after_persistence() -> None:
    require_passing_result({"pass": True})
    with pytest.raises(SystemExit) as error:
        require_passing_result({"pass": False})

    assert error.value.code == 1


@pytest.mark.parametrize("require_pass", [require_home_pass, require_multi_pass])
def test_other_custom_runners_fail_closed(require_pass) -> None:
    require_pass({"pass": True})
    with pytest.raises(SystemExit) as error:
        require_pass({"pass": False})

    assert error.value.code == 1
