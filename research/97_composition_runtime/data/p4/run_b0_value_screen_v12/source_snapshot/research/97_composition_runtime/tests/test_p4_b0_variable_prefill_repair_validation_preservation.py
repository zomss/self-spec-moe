from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

PHASE_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PHASE_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import preserve_p4_b0_variable_prefill_repair_validation_v1 as preservation  # noqa: E402


def _load_case(case_id: str) -> dict:
    return preservation._load_json(
        preservation.RUN_DIR / case_id / "case_result.json"
    )


@pytest.mark.parametrize("case_id", tuple(preservation.attempted.base.JOBS))
def test_immutable_gpu_case_passes_exact_r8_audit(case_id: str) -> None:
    summary = preservation.audit_case(case_id, _load_case(case_id))
    assert summary["exact_r8_observation_count"] == 1
    assert summary["exact_r8_observation"]["draft_step0_query_width"] is None


def test_exact_r8_audit_rejects_work_drift() -> None:
    result = copy.deepcopy(_load_case("isolated-r8"))
    observation = result["instrumentation"]["armed_prefill_evidence"][0]
    observation["draft_step0_num_tokens"] = 2112
    with pytest.raises(
        preservation.RepairValidationPreservationError,
        match="exactly one exact R8",
    ):
        preservation.audit_case("isolated-r8", result)


def test_parent_rejection_is_only_the_registered_count_guard() -> None:
    result = _load_case("r5cot-to-r8")
    with pytest.raises(
        preservation.attempted.VariablePrefillRepairValidationError,
        match=preservation.EXPECTED_PARENT_REJECTION,
    ):
        preservation.attempted.validate_case_result("r5cot-to-r8", result)


def test_preservation_closes_both_gpu_cases_without_authority_inflation() -> None:
    artifact = preservation.build_preservation()
    assert artifact["status"] == "gpu_cases_passed_parent_aggregate_rejected"
    assert artifact["disposition"]["gpu_repair_validation_passed"] is True
    assert artifact["disposition"]["parent_aggregate_passed"] is False
    assert artifact["disposition"]["value_screen_retry_authorized"] is False
    assert artifact["disposition"]["v11_authorized"] is False
