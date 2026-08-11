from __future__ import annotations

import sys
from pathlib import Path

import pytest

PHASE_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PHASE_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_p4_b0_variable_prefill_repair_validation as validation  # noqa: E402


def _case_result(case_id: str) -> dict:
    labels = [row["label"] for row in validation.base.JOBS[case_id]["cohorts"]]
    observations = []
    if case_id == "r5cot-to-r8":
        observations.append({"cohort": "r5cot-last"})
    observations.append(
        {
            "metadata": {
                "capture_cohort_arm": True,
                "pure_decode": False,
                "decode_req_ids": [],
                "next_action_id": validation.base.ACTION_ID,
            },
            "proposal_called": True,
            "draft_output_shape": [16, 4],
            "draft_step0_query_width": None,
            "draft_step0_num_tokens": 2116,
            "draft_step0_batch_size": 16,
            "draft_step0_runtime_mode": "NONE",
            "draft_chain_runtime_mode": "PIECEWISE",
            "validated_evidence": {
                "draft_step0_query_width": None,
                "draft_step0_num_tokens": 2116,
                "draft_step0_batch_size": 16,
                "produced_draft_width": 4,
                "draft_dispatched": True,
            },
        }
    )
    return {
        "status": "completed_without_invariant_failure",
        "active_stage_at_exit": "complete",
        "completed_cohorts": labels,
        "primary_exception": None,
        "shutdown_exception": None,
        "gpu": validation.base.JOBS[case_id]["gpu"],
        "resources": {"shared_target_kv_block_capacity": 24527},
        "instrumentation": {"armed_prefill_evidence": observations},
        "recorder": {"closed": True, "event_count": 10},
        "trace": {"present": True, "record_count": 12},
    }


@pytest.mark.parametrize("case_id", tuple(validation.base.JOBS))
def test_case_result_accepts_exact_repaired_r8(case_id: str) -> None:
    summary = validation.validate_case_result(case_id, _case_result(case_id))
    assert summary["case_id"] == case_id
    assert summary["r8_variable_prefill_evidence"]["draft_step0_num_tokens"] == 2116


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("draft_step0_query_width", 132, "variable-width R8"),
        ("draft_step0_num_tokens", 2112, "variable-width R8"),
        ("draft_step0_batch_size", 15, "variable-width R8"),
        ("draft_output_shape", [15, 4], "variable-width R8"),
    ],
)
def test_case_result_rejects_drift(field: str, value, match: str) -> None:
    result = _case_result("isolated-r8")
    result["instrumentation"]["armed_prefill_evidence"][-1][field] = value
    with pytest.raises(validation.VariablePrefillRepairValidationError, match=match):
        validation.validate_case_result("isolated-r8", result)


def test_authorization_is_source_bound_and_create_only() -> None:
    validation._configure_base()
    authorization = validation.base._load_json(validation.AUTHORIZATION_PATH)
    validation.base.validate_authorization(
        authorization,
        authorization_path=validation.AUTHORIZATION_PATH.resolve(),
        output_dir=validation.OUTPUT_DIR.resolve(),
        require_output_absent=True,
    )
