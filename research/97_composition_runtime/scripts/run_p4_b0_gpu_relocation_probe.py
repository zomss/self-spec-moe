#!/usr/bin/env python3
"""Run a fresh non-scored relocation probe on physical GPUs 0 and 1."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import run_p4_b0_r5cot_r8_diagnosis as base
import run_p4_b0_r5cot_r8_diagnosis_v2 as diagnosis_v2
import run_p4_b0_variable_prefill_repair_validation as repair

AUTHORIZATION_PATH = (
    base.PHASE_DIR / "data" / "p4" / "p4_b0_gpu_relocation_probe_authorization_v1.json"
)
OUTPUT_DIR = base.PHASE_DIR / "data" / "p4" / "run_b0_gpu_relocation_probe_v1"
EXPECTED_PACKAGE_ID = "p4-b0-gpu-relocation-probe-authorization-v1"
REQUIRED_SOURCE_PATHS = {
    "capture_manifest": (
        "research/97_composition_runtime/data/p4/run_b0_value_screen_v9/"
        "capture_manifest.json"
    ),
    "capture_plan": (
        "research/97_composition_runtime/data/p4/run_b0_value_screen_v9/"
        "plans/p4-b0-b1-p2-k4.json"
    ),
    "capture_runner": (
        "research/97_composition_runtime/scripts/run_p4_b0_value_screen.py"
    ),
    "diagnosis": (
        "research/97_composition_runtime/data/p4/"
        "run_b0_r5cot_r8_diagnosis_v2/diagnosis.json"
    ),
    "diagnosis_runner_base": (
        "research/97_composition_runtime/scripts/run_p4_b0_r5cot_r8_diagnosis.py"
    ),
    "diagnosis_runner_v2": (
        "research/97_composition_runtime/scripts/run_p4_b0_r5cot_r8_diagnosis_v2.py"
    ),
    "draft_proposer": "vllm/v1/spec_decode/llm_base_proposer.py",
    "failed_attempt": (
        "research/97_composition_runtime/data/p4/run_b0_value_screen_v9/failure.json"
    ),
    "gpu_model_runner": "vllm/v1/worker/gpu_model_runner.py",
    "gpu_worker": "vllm/v1/worker/gpu_worker.py",
    "koff_runtime": "vllm/v1/spec_decode/koff_runtime.py",
    "prompt_bundle": (
        "research/97_composition_runtime/data/p4/p4_b0_prompt_tokens.jsonl.gz"
    ),
    "prompt_manifest": (
        "research/97_composition_runtime/data/p4/p4_b0_prompt_manifest.json"
    ),
    "relocation_probe_runner": (
        "research/97_composition_runtime/scripts/run_p4_b0_gpu_relocation_probe.py"
    ),
    "relocation_probe_tests": (
        "research/97_composition_runtime/tests/test_p4_b0_gpu_relocation_probe.py"
    ),
    "repair_validation_runner": (
        "research/97_composition_runtime/scripts/"
        "run_p4_b0_variable_prefill_repair_validation.py"
    ),
    "runtime_cpu_tests": "tests/v1/spec_decode/test_koff_runtime.py",
    "scheduler": "vllm/v1/core/sched/scheduler.py",
    "scheduler_cpu_tests": "tests/v1/core/test_scheduler.py",
    "v10_authorization": (
        "research/97_composition_runtime/data/p4/p4_b0_run_authorization_v10.json"
    ),
    "v11_authorization": (
        "research/97_composition_runtime/data/p4/p4_b0_run_authorization_v11.json"
    ),
    "v11_capture_manifest": (
        "research/97_composition_runtime/data/p4/run_b0_value_screen_v10/"
        "capture_manifest.json"
    ),
    "v11_interruption": (
        "research/97_composition_runtime/data/p4/run_b0_value_screen_v10/failure.json"
    ),
    "v11_preservation_script": (
        "research/97_composition_runtime/scripts/"
        "preserve_p4_b0_value_screen_v11_attempt.py"
    ),
}


class RelocationProbeError(RuntimeError):
    """Raised when the relocation probe does not close exactly."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RelocationProbeError(message)


def _validate_v11_boundary(authorization: Mapping[str, Any]) -> None:
    manifest = base._load_json(
        base.REPO_ROOT / REQUIRED_SOURCE_PATHS["v11_capture_manifest"]
    )
    interruption = base._load_json(
        base.REPO_ROOT / REQUIRED_SOURCE_PATHS["v11_interruption"]
    )
    _require(
        manifest.get("counts", {}).get("complete_captures") == 48
        and manifest.get("counts", {}).get("empty_placeholders") == 1
        and manifest.get("invariants", {}).get("score_absent") is True,
        "V11 immutable capture boundary drifted",
    )
    _require(
        interruption.get("diagnostic", {}).get("classification")
        == "external_resource_reassignment"
        and interruption.get("disposition", {}).get("v11_consumed") is True
        and interruption.get("disposition", {}).get("scoring_allowed") is False,
        "V11 interruption disposition drifted",
    )
    _require(
        authorization.get("claims")
        == {
            "v11_interruption_preserved": True,
            "gpu0_gpu1_relocation_probe_authorized": True,
            "value_screen_execution_authorized": False,
            "performance_claim_allowed": False,
            "action_admission_allowed": False,
        },
        "relocation probe claims exceed the non-scored boundary",
    )
    _require(
        authorization.get("authorizations")
        == {
            "gpu0_isolated_r8_probe": True,
            "gpu1_r5cot_r8_transition_probe": True,
            "value_screen_execution": False,
            "value_screen_scoring": False,
            "p4a_engineering": False,
            "action_admission": False,
            "production_value_claim": False,
        },
        "relocation probe authority is inflated",
    )


def validate_case_result(case_id: str, result: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one relocation case and select its exact R8 observation."""
    expected_labels = [row["label"] for row in base.JOBS[case_id]["cohorts"]]
    _require(
        result.get("status") == "completed_without_invariant_failure"
        and result.get("active_stage_at_exit") == "complete"
        and result.get("completed_cohorts") == expected_labels,
        f"{case_id} did not complete its exact cohort sequence",
    )
    _require(
        result.get("primary_exception") is None
        and result.get("shutdown_exception") is None,
        f"{case_id} retained an execution or shutdown exception",
    )
    resources = result.get("resources")
    _require(isinstance(resources, Mapping), f"{case_id} has no resource evidence")
    _require(
        resources.get("shared_target_kv_block_capacity")
        >= base.MINIMUM_SHARED_KV_BLOCKS,
        f"{case_id} fell below the shared-KV floor",
    )
    instrumentation = result.get("instrumentation")
    _require(
        isinstance(instrumentation, Mapping),
        f"{case_id} has no repaired instrumentation",
    )
    observations = instrumentation.get("armed_prefill_evidence")
    expected_observation_count = 1 if case_id == "isolated-r8" else 9
    _require(
        isinstance(observations, list)
        and len(observations) == expected_observation_count,
        f"{case_id} armed-prefill evidence count drifted",
    )
    exact_r8 = [
        row
        for row in observations
        if row.get("draft_step0_query_width") is None
        and row.get("draft_step0_num_tokens") == 2116
        and row.get("draft_step0_batch_size") == 16
        and row.get("draft_output_shape") == [16, 4]
        and row.get("proposal_called") is True
        and row.get("validated_evidence", {}).get("produced_draft_width") == 4
        and row.get("validated_evidence", {}).get("draft_dispatched") is True
    ]
    _require(len(exact_r8) == 1, f"{case_id} lacks one exact R8 observation")
    recorder = result.get("recorder")
    trace = result.get("trace")
    _require(
        isinstance(recorder, Mapping)
        and recorder.get("closed") is True
        and recorder.get("event_count", 0) > 0,
        f"{case_id} did not close measured decode events",
    )
    _require(
        isinstance(trace, Mapping)
        and trace.get("present") is True
        and trace.get("record_count", 0) > 0,
        f"{case_id} did not preserve its passive trace",
    )
    return {
        "case_id": case_id,
        "gpu": result.get("gpu"),
        "completed_cohorts": expected_labels,
        "measured_event_count": recorder["event_count"],
        "trace_record_count": trace["record_count"],
        "shared_target_kv_block_capacity": resources["shared_target_kv_block_capacity"],
        "armed_prefill_observation_count": len(observations),
        "exact_r8_observation": exact_r8[0],
    }


def finalize_probe(output_dir: Path) -> None:
    """Emit a strict pass only after both physical-GPU cases close."""
    diagnosis = base._load_json(output_dir / "diagnosis.json")
    _require(
        diagnosis.get("classification", {}).get("conclusion")
        == "original_failure_not_reproduced",
        "two-case relocation diagnosis did not complete",
    )
    cases = [
        validate_case_result(
            case_id,
            base._load_json(output_dir / case_id / "case_result.json"),
        )
        for case_id in base.JOBS
    ]
    base._write_exclusive(
        output_dir / "probe_result.json",
        {
            "schema_version": 1,
            "artifact_id": "p4-b0-gpu-relocation-probe-v1",
            "status": "pass",
            "scored": False,
            "authorization_consumed": True,
            "authorization": base._reference(AUTHORIZATION_PATH),
            "cases": cases,
            "claims": {
                "gpu0_exact_r8_passed": True,
                "gpu1_r5cot_to_r8_passed": True,
                "shared_kv_floor_passed_on_both": True,
                "full_value_screen_authorized": False,
                "performance_claim_allowed": False,
            },
            "next_action": (
                "Prepare a separate source-bound single-GPU value-screen "
                "authorization; do not reuse probe output."
            ),
        },
    )


def _configure_base() -> None:
    original_validate = base.validate_authorization
    base.AUTHORIZATION_PATH = AUTHORIZATION_PATH
    base.OUTPUT_DIR = OUTPUT_DIR
    base.EXPECTED_PACKAGE_ID = EXPECTED_PACKAGE_ID
    base.REQUIRED_SOURCE_PATHS = REQUIRED_SOURCE_PATHS
    base._scheduler_snapshot = diagnosis_v2.scheduler_snapshot
    base._install_instrumentation = repair._install_repair_instrumentation
    base.__file__ = str(Path(__file__).resolve())

    def validate_authorization(*args: Any, **kwargs: Any) -> None:
        original_validate(*args, **kwargs)
        _validate_v11_boundary(args[0])

    base.validate_authorization = validate_authorization


def main() -> int:
    """Run the create-only, non-scored relocation probe."""
    _configure_base()
    child = "--child-case" in sys.argv
    prepare_only = "--prepare-only" in sys.argv
    result = base.main()
    if not child and not prepare_only:
        finalize_probe(OUTPUT_DIR.resolve())
    return result


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (base.P4TransitionDiagnosisError, RelocationProbeError) as exc:
        print(f"P4 GPU relocation probe refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
