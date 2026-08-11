#!/usr/bin/env python3
"""Validate repaired variable-width prefill evidence on GPUs 0 and 1."""

from __future__ import annotations

import dataclasses
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import run_p4_b0_r5cot_r8_diagnosis as base
import run_p4_b0_r5cot_r8_diagnosis_v2 as diagnosis_v2

AUTHORIZATION_PATH = (
    base.PHASE_DIR
    / "data"
    / "p4"
    / "p4_b0_variable_prefill_repair_validation_authorization_v1.json"
)
OUTPUT_DIR = (
    base.PHASE_DIR / "data" / "p4" / "run_b0_variable_prefill_repair_validation_v1"
)
EXPECTED_PACKAGE_ID = "p4-b0-variable-prefill-repair-validation-authorization-v1"
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
        "research/97_composition_runtime/scripts/"
        "run_p4_b0_r5cot_r8_diagnosis.py"
    ),
    "diagnosis_runner_v2": (
        "research/97_composition_runtime/scripts/"
        "run_p4_b0_r5cot_r8_diagnosis_v2.py"
    ),
    "draft_proposer": "vllm/v1/spec_decode/llm_base_proposer.py",
    "failed_attempt": (
        "research/97_composition_runtime/data/p4/run_b0_value_screen_v9/"
        "failure.json"
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
    "repair_validation_runner": (
        "research/97_composition_runtime/scripts/"
        "run_p4_b0_variable_prefill_repair_validation.py"
    ),
    "repair_validation_tests": (
        "research/97_composition_runtime/tests/"
        "test_p4_b0_variable_prefill_repair_validation.py"
    ),
    "runtime_cpu_tests": "tests/v1/spec_decode/test_koff_runtime.py",
    "scheduler": "vllm/v1/core/sched/scheduler.py",
    "scheduler_cpu_tests": "tests/v1/core/test_scheduler.py",
    "v10_authorization": (
        "research/97_composition_runtime/data/p4/"
        "p4_b0_run_authorization_v10.json"
    ),
}


class VariablePrefillRepairValidationError(RuntimeError):
    """Raised when repaired GPU evidence does not close exactly."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VariablePrefillRepairValidationError(message)


_ORIGINAL_INSTALL_INSTRUMENTATION = base._install_instrumentation


def _install_repair_instrumentation(
    scheduler: Any, worker: Any, state: dict[str, Any]
) -> None:
    """Extend the consumed diagnosis observer with repaired work facts."""
    import vllm.v1.worker.gpu_model_runner as gpu_model_runner

    _ORIGINAL_INSTALL_INSTRUMENTATION(scheduler, worker, state)
    original_make_evidence = gpu_model_runner.make_runner_evidence
    armed_prefill_evidence: list[dict[str, Any]] = []
    state["armed_prefill_evidence"] = armed_prefill_evidence

    def make_evidence_wrapper(*args: Any, **kwargs: Any) -> Any:
        evidence = original_make_evidence(*args, **kwargs)
        metadata = kwargs.get("metadata")
        if (
            dataclasses.is_dataclass(metadata)
            and metadata.capture_cohort_arm
            and not metadata.pure_decode
        ):
            output = kwargs.get("output")
            armed_prefill_evidence.append(
                {
                    "metadata": dataclasses.asdict(metadata),
                    "proposal_called": kwargs.get("proposal_called"),
                    "draft_output_shape": list(getattr(output, "shape", ())),
                    "draft_step0_query_width": kwargs.get(
                        "draft_step0_query_width"
                    ),
                    "draft_step0_num_tokens": kwargs.get("draft_step0_num_tokens"),
                    "draft_step0_batch_size": kwargs.get("draft_step0_batch_size"),
                    "draft_step0_runtime_mode": kwargs.get(
                        "draft_step0_runtime_mode"
                    ),
                    "draft_chain_runtime_mode": kwargs.get(
                        "draft_chain_runtime_mode"
                    ),
                    "validated_evidence": {
                        "draft_step0_query_width": (
                            evidence.draft_step0_query_width
                        ),
                        "draft_step0_num_tokens": evidence.draft_step0_num_tokens,
                        "draft_step0_batch_size": evidence.draft_step0_batch_size,
                        "produced_draft_width": evidence.produced_draft_width,
                        "draft_dispatched": evidence.draft_dispatched,
                    },
                }
            )
        return evidence

    gpu_model_runner.make_runner_evidence = make_evidence_wrapper


def validate_case_result(case_id: str, result: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one completed repair case and return its R8 proof."""
    expected_labels = [row["label"] for row in base.JOBS[case_id]["cohorts"]]
    _require(
        result.get("status") == "completed_without_invariant_failure",
        f"{case_id} did not complete after the repair",
    )
    _require(
        result.get("active_stage_at_exit") == "complete"
        and result.get("completed_cohorts") == expected_labels,
        f"{case_id} did not close its exact cohort sequence",
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
    expected_observation_count = 1 if case_id == "isolated-r8" else 2
    _require(
        isinstance(observations, list)
        and len(observations) == expected_observation_count,
        f"{case_id} armed-prefill evidence count drifted",
    )
    r8 = observations[-1]
    metadata = r8.get("metadata")
    validated = r8.get("validated_evidence")
    _require(
        isinstance(metadata, Mapping)
        and metadata.get("capture_cohort_arm") is True
        and metadata.get("pure_decode") is False
        and metadata.get("decode_req_ids") == []
        and metadata.get("next_action_id") == base.ACTION_ID,
        f"{case_id} R8 scheduler arm is malformed",
    )
    _require(
        r8.get("proposal_called") is True
        and r8.get("draft_output_shape") == [16, 4]
        and r8.get("draft_step0_query_width") is None
        and r8.get("draft_step0_num_tokens") == 2116
        and r8.get("draft_step0_batch_size") == 16
        and r8.get("draft_step0_runtime_mode") == "NONE"
        and r8.get("draft_chain_runtime_mode") == "PIECEWISE",
        f"{case_id} did not preserve the exact variable-width R8 dispatch",
    )
    _require(
        isinstance(validated, Mapping)
        and validated.get("draft_step0_query_width") is None
        and validated.get("draft_step0_num_tokens") == 2116
        and validated.get("draft_step0_batch_size") == 16
        and validated.get("produced_draft_width") == 4
        and validated.get("draft_dispatched") is True,
        f"{case_id} repaired runner evidence did not close",
    )
    recorder = result.get("recorder")
    _require(
        isinstance(recorder, Mapping)
        and recorder.get("closed") is True
        and recorder.get("event_count", 0) > 0,
        f"{case_id} did not close measured decode events",
    )
    trace = result.get("trace")
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
        "shared_target_kv_block_capacity": resources[
            "shared_target_kv_block_capacity"
        ],
        "r8_variable_prefill_evidence": r8,
    }


def finalize_validation(output_dir: Path) -> None:
    """Emit a strict aggregate only after both GPU cases pass."""
    diagnosis = base._load_json(output_dir / "diagnosis.json")
    _require(
        diagnosis.get("classification", {}).get("conclusion")
        == "original_failure_not_reproduced",
        "base two-case classification did not observe both repaired completions",
    )
    summaries = []
    for case_id in base.JOBS:
        result = base._load_json(output_dir / case_id / "case_result.json")
        summaries.append(validate_case_result(case_id, result))
    payload = {
        "schema_version": 1,
        "artifact_id": "p4-b0-variable-prefill-repair-validation-v1",
        "status": "pass",
        "scored": False,
        "authorization_consumed": True,
        "authorization": base._reference(AUTHORIZATION_PATH),
        "cases": summaries,
        "claims": {
            "isolated_r8_repair_gpu_validated": True,
            "r5cot_to_r8_repair_gpu_validated": True,
            "pure_decode_width_one_cpu_retained": True,
            "performance_claim_allowed": False,
            "value_screen_retry_authorized": False,
            "action_admission_authorized": False,
        },
        "next_action": (
            "Prepare a separate source-bound V11 review; do not reuse V10 or this "
            "non-scored validation output."
        ),
    }
    base._write_exclusive(output_dir / "validation.json", payload)


def _configure_base() -> None:
    base.AUTHORIZATION_PATH = AUTHORIZATION_PATH
    base.OUTPUT_DIR = OUTPUT_DIR
    base.EXPECTED_PACKAGE_ID = EXPECTED_PACKAGE_ID
    base.REQUIRED_SOURCE_PATHS = REQUIRED_SOURCE_PATHS
    base._scheduler_snapshot = diagnosis_v2.scheduler_snapshot
    base._install_instrumentation = _install_repair_instrumentation
    base.__file__ = str(Path(__file__).resolve())


def main() -> int:
    """Run the create-only repair validation package."""
    _configure_base()
    child = "--child-case" in sys.argv
    prepare_only = "--prepare-only" in sys.argv
    result = base.main()
    if not child and not prepare_only:
        finalize_validation(OUTPUT_DIR.resolve())
    return result


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        base.P4TransitionDiagnosisError,
        VariablePrefillRepairValidationError,
    ) as exc:
        print(f"P4 variable-prefill repair validation refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
