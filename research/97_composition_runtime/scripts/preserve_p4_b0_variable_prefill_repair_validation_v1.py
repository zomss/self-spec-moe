#!/usr/bin/env python3
"""Preserve and audit the consumed variable-prefill repair validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import run_p4_b0_variable_prefill_repair_validation as attempted

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
RUN_DIR = (
    PHASE_DIR / "data" / "p4" / "run_b0_variable_prefill_repair_validation_v1"
)
AUTHORIZATION_PATH = (
    PHASE_DIR
    / "data"
    / "p4"
    / "p4_b0_variable_prefill_repair_validation_authorization_v1.json"
)
DEFAULT_OUTPUT = (
    PHASE_DIR
    / "data"
    / "p4"
    / "p4_b0_variable_prefill_repair_validation_attempt_v1.json"
)
EXPECTED_PARENT_REJECTION = "r5cot-to-r8 armed-prefill evidence count drifted"


class RepairValidationPreservationError(RuntimeError):
    """Raised when the consumed GPU evidence cannot be preserved exactly."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RepairValidationPreservationError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RepairValidationPreservationError(f"cannot load {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON artifact is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _reference(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(REPO_ROOT)),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _is_exact_r8_observation(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    metadata = value.get("metadata")
    validated = value.get("validated_evidence")
    return (
        isinstance(metadata, Mapping)
        and metadata.get("capture_cohort_arm") is True
        and metadata.get("pure_decode") is False
        and metadata.get("decode_req_ids") == []
        and metadata.get("next_action_id") == attempted.base.ACTION_ID
        and value.get("proposal_called") is True
        and value.get("draft_output_shape") == [16, 4]
        and value.get("draft_step0_query_width") is None
        and value.get("draft_step0_num_tokens") == 2116
        and value.get("draft_step0_batch_size") == 16
        and value.get("draft_step0_runtime_mode") == "NONE"
        and value.get("draft_chain_runtime_mode") == "PIECEWISE"
        and isinstance(validated, Mapping)
        and validated.get("draft_step0_query_width") is None
        and validated.get("draft_step0_num_tokens") == 2116
        and validated.get("draft_step0_batch_size") == 16
        and validated.get("produced_draft_width") == 4
        and validated.get("draft_dispatched") is True
    )


def audit_case(case_id: str, result: Mapping[str, Any]) -> dict[str, Any]:
    """Audit one immutable GPU case without counting other prefill arms."""
    expected_labels = [row["label"] for row in attempted.base.JOBS[case_id]["cohorts"]]
    _require(
        result.get("status") == "completed_without_invariant_failure"
        and result.get("active_stage_at_exit") == "complete",
        f"{case_id} did not complete",
    )
    _require(
        result.get("completed_cohorts") == expected_labels,
        f"{case_id} cohort sequence drifted",
    )
    _require(
        result.get("primary_exception") is None
        and result.get("shutdown_exception") is None,
        f"{case_id} retained an exception",
    )
    resources = result.get("resources")
    recorder = result.get("recorder")
    trace = result.get("trace")
    instrumentation = result.get("instrumentation")
    _require(
        isinstance(resources, Mapping)
        and resources.get("shared_target_kv_block_capacity")
        >= attempted.base.MINIMUM_SHARED_KV_BLOCKS,
        f"{case_id} shared-KV capacity evidence failed",
    )
    _require(
        isinstance(recorder, Mapping)
        and recorder.get("closed") is True
        and recorder.get("event_count", 0) > 0,
        f"{case_id} recorder did not close",
    )
    _require(
        isinstance(trace, Mapping)
        and trace.get("present") is True
        and trace.get("record_count", 0) > 0,
        f"{case_id} trace did not close",
    )
    _require(
        isinstance(instrumentation, Mapping),
        f"{case_id} instrumentation is missing",
    )
    observations = instrumentation.get("armed_prefill_evidence")
    _require(isinstance(observations, list), f"{case_id} prefill evidence is missing")
    r8_matches = [row for row in observations if _is_exact_r8_observation(row)]
    _require(
        len(r8_matches) == 1,
        f"{case_id} does not contain exactly one exact R8 repair observation",
    )
    return {
        "case_id": case_id,
        "gpu": result.get("gpu"),
        "completed_cohorts": expected_labels,
        "armed_prefill_observation_count": len(observations),
        "exact_r8_observation_count": len(r8_matches),
        "exact_r8_observation": r8_matches[0],
        "measured_event_count": recorder["event_count"],
        "trace_record_count": trace["record_count"],
        "shared_target_kv_block_capacity": resources[
            "shared_target_kv_block_capacity"
        ],
    }


def build_preservation() -> dict[str, Any]:
    """Build the immutable attempt disposition from its existing artifacts."""
    _require(RUN_DIR.is_dir(), "consumed repair-validation output is missing")
    _require(
        not (RUN_DIR / "validation.json").exists(),
        "consumed parent unexpectedly emitted validation.json",
    )
    diagnosis = _load_json(RUN_DIR / "diagnosis.json")
    _require(
        diagnosis.get("status") == "complete"
        and diagnosis.get("authorization_consumed") is True
        and diagnosis.get("classification", {}).get("conclusion")
        == "original_failure_not_reproduced",
        "base two-case diagnosis did not preserve both repaired completions",
    )
    cases = []
    for case_id in attempted.base.JOBS:
        result = _load_json(RUN_DIR / case_id / "case_result.json")
        cases.append(audit_case(case_id, result))

    transition = _load_json(RUN_DIR / "r5cot-to-r8" / "case_result.json")
    try:
        attempted.validate_case_result("r5cot-to-r8", transition)
    except attempted.VariablePrefillRepairValidationError as exc:
        parent_rejection = str(exc)
    else:
        raise RepairValidationPreservationError(
            "consumed parent count guard no longer reproduces"
        )
    _require(
        parent_rejection == EXPECTED_PARENT_REJECTION,
        "consumed parent rejected for an unexpected reason",
    )

    artifact_paths = {
        "authorization": AUTHORIZATION_PATH,
        "aggregate_diagnosis": RUN_DIR / "diagnosis.json",
        "isolated_case": RUN_DIR / "isolated-r8" / "case_result.json",
        "isolated_trace": RUN_DIR / "isolated-r8" / "koff_trace.jsonl",
        "transition_case": RUN_DIR / "r5cot-to-r8" / "case_result.json",
        "transition_trace": RUN_DIR / "r5cot-to-r8" / "koff_trace.jsonl",
    }
    return {
        "schema_version": 1,
        "artifact_id": "p4-b0-variable-prefill-repair-validation-attempt-v1",
        "status": "gpu_cases_passed_parent_aggregate_rejected",
        "scored": False,
        "authorization_consumed": True,
        "artifacts": {
            name: _reference(path) for name, path in artifact_paths.items()
        },
        "parent_rejection": {
            "message": parent_rejection,
            "classification": "observer_cardinality_bug",
            "expected_count": 2,
            "observed_transition_armed_prefill_count": cases[1][
                "armed_prefill_observation_count"
            ],
            "caused_gpu_case_failure": False,
        },
        "cases": cases,
        "disposition": {
            "gpu_repair_validation_passed": True,
            "isolated_r8_passed": True,
            "r5cot_to_r8_passed": True,
            "parent_aggregate_passed": False,
            "retry_allowed": False,
            "resume_allowed": False,
            "output_reuse_allowed": False,
            "performance_claim_allowed": False,
            "value_screen_retry_authorized": False,
            "v11_authorized": False,
        },
        "next_action": (
            "Use this immutable case-level audit in a separate source-bound V11 "
            "authorization review; do not rerun or modify this attempt."
        ),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> int:
    """Print or exclusively write the preservation artifact."""
    args = parse_args()
    payload = json.dumps(build_preservation(), indent=2, sort_keys=True) + "\n"
    if args.out is None:
        print(payload, end="")
    else:
        try:
            with args.out.open("x", encoding="utf-8") as stream:
                stream.write(payload)
        except FileExistsError as exc:
            raise RepairValidationPreservationError(
                f"refusing to overwrite {args.out}"
            ) from exc
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RepairValidationPreservationError as exc:
        print(f"P4 repair validation preservation refused: {exc}")
        raise SystemExit(2) from exc
