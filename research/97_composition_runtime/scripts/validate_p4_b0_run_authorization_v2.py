#!/usr/bin/env python3
"""Validate the fresh source-bound Phase 97 B0 GPU-4 authorization."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from run_p4_b0_value_screen import (
    P4RunnerError,
    build_boot_specs,
    validate_execution_authority,
)
from validate_p4_b0_capture_runner_conformance import (
    P4ConformanceError,
    validate_conformance,
)
from validate_p4_b0_run_authorization import (
    B0RunAuthorizationError,
)
from validate_p4_b0_run_authorization import (
    validate_authorization as validate_historical_authorization,
)

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
SCHEMA_PATH = PHASE_DIR / "schemas" / "p4_b0_run_authorization_v2.schema.json"
AUTHORIZATION_PATH = PHASE_DIR / "data" / "p4" / "p4_b0_run_authorization_v2.json"
HISTORICAL_PATH = PHASE_DIR / "data" / "p4" / "p4_b0_run_authorization.json"
CONFORMANCE_PATH = PHASE_DIR / "data" / "p4" / "p4_b0_capture_runner_conformance.json"
RUNNER_PATH = PHASE_DIR / "scripts" / "run_p4_b0_value_screen.py"
OUTPUT_PATH = PHASE_DIR / "data" / "p4" / "run_b0_value_screen_v1"

EXPECTED_BASIS = (
    "frozen_evidence_chain_valid",
    "six_capture_runner_checks_pass",
    "all_execution_sources_hash_bound",
    "runtime_resource_floor_fail_closed",
)
EXPECTED_INVALIDATORS = (
    "approved_source_hash_drift",
    "output_directory_exists",
    "gpu_identity_drift",
    "resource_floor_failure",
    "matrix_or_contract_drift",
)


class B0RunAuthorizationV2Error(ValueError):
    """Raised when the approved package drifts or inflates authority."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise B0RunAuthorizationV2Error(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise B0RunAuthorizationV2Error(
            f"cannot load JSON artifact {path}: {exc}"
        ) from exc
    _require(isinstance(value, dict), f"JSON artifact must be an object: {path}")
    return value


def _format_json_path(parts: Sequence[Any]) -> str:
    path = "$"
    for part in parts:
        path += f"[{part}]" if isinstance(part, int) else f".{part}"
    return path


def _repository_path(relative_path: str) -> Path:
    path = (REPO_ROOT / relative_path).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise B0RunAuthorizationV2Error(
            f"approved source escapes repository: {path}"
        ) from exc
    return path


def _validate_schema(authorization: Mapping[str, Any]) -> None:
    schema = _load_json(SCHEMA_PATH)
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise B0RunAuthorizationV2Error(
            f"invalid V2 authorization schema: {exc}"
        ) from exc
    errors = sorted(
        Draft202012Validator(schema).iter_errors(authorization),
        key=lambda error: [str(part) for part in error.absolute_path],
    )
    if errors:
        first = errors[0]
        path = _format_json_path(list(first.absolute_path))
        raise B0RunAuthorizationV2Error(
            f"V2 authorization schema rejected {path}: {first.message}"
        )


def _validate_sources(
    authorization: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    conformance = _load_json(CONFORMANCE_PATH)
    conformance_hash = hashlib.sha256(CONFORMANCE_PATH.read_bytes()).hexdigest()
    expected = {
        "capture_runner_conformance": {
            "path": str(CONFORMANCE_PATH.relative_to(REPO_ROOT)),
            "sha256": conformance_hash,
        },
        **conformance["source_artifacts"],
    }
    references = authorization["source_artifacts"]
    _require(
        references == expected,
        "approved source bindings differ from the conformance closure",
    )
    for role, reference in references.items():
        path = _repository_path(reference["path"])
        _require(path.is_file(), f"approved source is missing for {role}: {path}")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        _require(
            actual_hash == reference["sha256"],
            f"approved source hash drifted for {role}",
        )

    historical = _load_json(HISTORICAL_PATH)
    _require(
        authorization["supersedes"]
        == {
            "path": str(HISTORICAL_PATH.relative_to(REPO_ROOT)),
            "sha256": hashlib.sha256(HISTORICAL_PATH.read_bytes()).hexdigest(),
            "historical_decision": "hold",
        },
        "superseded HOLD binding drifted",
    )
    try:
        historical_result = validate_historical_authorization(
            historical,
            enforce_current_sources=False,
        )
        conformance_result = validate_conformance(conformance)
    except (B0RunAuthorizationError, P4ConformanceError) as exc:
        raise B0RunAuthorizationV2Error(
            f"authorization prerequisite no longer validates: {exc}"
        ) from exc
    _require(
        historical_result["authorization_decision"] == "hold"
        and conformance_result["status"] == "pass"
        and conformance_result["implementation_checks_passed"] == 6,
        "HOLD-to-conformance decision chain did not close",
    )
    return historical, conformance


def _validate_run_contract(
    authorization: Mapping[str, Any], historical: Mapping[str, Any]
) -> None:
    run = authorization["run_contract"]
    expected = copy.deepcopy(historical["run_contract"])
    expected["invocation"] = {
        "runner_path": str(RUNNER_PATH.relative_to(REPO_ROOT)),
        "argv": [
            ".venv/bin/python",
            str(RUNNER_PATH.relative_to(REPO_ROOT)),
            "--authorization",
            str(AUTHORIZATION_PATH.relative_to(REPO_ROOT)),
            "--output-dir",
            str(OUTPUT_PATH.relative_to(REPO_ROOT)),
        ],
        "output_dir": str(OUTPUT_PATH.relative_to(REPO_ROOT)),
        "overwrite_allowed": False,
        "runner_exists": True,
        "launchable_now": True,
    }
    _require(run == expected, "approved run contract differs from the frozen screen")
    _require(RUNNER_PATH.is_file(), "approved matrix runner is missing")
    _require(not OUTPUT_PATH.exists(), "approved create-new output already exists")
    specs = build_boot_specs(authorization, OUTPUT_PATH)
    _require(
        len(specs) == 9 and sum(len(spec["plan"]["cells"]) for spec in specs) == 432,
        "approved runner no longer constructs the complete matrix",
    )


def _validate_decision(
    authorization: Mapping[str, Any],
    historical: Mapping[str, Any],
    conformance: Mapping[str, Any],
) -> None:
    _require(
        authorization["resource_gates"] == historical["resource_gates"],
        "approved resource gates differ from the frozen HOLD",
    )
    _require(
        authorization["implementation_audit"]
        == {
            "state": "pass",
            "checks": conformance["implementation_checks"],
        },
        "approved implementation audit differs from conformance evidence",
    )
    readiness = authorization["evidence_readiness"]
    _require(
        readiness
        == {
            "state": "complete",
            "historical_hold_validated": True,
            "capture_runner_conformance_status": "pass",
            "implementation_checks_passed": 6,
        },
        "authorization readiness summary drifted",
    )
    decision = authorization["decision"]
    _require(
        tuple(decision["basis"]) == EXPECTED_BASIS
        and tuple(decision["invalidated_by"]) == EXPECTED_INVALIDATORS,
        "approval basis or invalidators drifted",
    )
    _require(
        authorization["claims"]
        == {
            "evidence_readiness_complete": True,
            "capture_runner_conformance_complete": True,
            "executable_run_ready": True,
            "exact_invocation_registered": True,
            "gpu_run_performed": False,
            "runtime_w512_switching_implemented": False,
            "action_admitted": False,
            "performance_claim_allowed": False,
        },
        "approval claims overstate completed measurement or admission",
    )
    _require(
        authorization["authorizations"]
        == {
            "capture_runner_conformance_engineering": True,
            "gpu_measurement": True,
            "p4a_engineering": False,
            "action_admission": False,
            "production_value_claim": False,
        },
        "approval grants authority beyond the GPU-4 value screen",
    )
    _require(
        authorization["execution_policy"]
        == {
            "physical_gpu_index": 4,
            "fallback_gpu_authorized": False,
            "physical_boot_count": 9,
            "capture_count": 432,
            "create_new_output_required": True,
            "partial_resume_allowed": False,
            "score_grants_authority": False,
            "on_any_failure": ("stop_without_scoring_and_require_fresh_authorization"),
        },
        "execution policy drifted or allows an unsafe retry",
    )
    _require(
        authorization["next_artifact"]
        == {
            "kind": "p4_b0_value_screen_result",
            "requires_complete_capture_count": 432,
            "may_authorize_p4a": False,
            "may_admit_action": False,
        },
        "post-run handoff grants downstream authority",
    )


def validate_authorization_v2(
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the one-screen approval and return its authority summary.

    Args:
        authorization: Parsed V2 run-authorization package.

    Returns:
        Machine-readable source, matrix, and authority summary.

    Raises:
        B0RunAuthorizationV2Error: If any approved input or boundary drifts.
    """
    _validate_schema(authorization)
    historical, conformance = _validate_sources(authorization)
    _validate_run_contract(authorization, historical)
    _validate_decision(authorization, historical, conformance)
    try:
        validate_execution_authority(authorization)
    except P4RunnerError as exc:
        raise B0RunAuthorizationV2Error(
            f"live runner rejects the approved package: {exc}"
        ) from exc
    return {
        "status": "pass",
        "package_id": authorization["package_id"],
        "authorization_decision": "approve",
        "approved_scope": authorization["decision"]["scope"],
        "source_bindings_current": True,
        "executable_run_ready": True,
        "gpu_physical_index": 4,
        "physical_boot_count": 9,
        "capture_count": 432,
        "minimum_launch_capacity_blocks": 21682,
        "gpu_measurement_authorized": True,
        "p4a_engineering_authorized": False,
        "action_admitted": False,
        "output_dir_create_new_ready": True,
        "next_artifact": "p4_b0_value_screen_result",
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Validate one fresh authorization package and print its summary."""
    args = parse_args()
    result = validate_authorization_v2(_load_json(args.authorization))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
