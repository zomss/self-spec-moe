#!/usr/bin/env python3
"""Validate the CPU-only Phase 97 B0 capture-runner conformance package."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from run_p4_b0_value_screen import (
    MINIMUM_SHARED_KV_BLOCKS,
    P4RunnerError,
    _validate_plan,
    apply_full_prefill_engine_contract,
    build_boot_specs,
    validate_execution_authority,
)
from validate_p4_b0_run_authorization import (
    B0RunAuthorizationError,
    validate_authorization,
)

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
SCHEMA_PATH = PHASE_DIR / "schemas" / "p4_b0_capture_runner_conformance.schema.json"

EXPECTED_SOURCES = {
    "held_authorization": (
        "research/97_composition_runtime/data/p4/p4_b0_run_authorization.json"
    ),
    "live_runtime": "vllm/v1/spec_decode/koff_runtime.py",
    "live_scheduler": "vllm/v1/core/sched/scheduler.py",
    "model_runner": "vllm/v1/worker/gpu_model_runner.py",
    "environment_registry": "vllm/envs.py",
    "matrix_runner": (
        "research/97_composition_runtime/scripts/run_p4_b0_value_screen.py"
    ),
    "runtime_tests": "tests/v1/spec_decode/test_koff_runtime.py",
    "recorder_tests": (
        "research/97_composition_runtime/tests/test_p4_live_recorder.py"
    ),
    "runner_tests": (
        "research/97_composition_runtime/tests/test_p4_b0_capture_runner.py"
    ),
    "conformance_tests": (
        "research/97_composition_runtime/tests/test_p4_b0_capture_runner_conformance.py"
    ),
    "schema": (
        "research/97_composition_runtime/schemas/"
        "p4_b0_capture_runner_conformance.schema.json"
    ),
    "validator": (
        "research/97_composition_runtime/scripts/"
        "validate_p4_b0_capture_runner_conformance.py"
    ),
}
EXPECTED_CHECKS = (
    "w512_boot_contract_blocked",
    "w512_recorder_action_blocked",
    "multi_cell_same_boot_capture_missing",
    "boot_static_action_relabel_missing",
    "logical_weight_version_unstable",
    "runtime_resource_floor_guard_missing",
)


class P4ConformanceError(ValueError):
    """Raised when the conformance package drifts or inflates authority."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise P4ConformanceError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise P4ConformanceError(f"cannot load JSON artifact {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON artifact must be an object: {path}")
    return value


def _format_json_path(parts: Sequence[Any]) -> str:
    path = "$"
    for part in parts:
        path += f"[{part}]" if isinstance(part, int) else f".{part}"
    return path


def _validate_schema(conformance: Mapping[str, Any]) -> None:
    schema = _load_json(SCHEMA_PATH)
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise P4ConformanceError(f"invalid conformance schema: {exc}") from exc
    errors = sorted(
        Draft202012Validator(schema).iter_errors(conformance),
        key=lambda error: [str(part) for part in error.absolute_path],
    )
    if errors:
        first = errors[0]
        path = _format_json_path(list(first.absolute_path))
        raise P4ConformanceError(
            f"capture-runner conformance schema rejected {path}: {first.message}"
        )


def _repository_path(relative_path: str) -> Path:
    path = (REPO_ROOT / relative_path).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise P4ConformanceError(f"artifact escapes repository: {path}") from exc
    return path


def _validate_sources(conformance: Mapping[str, Any]) -> dict[str, Path]:
    references = conformance["source_artifacts"]
    _require(
        set(references) == set(EXPECTED_SOURCES),
        "conformance source roles are incomplete or inflated",
    )
    paths = {}
    for role, expected_path in EXPECTED_SOURCES.items():
        reference = references[role]
        _require(
            reference["path"] == expected_path,
            f"conformance source path drifted for {role}",
        )
        path = _repository_path(expected_path)
        _require(path.is_file(), f"missing conformance source: {path}")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        _require(
            reference["sha256"] == actual_hash,
            f"conformance source hash mismatch for {role}: "
            f"{actual_hash} != {reference['sha256']}",
        )
        paths[role] = path
    return paths


def _validate_live_contracts(
    conformance: Mapping[str, Any], paths: Mapping[str, Path]
) -> None:
    from vllm.v1.spec_decode.koff_runtime import (
        ACTIONS_BY_ID,
        K4_ACTION_ID,
        OFF_ACTION_ID,
        P4_CAPTURE_PLAN_CONTRACT_ID,
        P4_MIN_SHARED_KV_BLOCKS,
        W512_ACTION_ID,
    )

    _require(
        set(ACTIONS_BY_ID) == {OFF_ACTION_ID, K4_ACTION_ID}
        and W512_ACTION_ID not in ACTIONS_BY_ID,
        "w512 leaked into the executable K4/OFF registry",
    )
    _require(
        P4_CAPTURE_PLAN_CONTRACT_ID == "p4-b0-same-boot-capture-plan-v1",
        "same-boot capture-plan contract id drifted",
    )
    _require(
        P4_MIN_SHARED_KV_BLOCKS == MINIMUM_SHARED_KV_BLOCKS == 21682,
        "runtime and runner resource floors differ",
    )

    runtime_source = paths["live_runtime"].read_text(encoding="utf-8")
    scheduler_source = paths["live_scheduler"].read_text(encoding="utf-8")
    worker_source = paths["model_runner"].read_text(encoding="utf-8")
    _require(
        "w512 may relabel only an exact live K4-to-K4 event" in runtime_source
        and "shared_weight_binding_id" in runtime_source,
        "trusted relabel or pointer-binding proof is absent",
    )
    _require(
        scheduler_source.count("validate_p4_shared_kv_capacity(") >= 2,
        "scheduler does not guard launch and capture capacity",
    )
    _require(
        "self._koff_options.p4_logical_weight_version or None" in worker_source,
        "worker does not bind the stable logical weight version",
    )

    authorization = _load_json(paths["held_authorization"])
    historical = validate_authorization(
        authorization,
        enforce_current_sources=False,
    )
    _require(
        historical["authorization_decision"] == "hold"
        and not historical["reviewed_sources_current"],
        "the old authorization is not preserved as a historical HOLD",
    )
    try:
        validate_authorization(authorization)
    except B0RunAuthorizationError:
        pass
    else:
        raise P4ConformanceError(
            "the old authorization unexpectedly validates changed sources"
        )
    try:
        validate_execution_authority(authorization)
    except P4RunnerError:
        pass
    else:
        raise P4ConformanceError("the held package unexpectedly launches the runner")

    repaired = apply_full_prefill_engine_contract(authorization)
    specs = build_boot_specs(repaired, Path("/p4-conformance-validation-only"))
    _require(len(specs) == 9, "matrix runner does not build nine physical boots")
    for spec in specs:
        _validate_plan(spec["plan"])
        _require(
            len(spec["plan"]["cells"]) == 48,
            f"boot {spec['boot_id']} does not contain 48 cells",
        )
    _require(
        sum(len(spec["plan"]["cells"]) for spec in specs) == 432,
        "matrix runner does not close to 432 captures",
    )


def validate_conformance(conformance: Mapping[str, Any]) -> dict[str, Any]:
    """Validate sources, six checks, matrix closure, and authority boundary."""
    _validate_schema(conformance)
    paths = _validate_sources(conformance)
    checks = conformance["implementation_checks"]
    _require(
        tuple(row["code"] for row in checks) == EXPECTED_CHECKS,
        "conformance checks are missing, repeated, or reordered",
    )
    _require(
        all(row["status"] == "pass" and row["evidence"] for row in checks),
        "one capture-runner conformance check did not pass",
    )
    _require(
        not any(conformance["authorizations"].values()),
        "capture-runner conformance cannot grant authority",
    )
    claims = conformance["claims"]
    _require(
        claims["all_six_conformance_checks_pass"]
        and claims["executable_source_ready_for_reauthorization"]
        and not claims["gpu_measurement_performed"]
        and not claims["value_measured"]
        and not claims["action_admitted"]
        and not claims["performance_claim_allowed"],
        "conformance claims overstate measurement, admission, or performance",
    )
    _validate_live_contracts(conformance, paths)
    validation = conformance["validation"]
    return {
        "status": "pass",
        "artifact_id": conformance["artifact_id"],
        "implementation_checks_passed": len(checks),
        "physical_boot_count": conformance["contracts"]["same_boot_matrix"][
            "physical_boots"
        ],
        "cells_per_boot": conformance["contracts"]["same_boot_matrix"][
            "cells_per_boot"
        ],
        "capture_count": conformance["contracts"]["same_boot_matrix"]["captures"],
        "minimum_shared_kv_blocks": conformance["contracts"]["resource_floor"][
            "minimum_shared_kv_blocks"
        ],
        "focused_tests_passed": validation["focused_tests_passed"],
        "phase97_tests_passed": validation["phase97_tests_passed"],
        "gpu_commands_run": 0,
        "gpu_measurement_authorized": False,
        "p4a_engineering_authorized": False,
        "next_artifact": conformance["next_artifact"]["kind"],
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conformance", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Validate one checked conformance package and print its summary."""
    args = parse_args()
    result = validate_conformance(_load_json(args.conformance))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
