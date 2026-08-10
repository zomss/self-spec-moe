#!/usr/bin/env python3
"""Prepare or execute the fail-closed Phase 97 B0 value-screen matrix."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
PROMPT_MANIFEST_PATH = PHASE_DIR / "data" / "p4" / "p4_b0_prompt_manifest.json"
PROMPT_BUNDLE_PATH = PHASE_DIR / "data" / "p4" / "p4_b0_prompt_tokens.jsonl.gz"
SCORER_CONTRACT_PATH = PHASE_DIR / "data" / "p4" / "p4_b0_runner_scorer_contract.json"
CONFORMANCE_PATH = PHASE_DIR / "data" / "p4" / "p4_b0_capture_runner_conformance.json"

PLAN_CONTRACT_ID = "p4-b0-same-boot-capture-plan-v1"
CURRENT_HOLD_PACKAGE_ID = "p4-b0-value-screen-run-authorization-v1"
BASE_PACKAGE_ID = "p4-b0-value-screen-run-authorization-v2"
CONSUMED_PACKAGE_ID = "p4-b0-value-screen-run-authorization-v3"
V4_PACKAGE_ID = "p4-b0-value-screen-run-authorization-v4"
APPROVED_PACKAGE_ID = "p4-b0-value-screen-run-authorization-v5"
BASE_AUTHORIZATION_PATH = (
    "research/97_composition_runtime/data/p4/p4_b0_run_authorization_v2.json"
)
CONSUMED_AUTHORIZATION_PATH = (
    "research/97_composition_runtime/data/p4/p4_b0_run_authorization_v3.json"
)
V4_AUTHORIZATION_PATH = (
    "research/97_composition_runtime/data/p4/p4_b0_run_authorization_v4.json"
)
V4_OUTPUT_PATH = "research/97_composition_runtime/data/p4/run_b0_value_screen_v3"
V4_PRIOR_FAILED_ATTEMPT_PATH = (
    "research/97_composition_runtime/data/p4/run_b0_value_screen_v2/failure.json"
)
APPROVED_AUTHORIZATION_PATH = (
    "research/97_composition_runtime/data/p4/p4_b0_run_authorization_v5.json"
)
APPROVED_OUTPUT_PATH = "research/97_composition_runtime/data/p4/run_b0_value_screen_v4"
V5_FAILED_ATTEMPT_PATH = (
    "research/97_composition_runtime/data/p4/run_b0_value_screen_v4/failure.json"
)
V6_PACKAGE_ID = "p4-b0-value-screen-run-authorization-v6"
V6_AUTHORIZATION_PATH = (
    "research/97_composition_runtime/data/p4/p4_b0_run_authorization_v6.json"
)
V6_OUTPUT_PATH = "research/97_composition_runtime/data/p4/run_b0_value_screen_v5"
V6_FAILED_ATTEMPT_PATH = (
    "research/97_composition_runtime/data/p4/run_b0_value_screen_v5/failure.json"
)
V7_PACKAGE_ID = "p4-b0-value-screen-run-authorization-v7"
V7_AUTHORIZATION_PATH = (
    "research/97_composition_runtime/data/p4/p4_b0_run_authorization_v7.json"
)
V7_OUTPUT_PATH = "research/97_composition_runtime/data/p4/run_b0_value_screen_v6"
V8_PACKAGE_ID = "p4-b0-value-screen-run-authorization-v8"
V8_AUTHORIZATION_PATH = (
    "research/97_composition_runtime/data/p4/p4_b0_run_authorization_v8.json"
)
V8_OUTPUT_PATH = "research/97_composition_runtime/data/p4/run_b0_value_screen_v7"
V8_FAILED_ATTEMPT_PATH = (
    "research/97_composition_runtime/data/p4/run_b0_value_screen_v6/failure.json"
)
V9_PACKAGE_ID = "p4-b0-value-screen-run-authorization-v9"
V9_AUTHORIZATION_PATH = (
    "research/97_composition_runtime/data/p4/p4_b0_run_authorization_v9.json"
)
V9_OUTPUT_PATH = "research/97_composition_runtime/data/p4/run_b0_value_screen_v8"
V9_FAILED_ATTEMPT_PATH = (
    "research/97_composition_runtime/data/p4/run_b0_value_screen_v7/failure.json"
)
V10_PACKAGE_ID = "p4-b0-value-screen-run-authorization-v10"
V10_AUTHORIZATION_PATH = (
    "research/97_composition_runtime/data/p4/p4_b0_run_authorization_v10.json"
)
V10_OUTPUT_PATH = "research/97_composition_runtime/data/p4/run_b0_value_screen_v9"
V10_FAILED_ATTEMPT_PATH = (
    "research/97_composition_runtime/data/p4/run_b0_value_screen_v8/failure.json"
)
V11_PACKAGE_ID = "p4-b0-value-screen-run-authorization-v11"
V11_AUTHORIZATION_PATH = (
    "research/97_composition_runtime/data/p4/p4_b0_run_authorization_v11.json"
)
V11_OUTPUT_PATH = "research/97_composition_runtime/data/p4/run_b0_value_screen_v10"
V11_FAILED_ATTEMPT_PATH = (
    "research/97_composition_runtime/data/p4/run_b0_value_screen_v9/failure.json"
)
V11_CAPTURE_MANIFEST_PATH = (
    "research/97_composition_runtime/data/p4/"
    "run_b0_value_screen_v9/capture_manifest.json"
)
VARIABLE_PREFILL_REPAIR_AUTHORIZATION_PATH = (
    "research/97_composition_runtime/data/p4/"
    "p4_b0_variable_prefill_repair_validation_authorization_v1.json"
)
VARIABLE_PREFILL_REPAIR_AUDIT_PATH = (
    "research/97_composition_runtime/data/p4/"
    "p4_b0_variable_prefill_repair_validation_attempt_v1.json"
)
VARIABLE_PREFILL_REPAIR_AGGREGATE_PATH = (
    "research/97_composition_runtime/data/p4/"
    "run_b0_variable_prefill_repair_validation_v1/diagnosis.json"
)
VARIABLE_PREFILL_REPAIR_ISOLATED_CASE_PATH = (
    "research/97_composition_runtime/data/p4/"
    "run_b0_variable_prefill_repair_validation_v1/isolated-r8/case_result.json"
)
VARIABLE_PREFILL_REPAIR_ISOLATED_TRACE_PATH = (
    "research/97_composition_runtime/data/p4/"
    "run_b0_variable_prefill_repair_validation_v1/isolated-r8/koff_trace.jsonl"
)
VARIABLE_PREFILL_REPAIR_TRANSITION_CASE_PATH = (
    "research/97_composition_runtime/data/p4/"
    "run_b0_variable_prefill_repair_validation_v1/r5cot-to-r8/case_result.json"
)
VARIABLE_PREFILL_REPAIR_TRANSITION_TRACE_PATH = (
    "research/97_composition_runtime/data/p4/"
    "run_b0_variable_prefill_repair_validation_v1/r5cot-to-r8/koff_trace.jsonl"
)
CHUNKED_PREFILL_PROBE_AUTHORIZATION_PATH = (
    "research/97_composition_runtime/data/p4/"
    "p4_b0_chunked_prefill_probe_authorization_v5.json"
)
CHUNKED_PREFILL_PROBE_PREPARATION_PATH = (
    "research/97_composition_runtime/data/p4/"
    "run_b0_chunked_prefill_probe_v5/preparation.json"
)
CHUNKED_PREFILL_PROBE_RESULT_PATH = (
    "research/97_composition_runtime/data/p4/"
    "run_b0_chunked_prefill_probe_v5/probe_result.json"
)
CHUNKED_PREFILL_PROBE_TRACE_PATH = (
    "research/97_composition_runtime/data/p4/"
    "run_b0_chunked_prefill_probe_v5/koff_trace.jsonl"
)
PREFILL_SAMPLED_TOKENS_PER_REQUEST = 1
ATOMIC_INGRESS_AUTHORIZATION_PATH = (
    "research/97_composition_runtime/data/p4/"
    "p4_b0_atomic_ingress_proof_authorization_v2.json"
)
ATOMIC_INGRESS_PROOF_PATH = (
    "research/97_composition_runtime/data/p4/run_b0_atomic_ingress_proof_v2/proof.json"
)
ATOMIC_INGRESS_TRACE_PATH = (
    "research/97_composition_runtime/data/p4/"
    "run_b0_atomic_ingress_proof_v2/koff_trace.jsonl"
)
FAILED_ATTEMPT_PATH = (
    "research/97_composition_runtime/data/p4/run_b0_value_screen_v3/failure.json"
)
RESOURCE_REPAIR_AUTHORIZATION_PATH = (
    "research/97_composition_runtime/data/p4/"
    "p4_b0_full_prefill_resource_repair_probe_authorization.json"
)
RESOURCE_REPAIR_RESULT_PATH = (
    "research/97_composition_runtime/data/p4/"
    "p4_b0_full_prefill_resource_repair_probe.json"
)
NATIVE_SAMPLER_ENV = "VLLM_USE_FLASHINFER_SAMPLER"
NATIVE_SAMPLER_VALUE = "0"
V1_MULTIPROCESSING_ENV = "VLLM_ENABLE_V1_MULTIPROCESSING"
V1_MULTIPROCESSING_VALUE = "0"
INPROCESS_ENGINE_CORE_CLASS = "InprocClient"
GPU0_UUID = "GPU-4938442e-5508-9249-0fa6-37baa1985703"
GPU1_UUID = "GPU-ba39f4f0-61fe-34ca-c1af-ffe565b70923"
GPU4_UUID = "GPU-c9d19019-5065-2353-80a9-f1797eb19d51"
SCREEN_ID = "p4-b0-off-k4-w512-value-screen-v1"
LOGICAL_VERSION_PREFIX = "target-matching-config-sha256-"
MINIMUM_SHARED_KV_BLOCKS = 21682
FULL_PREFILL_MAX_NUM_BATCHED_TOKENS = 114688
MEASUREMENT_GPU_MEMORY_UTILIZATION = 0.96
SERVING_MAX_NUM_BATCHED_TOKENS = 8192
SERVING_GPU_MEMORY_UTILIZATION = 0.90
SERVING_EFFECTIVE_MAX_NUM_SCHEDULED_TOKENS = 8160
EXPECTED_MAX_MICROBATCH_PROMPT_TOKENS = 112908
SPECULATIVE_SLOT_RESERVE_PER_SEQUENCE = 1
REGIME_ORDER = ("R4", "R5", "R5cot", "R8", "R1", "R6")
CONTENT_SEEDS = (0, 1)
ROUNDS = (1, 2, 3, 4)
ACTION_ORDERS = {
    1: ("off", "target-matching-k4", "target-matching-w512-masked-k4"),
    2: ("target-matching-k4", "target-matching-w512-masked-k4", "off"),
    3: ("target-matching-w512-masked-k4", "off", "target-matching-k4"),
}
ACTION_REALIZATIONS = {
    "off": "live-b0-forced-off",
    "target-matching-k4": "live-b0-target-matching-k4",
    "target-matching-w512-masked-k4": "boot-static-mask-equivalent-surrogate",
}
ACTION_SLUGS = {
    "off": "off",
    "target-matching-k4": "k4",
    "target-matching-w512-masked-k4": "w512",
}


class P4RunnerError(RuntimeError):
    """Raised when preparation or execution cannot prove its authority."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise P4RunnerError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise P4RunnerError(f"cannot load JSON artifact {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON artifact must be an object: {path}")
    return value


def _repository_path(relative_path: str) -> Path:
    path = (REPO_ROOT / relative_path).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise P4RunnerError(f"approved source escapes repository: {path}") from exc
    return path


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _file_reference(relative_path: str) -> dict[str, str]:
    path = _repository_path(relative_path)
    _require(path.is_file(), f"required retry artifact is missing: {path}")
    return {
        "path": relative_path,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _validate_retry_package(
    package: Mapping[str, Any], *, require_output_absent: bool
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "package_id",
        "date",
        "status",
        "prior_authorization",
        "failed_attempt",
        "source_artifacts",
        "run_contract",
        "repair_audit",
        "decision",
        "claims",
        "authorizations",
        "execution_policy",
        "next_artifact",
    }
    _require(set(package) == expected_keys, "V4 retry package fields drifted")
    _require(
        package.get("schema_version") == 4
        and package.get("package_id") == V4_PACKAGE_ID
        and package.get("status") == "authorized_gpu4_native_sampler_retry_only",
        "runner accepts only the reviewed V4 GPU-4 retry authorization",
    )

    prior_ref = _file_reference(CONSUMED_AUTHORIZATION_PATH)
    _require(
        package["prior_authorization"]
        == {
            **prior_ref,
            "disposition": "consumed_failed_before_capture",
        },
        "V4 package does not bind the consumed V3 authorization",
    )
    prior = _load_json(_repository_path(CONSUMED_AUTHORIZATION_PATH))
    _require(
        prior.get("package_id") == CONSUMED_PACKAGE_ID
        and prior.get("schema_version") == 3,
        "V4 prior package is not the immutable V3 authorization",
    )
    base_ref = _file_reference(BASE_AUTHORIZATION_PATH)
    base = _load_json(_repository_path(BASE_AUTHORIZATION_PATH))
    _require(
        base.get("package_id") == BASE_PACKAGE_ID and base.get("schema_version") == 2,
        "V4 base is not the immutable full V2 authorization",
    )

    failure_ref = _file_reference(V4_PRIOR_FAILED_ATTEMPT_PATH)
    _require(
        package["failed_attempt"]
        == {
            **failure_ref,
            "capture_count": 0,
            "score_emitted": False,
            "preserve_without_resume": True,
        },
        "V4 package does not bind the zero-capture V3 failed attempt",
    )
    failure = _load_json(_repository_path(V4_PRIOR_FAILED_ATTEMPT_PATH))
    _require(
        failure.get("record_type") == "p4_b0_value_screen_execution_failure"
        and failure.get("attempt", {}).get("captures_emitted") == 0
        and failure.get("attempt", {}).get("score_emitted") is False
        and failure.get("disposition", {}).get("requires_fresh_authorization") is True,
        "V3 failed-attempt evidence does not support a fresh authorization",
    )

    retry_sources = {
        "matrix_runner": (
            "research/97_composition_runtime/scripts/run_p4_b0_value_screen.py"
        ),
        "runner_tests": (
            "research/97_composition_runtime/tests/test_p4_b0_capture_runner.py"
        ),
        "conformance_tests": (
            "research/97_composition_runtime/tests/"
            "test_p4_b0_capture_runner_conformance.py"
        ),
        "sampler_backend": "vllm/v1/sample/ops/topk_topp_sampler.py",
        "retry_schema": (
            "research/97_composition_runtime/schemas/"
            "p4_b0_run_authorization_v4.schema.json"
        ),
        "retry_validator": (
            "research/97_composition_runtime/scripts/"
            "validate_p4_b0_run_authorization_v4.py"
        ),
        "retry_tests": (
            "research/97_composition_runtime/tests/test_p4_b0_run_authorization_v4.py"
        ),
    }
    references = package["source_artifacts"]
    _require(
        set(references) == set(retry_sources),
        "V4 retry source closure is incomplete or inflated",
    )
    for role, path in retry_sources.items():
        _require(
            references[role] == _file_reference(path),
            f"V4 retry source hash drifted for {role}",
        )

    expected_invocation = {
        "runner_path": retry_sources["matrix_runner"],
        "argv": [
            ".venv/bin/python",
            retry_sources["matrix_runner"],
            "--authorization",
            V4_AUTHORIZATION_PATH,
            "--output-dir",
            V4_OUTPUT_PATH,
        ],
        "output_dir": V4_OUTPUT_PATH,
        "overwrite_allowed": False,
        "runner_exists": True,
        "launchable_now": True,
    }
    _require(
        package["run_contract"]
        == {
            "base_authorization": base_ref,
            "invocation": expected_invocation,
        },
        "V4 retry invocation differs from the create-new contract",
    )
    _require(
        package["repair_audit"]
        == {
            "state": "pass",
            "root_cause": (
                "flashinfer_cached_sampler_linked_to_unresolved_libcudart13"
            ),
            "sampler_policy": "pytorch_native",
            "required_environment": {NATIVE_SAMPLER_ENV: NATIVE_SAMPLER_VALUE},
            "workload_seed_policy": "per_request_seeded",
            "preflight_before_output_creation": True,
            "checks": [
                "native_sampler_environment_forced",
                "boot_spec_cannot_override_sampler_policy",
                "native_sampler_binding_subprocess_passes",
                "seeded_workload_native_fallback_bound",
                "virtualenv_ninja_preflight_retained",
            ],
        },
        "V4 native-sampler repair audit drifted",
    )
    _require(
        package["decision"]
        == {
            "state": "approve",
            "scope": "gpu4_native_sampler_boot_static_value_screen_retry_only",
            "basis": [
                "v3_stopped_before_first_capture",
                "v3_output_preserved_without_resume",
                "native_sampler_matches_seeded_request_path",
                "native_sampler_preflight_passes",
                "retry_sources_hash_bound",
                "fresh_output_path_registered",
            ],
            "invalidated_by": [
                "approved_source_hash_drift",
                "fresh_output_directory_exists",
                "native_sampler_preflight_failure",
                "virtualenv_tool_preflight_failure",
                "gpu_identity_drift",
                "resource_floor_failure",
                "matrix_or_contract_drift",
            ],
        },
        "V4 retry decision boundary drifted",
    )
    _require(
        package["claims"]
        == {
            "prior_gpu_attempts_performed": 2,
            "latest_prior_capture_count": 0,
            "latest_prior_score_emitted": False,
            "native_sampler_repair_complete": True,
            "executable_run_ready": True,
            "runtime_w512_switching_implemented": False,
            "action_admitted": False,
            "performance_claim_allowed": False,
        },
        "V4 retry claims drifted or overstate evidence",
    )
    _require(
        package["authorizations"]
        == {
            "gpu_measurement": True,
            "p4a_engineering": False,
            "action_admission": False,
            "production_value_claim": False,
        },
        "V4 retry authority exceeds one GPU measurement",
    )
    _require(
        package["execution_policy"]
        == {
            "physical_gpu_index": 4,
            "fallback_gpu_authorized": False,
            "physical_boot_count": 9,
            "capture_count": 432,
            "create_new_output_required": True,
            "prior_output_reuse_allowed": False,
            "partial_resume_allowed": False,
            "score_grants_authority": False,
            "on_any_failure": ("stop_without_scoring_and_require_fresh_authorization"),
        },
        "V4 retry execution policy drifted",
    )
    _require(
        package["next_artifact"]
        == {
            "kind": "p4_b0_value_screen_result",
            "requires_complete_capture_count": 432,
            "may_authorize_p4a": False,
            "may_admit_action": False,
        },
        "V4 post-run boundary drifted",
    )
    _require(
        _repository_path(V4_PRIOR_FAILED_ATTEMPT_PATH).parent
        != _repository_path(V4_OUTPUT_PATH),
        "V4 retry output aliases the failed V3 attempt",
    )
    if require_output_absent:
        _require(
            not _repository_path(V4_OUTPUT_PATH).exists(),
            "approved V4 create-new output already exists",
        )
    return base


def _serving_chunked_prefill_boundary() -> dict[str, Any]:
    return {
        "scope": "post_measurement_real_serving_diagnosis",
        "max_num_batched_tokens": SERVING_MAX_NUM_BATCHED_TOKENS,
        "chunked_prefill_enabled": True,
        "gpu_memory_utilization": SERVING_GPU_MEMORY_UTILIZATION,
        "measurement_override_is_serving_default": False,
        "mixed_prefill_decode_diagnosis_required": True,
    }


def _validate_v5_package(
    package: Mapping[str, Any], *, require_output_absent: bool
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "package_id",
        "date",
        "status",
        "prior_authorization",
        "failed_attempt",
        "resource_probe",
        "source_artifacts",
        "run_contract",
        "measurement_repair",
        "serving_chunked_prefill_boundary",
        "decision",
        "claims",
        "authorizations",
        "execution_policy",
        "next_artifact",
    }
    _require(set(package) == expected_keys, "V5 retry package fields drifted")
    _require(
        package.get("schema_version") == 5
        and package.get("package_id") == APPROVED_PACKAGE_ID
        and package.get("status")
        == "authorized_gpu4_full_prefill_resource_repair_retry_only",
        "runner accepts only the reviewed V5 GPU-4 retry authorization",
    )

    prior_ref = _file_reference(V4_AUTHORIZATION_PATH)
    _require(
        package["prior_authorization"]
        == {
            **prior_ref,
            "disposition": "consumed_failed_during_first_capture",
        },
        "V5 package does not bind the consumed V4 authorization",
    )
    prior = _load_json(_repository_path(V4_AUTHORIZATION_PATH))
    _require(
        prior.get("package_id") == V4_PACKAGE_ID and prior.get("schema_version") == 4,
        "V5 prior package is not the immutable V4 authorization",
    )

    failure_ref = _file_reference(FAILED_ATTEMPT_PATH)
    _require(
        package["failed_attempt"]
        == {
            **failure_ref,
            "capture_count": 0,
            "score_emitted": False,
            "preserve_without_resume": True,
        },
        "V5 package does not bind the zero-capture V4 failed attempt",
    )
    failure = _load_json(_repository_path(FAILED_ATTEMPT_PATH))
    _require(
        failure.get("record_type") == "p4_b0_value_screen_execution_failure"
        and failure.get("authorization", {}).get("package_id") == V4_PACKAGE_ID
        and failure.get("attempt", {}).get("captures_emitted") == 0
        and failure.get("attempt", {}).get("score_emitted") is False
        and failure.get("disposition", {}).get("requires_fresh_authorization") is True,
        "V4 failed-attempt evidence does not support a fresh authorization",
    )

    resource_auth_ref = _file_reference(RESOURCE_REPAIR_AUTHORIZATION_PATH)
    resource_result_ref = _file_reference(RESOURCE_REPAIR_RESULT_PATH)
    resource_result = _load_json(_repository_path(RESOURCE_REPAIR_RESULT_PATH))
    resource_probe = package["resource_probe"]
    _require(
        resource_probe
        == {
            "authorization": resource_auth_ref,
            "result": resource_result_ref,
            "decision": "pass",
            "gpu_memory_utilization": MEASUREMENT_GPU_MEMORY_UTILIZATION,
            "num_gpu_blocks": 22113,
            "minimum_launch_blocks": MINIMUM_SHARED_KV_BLOCKS,
            "headroom_blocks": 431,
            "requests_executed": 0,
        },
        "V5 resource-probe evidence drifted",
    )
    _require(
        resource_result.get("record_type")
        == "p4_b0_full_prefill_resource_repair_probe_result"
        and resource_result.get("decision", {}).get("state") == "pass"
        and resource_result.get("engine", {}).get("gpu_memory_utilization")
        == MEASUREMENT_GPU_MEMORY_UTILIZATION
        and resource_result.get("engine", {}).get("requests_executed") == 0
        and resource_result.get("shared_kv_capacity", {}).get("num_gpu_blocks") == 22113
        and resource_result.get("shared_kv_capacity", {}).get("headroom_blocks") == 431
        and resource_result.get("serving_chunked_prefill_boundary")
        == _serving_chunked_prefill_boundary(),
        "V5 resource-probe result does not close the repaired capacity gate",
    )

    base_ref = _file_reference(BASE_AUTHORIZATION_PATH)
    base = _load_json(_repository_path(BASE_AUTHORIZATION_PATH))
    _require(
        base.get("package_id") == BASE_PACKAGE_ID and base.get("schema_version") == 2,
        "V5 base is not the immutable full V2 authorization",
    )
    retry_sources = {
        "matrix_runner": (
            "research/97_composition_runtime/scripts/run_p4_b0_value_screen.py"
        ),
        "runner_tests": (
            "research/97_composition_runtime/tests/test_p4_b0_capture_runner.py"
        ),
        "conformance_tests": (
            "research/97_composition_runtime/tests/"
            "test_p4_b0_capture_runner_conformance.py"
        ),
        "validator": (
            "research/97_composition_runtime/scripts/"
            "validate_p4_b0_capture_runner_conformance.py"
        ),
        "sampler_backend": "vllm/v1/sample/ops/topk_topp_sampler.py",
        "resource_repair_probe_script": (
            "research/97_composition_runtime/scripts/"
            "probe_p4_b0_full_prefill_resource_repair.py"
        ),
        "resource_repair_probe_tests": (
            "research/97_composition_runtime/tests/"
            "test_p4_b0_full_prefill_resource_repair_probe.py"
        ),
        "retry_schema": (
            "research/97_composition_runtime/schemas/"
            "p4_b0_run_authorization_v5.schema.json"
        ),
        "retry_validator": (
            "research/97_composition_runtime/scripts/"
            "validate_p4_b0_run_authorization_v5.py"
        ),
        "retry_tests": (
            "research/97_composition_runtime/tests/test_p4_b0_run_authorization_v5.py"
        ),
    }
    references = package["source_artifacts"]
    _require(
        set(references) == set(retry_sources),
        "V5 retry source closure is incomplete or inflated",
    )
    for role, path in retry_sources.items():
        _require(
            references[role] == _file_reference(path),
            f"V5 retry source hash drifted for {role}",
        )

    expected_invocation = {
        "runner_path": retry_sources["matrix_runner"],
        "argv": [
            ".venv/bin/python",
            retry_sources["matrix_runner"],
            "--authorization",
            APPROVED_AUTHORIZATION_PATH,
            "--output-dir",
            APPROVED_OUTPUT_PATH,
        ],
        "output_dir": APPROVED_OUTPUT_PATH,
        "overwrite_allowed": False,
        "runner_exists": True,
        "launchable_now": True,
    }
    _require(
        package["run_contract"]
        == {
            "base_authorization": base_ref,
            "invocation": expected_invocation,
        },
        "V5 retry invocation differs from the create-new contract",
    )
    _require(
        package["measurement_repair"]
        == {
            "state": "pass",
            "root_cause": "full_prefill_profile_reduced_shared_kv_below_floor",
            "allowed_engine_overrides": {
                "max_num_batched_tokens": FULL_PREFILL_MAX_NUM_BATCHED_TOKENS,
                "enable_chunked_prefill": True,
                "gpu_memory_utilization": MEASUREMENT_GPU_MEMORY_UTILIZATION,
            },
            "minimum_shared_kv_blocks": MINIMUM_SHARED_KV_BLOCKS,
            "measured_shared_kv_blocks": 22113,
            "headroom_blocks": 431,
            "checks": [
                "manifest_derived_full_prefill_budget",
                "chunked_prefill_retained",
                "only_measurement_engine_overrides_applied",
                "zero_request_resource_probe_passes",
                "serving_8192_boundary_preserved",
            ],
        },
        "V5 full-prefill resource repair drifted",
    )
    _require(
        package["serving_chunked_prefill_boundary"]
        == _serving_chunked_prefill_boundary(),
        "V5 serving chunked-prefill boundary drifted",
    )
    _require(
        package["decision"]
        == {
            "state": "approve",
            "scope": "gpu4_full_prefill_resource_repair_value_screen_retry_only",
            "basis": [
                "v4_stopped_with_zero_complete_captures",
                "v4_output_preserved_without_resume",
                "full_prefill_budget_closes_capture_ingress",
                "resource_repair_probe_passes",
                "native_sampler_policy_retained",
                "retry_sources_hash_bound",
                "fresh_output_path_registered",
            ],
            "invalidated_by": [
                "approved_source_hash_drift",
                "fresh_output_directory_exists",
                "native_sampler_preflight_failure",
                "virtualenv_tool_preflight_failure",
                "gpu_identity_drift",
                "resource_floor_failure",
                "measurement_engine_override_drift",
                "serving_boundary_drift",
                "matrix_or_contract_drift",
            ],
        },
        "V5 retry decision boundary drifted",
    )
    _require(
        package["claims"]
        == {
            "prior_gpu_attempts_performed": 3,
            "latest_prior_capture_count": 0,
            "latest_prior_score_emitted": False,
            "full_prefill_resource_repair_complete": True,
            "executable_run_ready": True,
            "serving_chunked_prefill_validated": False,
            "runtime_w512_switching_implemented": False,
            "action_admitted": False,
            "performance_claim_allowed": False,
        },
        "V5 retry claims drifted or overstate evidence",
    )
    _require(
        package["authorizations"]
        == {
            "gpu_measurement": True,
            "serving_diagnosis": False,
            "p4a_engineering": False,
            "action_admission": False,
            "production_value_claim": False,
        },
        "V5 retry authority exceeds one GPU measurement",
    )
    _require(
        package["execution_policy"]
        == {
            "physical_gpu_index": 4,
            "fallback_gpu_authorized": False,
            "physical_boot_count": 9,
            "capture_count": 432,
            "create_new_output_required": True,
            "prior_output_reuse_allowed": False,
            "partial_resume_allowed": False,
            "score_grants_authority": False,
            "on_any_failure": ("stop_without_scoring_and_require_fresh_authorization"),
        },
        "V5 retry execution policy drifted",
    )
    _require(
        package["next_artifact"]
        == {
            "kind": "p4_b0_value_screen_result",
            "requires_complete_capture_count": 432,
            "may_authorize_p4a": False,
            "may_admit_action": False,
        },
        "V5 post-run boundary drifted",
    )
    _require(
        _repository_path(FAILED_ATTEMPT_PATH).parent
        != _repository_path(APPROVED_OUTPUT_PATH),
        "V5 retry output aliases the failed V4 attempt",
    )
    if require_output_absent:
        _require(
            not _repository_path(APPROVED_OUTPUT_PATH).exists(),
            "approved V5 create-new output already exists",
        )
    return base


def _v6_source_paths() -> dict[str, str]:
    return {
        "matrix_runner": (
            "research/97_composition_runtime/scripts/run_p4_b0_value_screen.py"
        ),
        "runner_tests": (
            "research/97_composition_runtime/tests/test_p4_b0_capture_runner.py"
        ),
        "conformance_tests": (
            "research/97_composition_runtime/tests/"
            "test_p4_b0_capture_runner_conformance.py"
        ),
        "conformance_validator": (
            "research/97_composition_runtime/scripts/"
            "validate_p4_b0_capture_runner_conformance.py"
        ),
        "runtime_tests": "tests/v1/spec_decode/test_koff_runtime.py",
        "recorder_tests": (
            "research/97_composition_runtime/tests/test_p4_live_recorder.py"
        ),
        "environment": "vllm/envs.py",
        "engine_args": "vllm/engine/arg_utils.py",
        "llm_engine": "vllm/v1/engine/llm_engine.py",
        "engine_core_client": "vllm/v1/engine/core_client.py",
        "uniproc_executor": "vllm/v1/executor/uniproc_executor.py",
        "scheduler": "vllm/v1/core/sched/scheduler.py",
        "koff_runtime": "vllm/v1/spec_decode/koff_runtime.py",
        "model_runner": "vllm/v1/worker/gpu_model_runner.py",
        "draft_model": "vllm/v1/spec_decode/draft_model.py",
        "draft_proposer": "vllm/v1/spec_decode/llm_base_proposer.py",
        "sampler_backend": "vllm/v1/sample/ops/topk_topp_sampler.py",
        "adapter": (
            "research/97_composition_runtime/scripts/adapt_p4_b0_same_event.py"
        ),
        "scorer": "research/97_composition_runtime/scripts/score_p4_b0.py",
        "authorization_schema": (
            "research/97_composition_runtime/schemas/"
            "p4_b0_run_authorization_v6.schema.json"
        ),
        "authorization_validator": (
            "research/97_composition_runtime/scripts/"
            "validate_p4_b0_run_authorization_v6.py"
        ),
        "authorization_tests": (
            "research/97_composition_runtime/tests/test_p4_b0_run_authorization_v6.py"
        ),
    }


def _v7_source_paths() -> dict[str, str]:
    return {
        "matrix_runner": (
            "research/97_composition_runtime/scripts/run_p4_b0_value_screen.py"
        ),
        "runner_tests": (
            "research/97_composition_runtime/tests/test_p4_b0_capture_runner.py"
        ),
        "conformance_tests": (
            "research/97_composition_runtime/tests/"
            "test_p4_b0_capture_runner_conformance.py"
        ),
        "conformance_validator": (
            "research/97_composition_runtime/scripts/"
            "validate_p4_b0_capture_runner_conformance.py"
        ),
        "runtime_tests": "tests/v1/spec_decode/test_koff_runtime.py",
        "recorder_tests": (
            "research/97_composition_runtime/tests/test_p4_live_recorder.py"
        ),
        "atomic_ingress_runner": (
            "research/97_composition_runtime/scripts/run_p4_b0_atomic_ingress_proof.py"
        ),
        "atomic_ingress_tests": (
            "research/97_composition_runtime/tests/test_p4_b0_atomic_ingress_proof.py"
        ),
        "input_processor": "vllm/v1/engine/input_processor.py",
        "environment": "vllm/envs.py",
        "engine_args": "vllm/engine/arg_utils.py",
        "llm_engine": "vllm/v1/engine/llm_engine.py",
        "engine_core_client": "vllm/v1/engine/core_client.py",
        "uniproc_executor": "vllm/v1/executor/uniproc_executor.py",
        "scheduler": "vllm/v1/core/sched/scheduler.py",
        "koff_runtime": "vllm/v1/spec_decode/koff_runtime.py",
        "model_runner": "vllm/v1/worker/gpu_model_runner.py",
        "draft_model": "vllm/v1/spec_decode/draft_model.py",
        "draft_proposer": "vllm/v1/spec_decode/llm_base_proposer.py",
        "sampler_backend": "vllm/v1/sample/ops/topk_topp_sampler.py",
        "adapter": (
            "research/97_composition_runtime/scripts/adapt_p4_b0_same_event.py"
        ),
        "scorer": "research/97_composition_runtime/scripts/score_p4_b0.py",
        "authorization_schema": (
            "research/97_composition_runtime/schemas/"
            "p4_b0_run_authorization_v7.schema.json"
        ),
        "authorization_validator": (
            "research/97_composition_runtime/scripts/"
            "validate_p4_b0_run_authorization_v7.py"
        ),
        "authorization_tests": (
            "research/97_composition_runtime/tests/test_p4_b0_run_authorization_v7.py"
        ),
    }


def _v8_source_paths() -> dict[str, str]:
    paths = _v7_source_paths()
    paths["authorization_schema"] = (
        "research/97_composition_runtime/schemas/p4_b0_run_authorization_v8.schema.json"
    )
    paths["authorization_validator"] = (
        "research/97_composition_runtime/scripts/validate_p4_b0_run_authorization_v8.py"
    )
    paths["authorization_tests"] = (
        "research/97_composition_runtime/tests/test_p4_b0_run_authorization_v8.py"
    )
    return paths


def _v9_source_paths() -> dict[str, str]:
    paths = _v8_source_paths()
    paths["authorization_schema"] = (
        "research/97_composition_runtime/schemas/p4_b0_run_authorization_v9.schema.json"
    )
    paths["authorization_validator"] = (
        "research/97_composition_runtime/scripts/validate_p4_b0_run_authorization_v9.py"
    )
    paths["authorization_tests"] = (
        "research/97_composition_runtime/tests/test_p4_b0_run_authorization_v9.py"
    )
    return paths


def _v10_source_paths() -> dict[str, str]:
    paths = _v9_source_paths()
    paths["authorization_schema"] = (
        "research/97_composition_runtime/schemas/"
        "p4_b0_run_authorization_v10.schema.json"
    )
    paths["authorization_validator"] = (
        "research/97_composition_runtime/scripts/"
        "validate_p4_b0_run_authorization_v10.py"
    )
    paths["authorization_tests"] = (
        "research/97_composition_runtime/tests/test_p4_b0_run_authorization_v10.py"
    )
    paths["scheduler_tests"] = "tests/v1/core/test_scheduler.py"
    paths["cohort_barrier_validator"] = (
        "research/97_composition_runtime/scripts/"
        "validate_p4_b0_chunked_prefill_cohort_barrier.py"
    )
    paths["cohort_barrier_tests"] = (
        "research/97_composition_runtime/tests/"
        "test_p4_b0_chunked_prefill_cohort_barrier.py"
    )
    return paths


def _v11_source_paths() -> dict[str, str]:
    paths = _v10_source_paths()
    paths["authorization_schema"] = (
        "research/97_composition_runtime/schemas/"
        "p4_b0_run_authorization_v11.schema.json"
    )
    paths["authorization_validator"] = (
        "research/97_composition_runtime/scripts/"
        "validate_p4_b0_run_authorization_v11.py"
    )
    paths["authorization_tests"] = (
        "research/97_composition_runtime/tests/test_p4_b0_run_authorization_v11.py"
    )
    paths["cohort_barrier_proof"] = (
        "research/97_composition_runtime/data/p4/"
        "p4_b0_chunked_prefill_cohort_barrier_design_and_cpu_proof.json"
    )
    paths["repair_validation_runner"] = (
        "research/97_composition_runtime/scripts/"
        "run_p4_b0_variable_prefill_repair_validation.py"
    )
    paths["repair_validation_tests"] = (
        "research/97_composition_runtime/tests/"
        "test_p4_b0_variable_prefill_repair_validation.py"
    )
    paths["repair_preservation_tests"] = (
        "research/97_composition_runtime/tests/"
        "test_p4_b0_variable_prefill_repair_validation_preservation.py"
    )
    return paths


def _v7_request_id_repair() -> dict[str, Any]:
    return {
        "state": "pass",
        "shared_helper": "canonicalize_p4_request_ids",
        "randomization_policy": "preserve_vllm_internal_request_id_randomization",
        "disable_randomization_allowed": False,
        "accepted_forms": [
            "exact_frozen_request_id",
            "frozen_request_id_plus_eight_lowercase_hex",
        ],
        "canonical_output": "exact_frozen_request_id",
        "consumers": ["same_event_recorder", "atomic_ingress_analyzer"],
        "fail_closed_cases": [
            "malformed_suffix",
            "unknown_request_id",
            "canonical_collision",
            "request_order_drift",
        ],
    }


def _v8_decode_work_repair() -> dict[str, Any]:
    return {
        "state": "pass",
        "measurement_currency": "S_dec",
        "measured_decode_tokens_field": "generation.max_output_tokens",
        "prefill_sampled_tokens_per_request": (PREFILL_SAMPLED_TOKENS_PER_REQUEST),
        "frontend_total_output_tokens": "generation.max_output_tokens + 1",
        "recorder_completion_tokens": "generation.max_output_tokens",
        "scorer_requested_output_tokens": (
            "len(prompt_record_ids) * generation.max_output_tokens"
        ),
        "applies_to_actions": list(ACTION_ORDERS[1]),
        "fail_closed_cases": [
            "prefill_sample_offset_drift",
            "frontend_total_output_mismatch",
            "recorder_fixed_work_underflow",
            "recorder_fixed_work_overflow",
            "capture_cell_rollover_drift",
        ],
    }


def _v9_launch_dispatch_repair() -> dict[str, Any]:
    return {
        "state": "pass",
        "shared_resolver": "_reviewed_output_path",
        "consumers": ["parent_execute_run", "child_main"],
        "authorization_path": V9_AUTHORIZATION_PATH,
        "output_path": V9_OUTPUT_PATH,
        "parent_dispatch_tested": True,
        "child_dispatch_tested": True,
        "consumed_v8_reexecution_allowed": False,
        "fail_closed_cases": [
            "unknown_authorization_path",
            "authorization_output_cross_pair",
            "child_spec_outside_reviewed_output",
            "approved_source_hash_drift",
            "fresh_output_directory_exists",
        ],
    }


def _v10_chunked_prefill_contract() -> dict[str, Any]:
    return {
        "state": "pass",
        "max_num_batched_tokens": SERVING_MAX_NUM_BATCHED_TOKENS,
        "effective_max_num_scheduled_tokens": (
            SERVING_EFFECTIVE_MAX_NUM_SCHEDULED_TOKENS
        ),
        "max_num_seqs": 32,
        "gpu_memory_utilization": SERVING_GPU_MEMORY_UTILIZATION,
        "enable_chunked_prefill": True,
        "full_microbatch_prefill_allowed": False,
        "capture_cohort_contract_id": "p4-capture-cohort-barrier-v1",
        "queue_all_exact_members_before_first_step": True,
        "hold_early_prefill_completers": True,
        "atomic_first_measured_decode": True,
        "require_every_action_and_prompt_slice": True,
        "ordinary_serving_without_marker_unchanged": True,
    }


def _v6_implementation_audit(base: Mapping[str, Any]) -> dict[str, Any]:
    checks = copy.deepcopy(base["implementation_audit"]["checks"])
    checks.extend(
        [
            {
                "code": "inprocess_engine_core_not_bound",
                "status": "pass",
                "evidence": (
                    "Every V6 boot forces VLLM_ENABLE_V1_MULTIPROCESSING=0 "
                    "before importing vLLM and asserts InprocClient after engine "
                    "construction."
                ),
            },
            {
                "code": "atomic_microbatch_ingress_missing",
                "status": "pass",
                "evidence": (
                    "The source-bound GPU-4 proof queued all eight requests before "
                    "execution, then observed one eight-request prefill step and one "
                    "eight-request q=1/OFF decode step without exclusions."
                ),
            },
        ]
    )
    return {"state": "pass", "checks": checks}


def _validate_v6_package(
    package: Mapping[str, Any], *, require_output_absent: bool
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "package_id",
        "date",
        "status",
        "prior_authorization",
        "failed_attempt",
        "atomic_ingress_proof",
        "resource_probe",
        "frozen_inputs",
        "source_artifacts",
        "run_contract",
        "measurement_repair",
        "serving_chunked_prefill_boundary",
        "implementation_audit",
        "decision",
        "claims",
        "authorizations",
        "execution_policy",
        "next_artifact",
    }
    _require(set(package) == expected_keys, "V6 authorization fields drifted")
    _require(
        package.get("schema_version") == 6
        and package.get("package_id") == V6_PACKAGE_ID
        and package.get("status")
        == "authorized_gpu4_inprocess_atomic_ingress_value_screen_only",
        "runner accepts only the reviewed V6 atomic-ingress authorization",
    )

    prior_ref = _file_reference(APPROVED_AUTHORIZATION_PATH)
    _require(
        package["prior_authorization"]
        == {
            **prior_ref,
            "disposition": "consumed_zero_capture_mixed_ingress_failure",
        },
        "V6 does not bind the consumed V5 authorization",
    )
    prior = _load_json(_repository_path(APPROVED_AUTHORIZATION_PATH))
    _require(
        prior.get("package_id") == APPROVED_PACKAGE_ID
        and prior.get("schema_version") == 5,
        "V6 prior package is not the immutable V5 authorization",
    )

    failure_ref = _file_reference(V5_FAILED_ATTEMPT_PATH)
    _require(
        package["failed_attempt"]
        == {
            **failure_ref,
            "capture_count": 0,
            "score_emitted": False,
            "preserve_without_resume": True,
        },
        "V6 does not bind the zero-capture V5 failure",
    )
    failure = _load_json(_repository_path(V5_FAILED_ATTEMPT_PATH))
    _require(
        failure.get("record_type") == "p4_b0_value_screen_execution_failure"
        and failure.get("authorization", {}).get("package_id") == APPROVED_PACKAGE_ID
        and failure.get("attempt", {}).get("captures_emitted") == 0
        and failure.get("attempt", {}).get("score_emitted") is False
        and failure.get("disposition", {}).get("requires_fresh_authorization") is True,
        "V5 failure does not support a fresh V6 authorization",
    )

    proof_ref = _file_reference(ATOMIC_INGRESS_PROOF_PATH)
    proof_auth_ref = _file_reference(ATOMIC_INGRESS_AUTHORIZATION_PATH)
    trace_ref = _file_reference(ATOMIC_INGRESS_TRACE_PATH)
    proof = _load_json(_repository_path(ATOMIC_INGRESS_PROOF_PATH))
    _require(
        proof.get("status") == "pass"
        and proof.get("scored") is False
        and proof.get("decision") == "atomic_ingress_gate_passed"
        and proof.get("trace", {}).get("engine_step_count") == 2
        and proof.get("trace", {}).get("initial_prefill_request_count") == 8
        and proof.get("trace", {}).get("first_decode_request_count") == 8
        and proof.get("trace", {}).get("first_decode_query_widths") == [1] * 8
        and proof.get("trace", {}).get("first_decode_exclusion_reasons") == []
        and proof.get("trace", {}).get("preemptions") == 0
        and proof.get("trace", {}).get("recomputed_tokens") == 0
        and proof.get("trace", {}).get("shared_target_kv_blocks") == 22090
        and proof.get("invariants", {}).get("shared_target_kv_layer_count") == 36
        and proof.get("invariants", {}).get("private_draft_kv_allocated") is False
        and proof.get("invariants", {}).get("target_draft_weight_alias_count") == 291,
        "V6 atomic-ingress proof does not close the live two-step gate",
    )
    _require(
        package["atomic_ingress_proof"]
        == {
            "authorization": proof_auth_ref,
            "result": proof_ref,
            "trace": trace_ref,
            "decision": "pass",
            "engine_core_class": INPROCESS_ENGINE_CORE_CLASS,
            "v1_multiprocessing": False,
            "initial_prefill_request_count": 8,
            "first_decode_request_count": 8,
            "first_decode_query_widths": [1] * 8,
            "first_decode_exclusion_reasons": [],
            "preemptions": 0,
            "recomputed_tokens": 0,
            "shared_target_kv_blocks": 22090,
            "shared_target_kv_layer_count": 36,
            "target_draft_weight_alias_count": 291,
            "scored": False,
        },
        "V6 atomic-ingress evidence summary drifted",
    )

    base_ref = _file_reference(BASE_AUTHORIZATION_PATH)
    base = _load_json(_repository_path(BASE_AUTHORIZATION_PATH))
    _require(
        base.get("package_id") == BASE_PACKAGE_ID and base.get("schema_version") == 2,
        "V6 base is not the immutable full V2 authorization",
    )
    expected_inputs = {
        "prompt_manifest": _file_reference(
            str(PROMPT_MANIFEST_PATH.relative_to(REPO_ROOT))
        ),
        "prompt_bundle": _file_reference(
            str(PROMPT_BUNDLE_PATH.relative_to(REPO_ROOT))
        ),
        "scorer_contract": _file_reference(
            str(SCORER_CONTRACT_PATH.relative_to(REPO_ROOT))
        ),
    }
    _require(
        package["frozen_inputs"] == expected_inputs,
        "V6 frozen prompt or scorer input drifted",
    )
    _require(
        package["resource_probe"] == prior["resource_probe"],
        "V6 resource-probe evidence differs from V5",
    )
    for reference in (
        prior["resource_probe"]["authorization"],
        prior["resource_probe"]["result"],
    ):
        _require(
            reference == _file_reference(reference["path"]),
            "V6 resource-probe source hash drifted",
        )

    source_paths = _v6_source_paths()
    references = package["source_artifacts"]
    _require(
        set(references) == set(source_paths),
        "V6 source closure is incomplete or inflated",
    )
    for role, path in source_paths.items():
        _require(
            references[role] == _file_reference(path),
            f"V6 source hash drifted for {role}",
        )

    invocation = {
        "runner_path": source_paths["matrix_runner"],
        "argv": [
            ".venv/bin/python",
            source_paths["matrix_runner"],
            "--authorization",
            V6_AUTHORIZATION_PATH,
            "--output-dir",
            V6_OUTPUT_PATH,
        ],
        "output_dir": V6_OUTPUT_PATH,
        "overwrite_allowed": False,
        "runner_exists": True,
        "launchable_now": True,
    }
    _require(
        package["run_contract"]
        == {"base_authorization": base_ref, "invocation": invocation},
        "V6 invocation differs from the create-new contract",
    )
    _require(
        package["measurement_repair"] == prior["measurement_repair"],
        "V6 measurement repair differs from V5",
    )
    _require(
        package["serving_chunked_prefill_boundary"]
        == _serving_chunked_prefill_boundary(),
        "V6 does not preserve the real-serving boundary",
    )
    _require(
        package["implementation_audit"] == _v6_implementation_audit(base),
        "V6 implementation audit drifted",
    )
    _require(
        package["decision"]
        == {
            "state": "approve",
            "scope": "gpu4_inprocess_atomic_ingress_value_screen_v6_only",
            "basis": [
                "v5_zero_capture_failure_preserved",
                "atomic_ingress_gpu_proof_passes",
                "inprocess_runner_wiring_fail_closed",
                "full_prefill_resource_repair_retained",
                "native_sampler_policy_retained",
                "current_execution_sources_hash_bound",
                "fresh_output_path_registered",
            ],
            "invalidated_by": [
                "approved_source_hash_drift",
                "fresh_output_directory_exists",
                "inprocess_engine_core_preflight_failure",
                "native_sampler_preflight_failure",
                "virtualenv_tool_preflight_failure",
                "gpu_identity_drift",
                "gpu_not_idle",
                "resource_floor_failure",
                "measurement_engine_override_drift",
                "serving_boundary_drift",
                "matrix_or_contract_drift",
            ],
        },
        "V6 decision boundary drifted",
    )
    _require(
        package["claims"]
        == {
            "prior_value_screen_attempts_performed": 4,
            "latest_prior_capture_count": 0,
            "latest_prior_score_emitted": False,
            "atomic_ingress_proof_passed": True,
            "inprocess_runner_wiring_complete": True,
            "executable_run_ready": True,
            "serving_chunked_prefill_validated": True,
            "runtime_w512_switching_implemented": False,
            "action_admitted": False,
            "performance_claim_allowed": False,
        },
        "V6 claims drifted or overstate evidence",
    )
    _require(
        package["authorizations"]
        == {
            "capture_runner_conformance_engineering": True,
            "gpu_measurement": True,
            "v6_execution": True,
            "serving_diagnosis": False,
            "p4a_engineering": False,
            "action_admission": False,
            "production_value_claim": False,
        },
        "V6 authority exceeds one GPU value screen",
    )
    _require(
        package["execution_policy"]
        == {
            "physical_gpu_index": 4,
            "physical_gpu_uuid": GPU4_UUID,
            "fallback_gpu_authorized": False,
            "engine_core_mode": "in_process",
            "engine_core_class": INPROCESS_ENGINE_CORE_CLASS,
            "v1_multiprocessing": False,
            "physical_boot_count": 9,
            "capture_count": 432,
            "create_new_output_required": True,
            "prior_output_reuse_allowed": False,
            "partial_resume_allowed": False,
            "retry_allowed": False,
            "score_grants_authority": False,
            "on_any_failure": ("stop_without_scoring_and_require_fresh_authorization"),
        },
        "V6 execution policy drifted",
    )
    _require(
        package["next_artifact"]
        == {
            "kind": "p4_b0_value_screen_result",
            "requires_complete_capture_count": 432,
            "may_authorize_p4a": False,
            "may_admit_action": False,
        },
        "V6 post-run boundary drifted",
    )
    _require(
        _repository_path(V5_FAILED_ATTEMPT_PATH).parent
        != _repository_path(V6_OUTPUT_PATH),
        "V6 output aliases the V5 attempt",
    )
    if require_output_absent:
        _require(
            not _repository_path(V6_OUTPUT_PATH).exists(),
            "approved V6 create-new output already exists",
        )
    return base


def _validate_v7_package(
    package: Mapping[str, Any], *, require_output_absent: bool
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "package_id",
        "date",
        "status",
        "prior_authorization",
        "failed_attempt",
        "request_id_repair",
        "retained_evidence",
        "frozen_inputs",
        "source_artifacts",
        "run_contract",
        "decision",
        "claims",
        "authorizations",
        "execution_policy",
        "next_artifact",
    }
    _require(set(package) == expected_keys, "V7 authorization fields drifted")
    _require(
        package.get("schema_version") == 7
        and package.get("package_id") == V7_PACKAGE_ID
        and package.get("status")
        == "authorized_gpu4_request_id_canonicalization_value_screen_only",
        "runner accepts only the reviewed V7 request-ID authorization",
    )

    prior_ref = _file_reference(V6_AUTHORIZATION_PATH)
    _require(
        package["prior_authorization"]
        == {
            **prior_ref,
            "disposition": "consumed_zero_capture_request_identity_failure",
        },
        "V7 does not bind the consumed V6 authorization",
    )
    prior = _load_json(_repository_path(V6_AUTHORIZATION_PATH))
    _require(
        prior.get("package_id") == V6_PACKAGE_ID and prior.get("schema_version") == 6,
        "V7 prior package is not the immutable V6 authorization",
    )

    failure_ref = _file_reference(V6_FAILED_ATTEMPT_PATH)
    _require(
        package["failed_attempt"]
        == {
            **failure_ref,
            "capture_count": 0,
            "score_emitted": False,
            "preserve_without_resume": True,
            "failure_scope": "request_identity_normalization_mismatch",
        },
        "V7 does not bind the zero-capture V6 request-ID failure",
    )
    failure = _load_json(_repository_path(V6_FAILED_ATTEMPT_PATH))
    _require(
        failure.get("record_type") == "p4_b0_value_screen_execution_failure"
        and failure.get("authorization", {}).get("package_id") == V6_PACKAGE_ID
        and failure.get("attempt", {}).get("captures_emitted") == 0
        and failure.get("attempt", {}).get("score_emitted") is False
        and failure.get("diagnostic", {}).get("scope")
        == "request_identity_normalization_mismatch"
        and failure.get("disposition", {}).get("v6_consumed") is True
        and failure.get("disposition", {}).get("retry_attempted") is False,
        "V6 failure does not support a fresh V7 authorization",
    )

    _require(
        package["request_id_repair"] == _v7_request_id_repair(),
        "V7 request-ID canonicalization contract drifted",
    )
    retained_evidence = {
        "atomic_ingress_proof": copy.deepcopy(prior["atomic_ingress_proof"]),
        "resource_probe": copy.deepcopy(prior["resource_probe"]),
        "serving_chunked_prefill_boundary": _serving_chunked_prefill_boundary(),
    }
    _require(
        package["retained_evidence"] == retained_evidence,
        "V7 retained proof, resource, or serving evidence drifted",
    )
    for section in ("atomic_ingress_proof", "resource_probe"):
        for value in package["retained_evidence"][section].values():
            if isinstance(value, dict) and set(value) == {"path", "sha256"}:
                _require(
                    value == _file_reference(value["path"]),
                    f"V7 retained {section} artifact hash drifted",
                )

    base_ref = _file_reference(BASE_AUTHORIZATION_PATH)
    base = _load_json(_repository_path(BASE_AUTHORIZATION_PATH))
    _require(
        base.get("package_id") == BASE_PACKAGE_ID and base.get("schema_version") == 2,
        "V7 base is not the immutable full V2 authorization",
    )
    expected_inputs = {
        "prompt_manifest": _file_reference(
            str(PROMPT_MANIFEST_PATH.relative_to(REPO_ROOT))
        ),
        "prompt_bundle": _file_reference(
            str(PROMPT_BUNDLE_PATH.relative_to(REPO_ROOT))
        ),
        "scorer_contract": _file_reference(
            str(SCORER_CONTRACT_PATH.relative_to(REPO_ROOT))
        ),
    }
    _require(
        package["frozen_inputs"] == expected_inputs,
        "V7 frozen prompt or scorer input drifted",
    )

    source_paths = _v7_source_paths()
    references = package["source_artifacts"]
    _require(
        set(references) == set(source_paths),
        "V7 source closure is incomplete or inflated",
    )
    for role, path in source_paths.items():
        _require(
            references[role] == _file_reference(path),
            f"V7 source hash drifted for {role}",
        )

    invocation = {
        "runner_path": source_paths["matrix_runner"],
        "argv": [
            ".venv/bin/python",
            source_paths["matrix_runner"],
            "--authorization",
            V7_AUTHORIZATION_PATH,
            "--output-dir",
            V7_OUTPUT_PATH,
        ],
        "output_dir": V7_OUTPUT_PATH,
        "overwrite_allowed": False,
        "runner_exists": True,
        "launchable_now": True,
    }
    _require(
        package["run_contract"]
        == {"base_authorization": base_ref, "invocation": invocation},
        "V7 invocation differs from the create-new contract",
    )
    _require(
        package["decision"]
        == {
            "state": "approve",
            "scope": "gpu4_request_id_canonicalization_value_screen_v7_only",
            "basis": [
                "v6_zero_capture_failure_preserved",
                "strict_request_id_canonicalization_tested",
                "vllm_internal_randomization_retained",
                "atomic_ingress_gpu_proof_retained",
                "full_prefill_resource_repair_retained",
                "current_execution_sources_hash_bound",
                "fresh_output_path_registered",
            ],
            "invalidated_by": [
                "approved_source_hash_drift",
                "fresh_output_directory_exists",
                "request_id_contract_drift",
                "inprocess_engine_core_preflight_failure",
                "native_sampler_preflight_failure",
                "virtualenv_tool_preflight_failure",
                "gpu_identity_drift",
                "gpu_not_idle",
                "resource_floor_failure",
                "measurement_engine_override_drift",
                "serving_boundary_drift",
                "matrix_or_contract_drift",
            ],
        },
        "V7 decision boundary drifted",
    )
    _require(
        package["claims"]
        == {
            "prior_value_screen_attempts_performed": 5,
            "latest_prior_capture_count": 0,
            "latest_prior_score_emitted": False,
            "request_id_canonicalization_tested": True,
            "internal_request_id_randomization_retained": True,
            "atomic_ingress_proof_passed": True,
            "inprocess_runner_wiring_complete": True,
            "executable_run_ready": True,
            "serving_chunked_prefill_validated": True,
            "runtime_w512_switching_implemented": False,
            "action_admitted": False,
            "performance_claim_allowed": False,
        },
        "V7 claims drifted or overstate evidence",
    )
    _require(
        package["authorizations"]
        == {
            "capture_runner_conformance_engineering": True,
            "gpu_measurement": True,
            "v7_execution": True,
            "serving_diagnosis": False,
            "p4a_engineering": False,
            "action_admission": False,
            "production_value_claim": False,
        },
        "V7 authority exceeds one GPU value screen",
    )
    _require(
        package["execution_policy"]
        == {
            "physical_gpu_index": 4,
            "physical_gpu_uuid": GPU4_UUID,
            "fallback_gpu_authorized": False,
            "engine_core_mode": "in_process",
            "engine_core_class": INPROCESS_ENGINE_CORE_CLASS,
            "v1_multiprocessing": False,
            "physical_boot_count": 9,
            "capture_count": 432,
            "create_new_output_required": True,
            "prior_output_reuse_allowed": False,
            "partial_resume_allowed": False,
            "retry_allowed": False,
            "score_grants_authority": False,
            "on_any_failure": "stop_without_scoring_and_require_fresh_authorization",
        },
        "V7 execution policy drifted",
    )
    _require(
        package["next_artifact"]
        == {
            "kind": "p4_b0_value_screen_result",
            "requires_complete_capture_count": 432,
            "may_authorize_p4a": False,
            "may_admit_action": False,
        },
        "V7 post-run boundary drifted",
    )
    _require(
        _repository_path(V6_FAILED_ATTEMPT_PATH).parent
        != _repository_path(V7_OUTPUT_PATH),
        "V7 output aliases the V6 attempt",
    )
    if require_output_absent:
        _require(
            not _repository_path(V7_OUTPUT_PATH).exists(),
            "approved V7 create-new output already exists",
        )
    return base


def _validate_v8_package(
    package: Mapping[str, Any], *, require_output_absent: bool
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "package_id",
        "date",
        "status",
        "prior_authorization",
        "failed_attempt",
        "request_id_repair",
        "decode_work_repair",
        "retained_evidence",
        "frozen_inputs",
        "source_artifacts",
        "run_contract",
        "decision",
        "claims",
        "authorizations",
        "execution_policy",
        "next_artifact",
    }
    _require(set(package) == expected_keys, "V8 authorization fields drifted")
    _require(
        package.get("schema_version") == 8
        and package.get("package_id") == V8_PACKAGE_ID
        and package.get("status")
        == "authorized_gpu4_decode_work_offset_value_screen_only",
        "runner accepts only the reviewed V8 decode-work authorization",
    )

    prior_ref = _file_reference(V7_AUTHORIZATION_PATH)
    _require(
        package["prior_authorization"]
        == {
            **prior_ref,
            "disposition": "consumed_incomplete_capture_decode_work_failure",
        },
        "V8 does not bind the consumed V7 authorization",
    )
    prior = _load_json(_repository_path(V7_AUTHORIZATION_PATH))
    _require(
        prior.get("package_id") == V7_PACKAGE_ID and prior.get("schema_version") == 7,
        "V8 prior package is not the immutable V7 authorization",
    )

    failure_ref = _file_reference(V8_FAILED_ATTEMPT_PATH)
    _require(
        package["failed_attempt"]
        == {
            **failure_ref,
            "complete_capture_count": 0,
            "incomplete_capture_count": 1,
            "score_emitted": False,
            "preserve_without_resume": True,
            "failure_scope": ("prefill_sample_vs_decode_only_fixed_work_off_by_one"),
        },
        "V8 does not bind the incomplete-capture V7 failure",
    )
    failure = _load_json(_repository_path(V8_FAILED_ATTEMPT_PATH))
    _require(
        failure.get("record_type") == "p4_b0_value_screen_execution_failure"
        and failure.get("authorization", {}).get("package_id") == V7_PACKAGE_ID
        and failure.get("attempt", {}).get("complete_captures_emitted") == 0
        and failure.get("attempt", {}).get("incomplete_captures_emitted") == 1
        and failure.get("attempt", {}).get("score_emitted") is False
        and failure.get("diagnostic", {}).get("scope")
        == "prefill_sample_vs_decode_only_fixed_work_off_by_one"
        and failure.get("disposition", {}).get("v7_consumed") is True
        and failure.get("disposition", {}).get("retry_attempted") is False,
        "V7 failure does not support a fresh V8 authorization",
    )

    _require(
        package["request_id_repair"] == _v7_request_id_repair(),
        "V8 request-ID canonicalization contract drifted",
    )
    _require(
        package["decode_work_repair"] == _v8_decode_work_repair(),
        "V8 decode-work offset contract drifted",
    )
    _require(
        package["retained_evidence"] == prior["retained_evidence"],
        "V8 retained proof, resource, or serving evidence drifted",
    )
    for section in ("atomic_ingress_proof", "resource_probe"):
        for value in package["retained_evidence"][section].values():
            if isinstance(value, dict) and set(value) == {"path", "sha256"}:
                _require(
                    value == _file_reference(value["path"]),
                    f"V8 retained {section} artifact hash drifted",
                )

    base_ref = _file_reference(BASE_AUTHORIZATION_PATH)
    base = _load_json(_repository_path(BASE_AUTHORIZATION_PATH))
    _require(
        base.get("package_id") == BASE_PACKAGE_ID and base.get("schema_version") == 2,
        "V8 base is not the immutable full V2 authorization",
    )
    expected_inputs = {
        "prompt_manifest": _file_reference(
            str(PROMPT_MANIFEST_PATH.relative_to(REPO_ROOT))
        ),
        "prompt_bundle": _file_reference(
            str(PROMPT_BUNDLE_PATH.relative_to(REPO_ROOT))
        ),
        "scorer_contract": _file_reference(
            str(SCORER_CONTRACT_PATH.relative_to(REPO_ROOT))
        ),
    }
    _require(
        package["frozen_inputs"] == expected_inputs,
        "V8 frozen prompt or scorer input drifted",
    )

    source_paths = _v8_source_paths()
    references = package["source_artifacts"]
    _require(
        set(references) == set(source_paths),
        "V8 source closure is incomplete or inflated",
    )
    for role, path in source_paths.items():
        _require(
            references[role] == _file_reference(path),
            f"V8 source hash drifted for {role}",
        )

    invocation = {
        "runner_path": source_paths["matrix_runner"],
        "argv": [
            ".venv/bin/python",
            source_paths["matrix_runner"],
            "--authorization",
            V8_AUTHORIZATION_PATH,
            "--output-dir",
            V8_OUTPUT_PATH,
        ],
        "output_dir": V8_OUTPUT_PATH,
        "overwrite_allowed": False,
        "runner_exists": True,
        "launchable_now": True,
    }
    _require(
        package["run_contract"]
        == {"base_authorization": base_ref, "invocation": invocation},
        "V8 invocation differs from the create-new contract",
    )
    _require(
        package["decision"]
        == {
            "state": "approve",
            "scope": "gpu4_decode_work_offset_value_screen_v8_only",
            "basis": [
                "v7_incomplete_capture_failure_preserved",
                "prefill_sample_offset_repaired_and_tested",
                "off_k4_w512_capture_rollover_tested",
                "strict_request_id_canonicalization_retained",
                "atomic_ingress_gpu_proof_retained",
                "full_prefill_resource_repair_retained",
                "current_execution_sources_hash_bound",
                "fresh_output_path_registered",
            ],
            "invalidated_by": [
                "approved_source_hash_drift",
                "fresh_output_directory_exists",
                "decode_work_contract_drift",
                "request_id_contract_drift",
                "inprocess_engine_core_preflight_failure",
                "native_sampler_preflight_failure",
                "virtualenv_tool_preflight_failure",
                "gpu_identity_drift",
                "gpu_not_idle",
                "resource_floor_failure",
                "measurement_engine_override_drift",
                "serving_boundary_drift",
                "matrix_or_contract_drift",
            ],
        },
        "V8 decision boundary drifted",
    )
    _require(
        package["claims"]
        == {
            "prior_value_screen_attempts_performed": 6,
            "latest_prior_complete_capture_count": 0,
            "latest_prior_incomplete_capture_count": 1,
            "latest_prior_score_emitted": False,
            "decode_work_offset_tested": True,
            "all_action_capture_rollover_tested": True,
            "request_id_canonicalization_tested": True,
            "internal_request_id_randomization_retained": True,
            "atomic_ingress_proof_passed": True,
            "inprocess_runner_wiring_complete": True,
            "executable_run_ready": True,
            "serving_chunked_prefill_validated": True,
            "runtime_w512_switching_implemented": False,
            "action_admitted": False,
            "performance_claim_allowed": False,
        },
        "V8 claims drifted or overstate evidence",
    )
    _require(
        package["authorizations"]
        == {
            "capture_runner_conformance_engineering": True,
            "gpu_measurement": True,
            "v8_execution": True,
            "serving_diagnosis": False,
            "p4a_engineering": False,
            "action_admission": False,
            "production_value_claim": False,
        },
        "V8 authority exceeds one GPU value screen",
    )
    _require(
        package["execution_policy"]
        == {
            "physical_gpu_index": 4,
            "physical_gpu_uuid": GPU4_UUID,
            "fallback_gpu_authorized": False,
            "engine_core_mode": "in_process",
            "engine_core_class": INPROCESS_ENGINE_CORE_CLASS,
            "v1_multiprocessing": False,
            "physical_boot_count": 9,
            "capture_count": 432,
            "create_new_output_required": True,
            "prior_output_reuse_allowed": False,
            "partial_resume_allowed": False,
            "retry_allowed": False,
            "score_grants_authority": False,
            "on_any_failure": "stop_without_scoring_and_require_fresh_authorization",
        },
        "V8 execution policy drifted",
    )
    _require(
        package["next_artifact"]
        == {
            "kind": "p4_b0_value_screen_result",
            "requires_complete_capture_count": 432,
            "may_authorize_p4a": False,
            "may_admit_action": False,
        },
        "V8 post-run boundary drifted",
    )
    _require(
        _repository_path(V8_FAILED_ATTEMPT_PATH).parent
        != _repository_path(V8_OUTPUT_PATH),
        "V8 output aliases the V7 attempt",
    )
    if require_output_absent:
        _require(
            not _repository_path(V8_OUTPUT_PATH).exists(),
            "approved V8 create-new output already exists",
        )
    return base


def _v10_probe_evidence() -> dict[str, Any]:
    return {
        "authorization": _file_reference(CHUNKED_PREFILL_PROBE_AUTHORIZATION_PATH),
        "preparation": _file_reference(CHUNKED_PREFILL_PROBE_PREPARATION_PATH),
        "result": _file_reference(CHUNKED_PREFILL_PROBE_RESULT_PATH),
        "trace": _file_reference(CHUNKED_PREFILL_PROBE_TRACE_PATH),
        "status": "pass",
        "scored": False,
        "authorization_consumed": True,
        "action_id": "target-matching-k4",
        "regimes": ["R4", "R5", "R5cot"],
        "physical_gpu_index": 4,
        "physical_gpu_uuid": GPU4_UUID,
        "configured_max_num_batched_tokens": SERVING_MAX_NUM_BATCHED_TOKENS,
        "effective_max_num_scheduled_tokens": (
            SERVING_EFFECTIVE_MAX_NUM_SCHEDULED_TOKENS
        ),
        "gpu_memory_utilization": SERVING_GPU_MEMORY_UTILIZATION,
        "cohort_count": 3,
        "complete_request_count": 24,
        "canonical_request_ids": True,
        "pure_first_measured_decode": True,
        "zero_preemption": True,
        "zero_recomputation": True,
        "zero_invalid_spec_tokens": True,
        "shared_target_kv_block_capacity": 24527,
        "minimum_shared_target_kv_blocks": MINIMUM_SHARED_KV_BLOCKS,
        "actual_cuda_graph_memory_bytes": 589365248,
        "trace_record_count": 40,
    }


def _validate_v10_probe_evidence(evidence: Mapping[str, Any]) -> None:
    _require(
        evidence == _v10_probe_evidence(),
        "V10 chunked-prefill probe binding drifted",
    )
    probe_authorization = _load_json(
        _repository_path(CHUNKED_PREFILL_PROBE_AUTHORIZATION_PATH)
    )
    preparation = _load_json(_repository_path(CHUNKED_PREFILL_PROBE_PREPARATION_PATH))
    result = _load_json(_repository_path(CHUNKED_PREFILL_PROBE_RESULT_PATH))
    probe_contract = probe_authorization.get("run_contract", {})
    _require(
        probe_authorization.get("package_id")
        == "p4-b0-chunked-prefill-probe-authorization-v5"
        and probe_authorization.get("decision", {}).get("decision") == "authorize"
        and probe_authorization.get("execution_policy", {}).get("score_output") is False
        and probe_contract.get("action", {}).get("action_id") == evidence["action_id"]
        and probe_contract.get("workload", {}).get("regimes") == evidence["regimes"]
        and probe_contract.get("engine", {}).get("gpu_memory_utilization")
        == SERVING_GPU_MEMORY_UTILIZATION,
        "V10 source probe authorization is not the consumed V5 authority",
    )
    _require(
        preparation.get("authorization_sha256") == evidence["authorization"]["sha256"]
        and preparation.get("physical_boot_count") == 1
        and preparation.get("scored") is False,
        "V10 source probe preparation is not the exact one-shot binding",
    )
    cohorts = result.get("cohorts", ())
    events = result.get("target_events", ())
    request_ids = [
        row.get("request_id")
        for event in events
        for row in event.get("request_steps", ())
    ]
    expected_request_ids = {
        f"{regime}-s0-p{prompt_index:03d}"
        for regime in evidence["regimes"]
        for prompt_index in range(8)
    }
    _require(
        result.get("status") == "pass"
        and result.get("scored") is False
        and result.get("authorization_consumed") is True
        and result.get("action_id") == evidence["action_id"]
        and result.get("regimes") == evidence["regimes"]
        and result.get("gpu") == {"physical_index": 4, "uuid": GPU4_UUID}
        and result.get("claims")
        == {
            "chunked_prefill_barrier_gpu_proven": True,
            "performance_claim_allowed": False,
            "v10_authorized": False,
        }
        and result.get("authorizations")
        == {
            "action_admission": False,
            "p4a_engineering": False,
            "production_value_claim": False,
            "v10_value_screen": False,
        }
        and len(cohorts) == 3
        and all(
            cohort.get("state") == "complete"
            and cohort.get("finished_request_count") == 8
            and cohort.get("pure_first_measured_decode") is True
            for cohort in cohorts
        )
        and len(events) == 3
        and all(
            event.get("pure_decode") is True
            and event.get("quality")
            == {
                "invalid_spec_tokens": 0,
                "preemptions": 0,
                "recomputed_tokens": 0,
            }
            for event in events
        )
        and len(request_ids) == 24
        and set(request_ids) == expected_request_ids,
        "V10 source probe result lost a complete or clean cohort",
    )
    resources = result.get("resources", {})
    _require(
        resources.get("configured_max_num_batched_tokens")
        == SERVING_MAX_NUM_BATCHED_TOKENS
        and resources.get("effective_max_num_scheduled_tokens")
        == SERVING_EFFECTIVE_MAX_NUM_SCHEDULED_TOKENS
        and resources.get("shared_target_kv_block_capacity") == 24527
        and resources.get("minimum_shared_target_kv_blocks") == MINIMUM_SHARED_KV_BLOCKS
        and resources.get("actual_cuda_graph_memory_bytes") == 589365248,
        "V10 source probe resource evidence drifted",
    )
    trace_path = _repository_path(CHUNKED_PREFILL_PROBE_TRACE_PATH)
    try:
        trace_records = sum(
            bool(line.strip())
            for line in trace_path.read_text(encoding="utf-8").splitlines()
        )
    except OSError as exc:
        raise P4RunnerError(f"cannot read V10 source probe trace: {exc}") from exc
    _require(trace_records == 40, "V10 source probe trace record count drifted")


def _v11_consumed_attempt_evidence() -> dict[str, Any]:
    return {
        "failure": _file_reference(V11_FAILED_ATTEMPT_PATH),
        "capture_manifest": _file_reference(V11_CAPTURE_MANIFEST_PATH),
        "output_dir": V10_OUTPUT_PATH,
        "complete_capture_count": 72,
        "empty_placeholder_count": 1,
        "adapted_rounds_emitted": False,
        "score_emitted": False,
        "preserve_without_resume_or_reuse": True,
    }


def _validate_v11_consumed_attempt(evidence: Mapping[str, Any]) -> None:
    _require(
        evidence == _v11_consumed_attempt_evidence(),
        "V11 consumed V10 attempt binding drifted",
    )
    failure = _load_json(_repository_path(V11_FAILED_ATTEMPT_PATH))
    manifest = _load_json(_repository_path(V11_CAPTURE_MANIFEST_PATH))
    attempt = failure.get("attempt", {})
    disposition = failure.get("disposition", {})
    capture_evidence = failure.get("capture_evidence", {})
    _require(
        failure.get("record_type") == "p4_b0_value_screen_execution_failure"
        and failure.get("authorization", {}).get("package_id") == V10_PACKAGE_ID
        and attempt.get("physical_gpu_index") == 4
        and attempt.get("gpu_model_executed") is True
        and attempt.get("complete_captures_emitted") == 72
        and attempt.get("empty_capture_placeholders") == 1
        and attempt.get("failed_capture_id") == "capture-b1-p2-k4-r8-s0-r1"
        and attempt.get("failed_regime") == "R8"
        and attempt.get("adapted_rounds_emitted") is False
        and attempt.get("score_emitted") is False
        and failure.get("diagnostic", {}).get("scope")
        == "k4_variable_width_pure_prefill_step0_evidence"
        and disposition.get("v10_consumed") is True
        and disposition.get("requires_fresh_authorization") is True
        and disposition.get("retry_attempted") is False
        and disposition.get("partial_resume_attempted") is False
        and disposition.get("scoring_allowed") is False
        and failure.get("output", {}).get("path") == V10_OUTPUT_PATH
        and failure.get("output", {}).get("preserve_without_overwrite_or_resume")
        is True,
        "V10 failure does not support a fresh non-reusing V11 review",
    )
    _require(
        capture_evidence.get("path") == V11_CAPTURE_MANIFEST_PATH
        and capture_evidence.get("sha256") == evidence["capture_manifest"]["sha256"]
        and capture_evidence.get("complete_capture_count") == 72
        and capture_evidence.get("empty_placeholder_count") == 1,
        "V10 failure lost its capture-manifest binding",
    )
    counts = manifest.get("counts", {})
    invariants = manifest.get("invariants", {})
    captures = manifest.get("captures", ())
    placeholders = manifest.get("empty_placeholders", ())
    _require(
        manifest.get("status") == "immutable_partial_attempt"
        and manifest.get("authorization", {}).get("package_id") == V10_PACKAGE_ID
        and manifest.get("output_dir") == V10_OUTPUT_PATH
        and counts.get("raw_capture_files") == 73
        and counts.get("complete_captures") == 72
        and counts.get("empty_placeholders") == 1
        and len(captures) == 72
        and all(
            row.get("complete") is True and row.get("scored") is False
            for row in captures
        )
        and len(placeholders) == 1
        and placeholders[0].get("capture_id") == "capture-b1-p2-k4-r8-s0-r1"
        and invariants
        == {
            "adapted_rounds_absent": True,
            "all_captures_unscored": True,
            "all_nonempty_captures_complete": True,
            "preserve_without_overwrite_resume_or_reuse": True,
            "score_absent": True,
        },
        "V10 immutable capture boundary drifted",
    )


def _v11_repair_validation_evidence() -> dict[str, Any]:
    exact_common = {
        "draft_step0_query_width": None,
        "draft_step0_num_tokens": 2116,
        "draft_step0_batch_size": 16,
        "draft_output_shape": [16, 4],
        "proposal_called": True,
        "draft_step0_runtime_mode": "NONE",
        "draft_chain_runtime_mode": "PIECEWISE",
        "produced_draft_width": 4,
    }
    return {
        "authorization": _file_reference(VARIABLE_PREFILL_REPAIR_AUTHORIZATION_PATH),
        "audit": _file_reference(VARIABLE_PREFILL_REPAIR_AUDIT_PATH),
        "aggregate": _file_reference(VARIABLE_PREFILL_REPAIR_AGGREGATE_PATH),
        "isolated_case": _file_reference(VARIABLE_PREFILL_REPAIR_ISOLATED_CASE_PATH),
        "isolated_trace": _file_reference(VARIABLE_PREFILL_REPAIR_ISOLATED_TRACE_PATH),
        "transition_case": _file_reference(
            VARIABLE_PREFILL_REPAIR_TRANSITION_CASE_PATH
        ),
        "transition_trace": _file_reference(
            VARIABLE_PREFILL_REPAIR_TRANSITION_TRACE_PATH
        ),
        "status": "gpu_cases_passed_parent_aggregate_rejected",
        "authorization_consumed": True,
        "scored": False,
        "parent_aggregate_passed": False,
        "parent_rejection": {
            "classification": "observer_cardinality_bug",
            "caused_gpu_case_failure": False,
            "expected_count": 2,
            "observed_transition_armed_prefill_count": 9,
        },
        "cases": [
            {
                "case_id": "isolated-r8",
                "physical_gpu_index": 0,
                "physical_gpu_uuid": ("GPU-4938442e-5508-9249-0fa6-37baa1985703"),
                "completed_cohorts": ["r8-first"],
                "measured_event_count": 610,
                "trace_record_count": 612,
                "armed_prefill_observation_count": 1,
                "exact_r8_observation": {
                    **exact_common,
                    "engine_step_index": 0,
                },
            },
            {
                "case_id": "r5cot-to-r8",
                "physical_gpu_index": 1,
                "physical_gpu_uuid": ("GPU-ba39f4f0-61fe-34ca-c1af-ffe565b70923"),
                "completed_cohorts": ["r5cot-last", "r8-first"],
                "measured_event_count": 1273,
                "trace_record_count": 1289,
                "armed_prefill_observation_count": 9,
                "exact_r8_observation": {
                    **exact_common,
                    "engine_step_index": 632,
                },
            },
        ],
    }


def _is_exact_repaired_r8_observation(
    observation: Mapping[str, Any], *, engine_step_index: int
) -> bool:
    metadata = observation.get("metadata", {})
    validated = observation.get("validated_evidence", {})
    return (
        observation.get("draft_step0_query_width") is None
        and observation.get("draft_step0_num_tokens") == 2116
        and observation.get("draft_step0_batch_size") == 16
        and observation.get("draft_output_shape") == [16, 4]
        and observation.get("proposal_called") is True
        and observation.get("draft_step0_runtime_mode") == "NONE"
        and observation.get("draft_chain_runtime_mode") == "PIECEWISE"
        and metadata.get("engine_step_index") == engine_step_index
        and metadata.get("capture_cohort_arm") is True
        and metadata.get("pure_decode") is False
        and metadata.get("decode_req_ids") == []
        and metadata.get("next_action_id") == "target-matching-k4"
        and metadata.get("preemptions") == 0
        and metadata.get("recomputed_tokens") == 0
        and validated
        == {
            "draft_dispatched": True,
            "draft_step0_batch_size": 16,
            "draft_step0_num_tokens": 2116,
            "draft_step0_query_width": None,
            "produced_draft_width": 4,
        }
    )


def _validate_v11_repair_validation(evidence: Mapping[str, Any]) -> None:
    expected = _v11_repair_validation_evidence()
    _require(
        evidence == expected,
        "V11 variable-prefill repair evidence binding drifted",
    )
    audit = _load_json(_repository_path(VARIABLE_PREFILL_REPAIR_AUDIT_PATH))
    aggregate = _load_json(_repository_path(VARIABLE_PREFILL_REPAIR_AGGREGATE_PATH))
    disposition = audit.get("disposition", {})
    parent_rejection = audit.get("parent_rejection", {})
    _require(
        audit.get("status") == expected["status"]
        and audit.get("authorization_consumed") is True
        and audit.get("scored") is False
        and disposition.get("gpu_repair_validation_passed") is True
        and disposition.get("isolated_r8_passed") is True
        and disposition.get("r5cot_to_r8_passed") is True
        and disposition.get("parent_aggregate_passed") is False
        and disposition.get("output_reuse_allowed") is False
        and disposition.get("resume_allowed") is False
        and disposition.get("retry_allowed") is False
        and disposition.get("value_screen_retry_authorized") is False
        and disposition.get("v11_authorized") is False
        and disposition.get("performance_claim_allowed") is False
        and parent_rejection.get("classification") == "observer_cardinality_bug"
        and parent_rejection.get("caused_gpu_case_failure") is False
        and parent_rejection.get("expected_count") == 2
        and parent_rejection.get("observed_transition_armed_prefill_count") == 9,
        "V11 repair audit overstates or lost its case-level boundary",
    )
    _require(
        aggregate.get("status") == "complete"
        and aggregate.get("authorization_consumed") is True
        and aggregate.get("scored") is False
        and aggregate.get("classification", {}).get("conclusion")
        == "original_failure_not_reproduced"
        and aggregate.get("claims")
        == {
            "action_admission_authorized": False,
            "diagnostic_only": True,
            "performance_claim_allowed": False,
            "value_screen_retry_authorized": False,
        },
        "V11 source repair aggregate drifted",
    )

    case_paths = {
        "isolated-r8": VARIABLE_PREFILL_REPAIR_ISOLATED_CASE_PATH,
        "r5cot-to-r8": VARIABLE_PREFILL_REPAIR_TRANSITION_CASE_PATH,
    }
    for expected_case in expected["cases"]:
        case = _load_json(_repository_path(case_paths[expected_case["case_id"]]))
        armed = case.get("instrumentation", {}).get("armed_prefill_evidence", ())
        exact_matches = [
            row
            for row in armed
            if _is_exact_repaired_r8_observation(
                row,
                engine_step_index=expected_case["exact_r8_observation"][
                    "engine_step_index"
                ],
            )
        ]
        recorder = case.get("recorder", {})
        trace = case.get("trace", {})
        resources = case.get("resources", {})
        _require(
            case.get("status") == "completed_without_invariant_failure"
            and case.get("active_stage_at_exit") == "complete"
            and case.get("authorization_consumed") is True
            and case.get("scored") is False
            and case.get("primary_exception") is None
            and case.get("shutdown_exception") is None
            and case.get("case_id") == expected_case["case_id"]
            and case.get("gpu")
            == {
                "physical_index": expected_case["physical_gpu_index"],
                "uuid": expected_case["physical_gpu_uuid"],
            }
            and case.get("completed_cohorts") == expected_case["completed_cohorts"]
            and len(armed) == expected_case["armed_prefill_observation_count"]
            and len(exact_matches) == 1
            and recorder.get("closed") is True
            and recorder.get("event_count") == expected_case["measured_event_count"]
            and trace.get("present") is True
            and trace.get("record_count") == expected_case["trace_record_count"]
            and resources.get("configured_max_num_batched_tokens")
            == SERVING_MAX_NUM_BATCHED_TOKENS
            and resources.get("effective_max_num_scheduled_tokens")
            == SERVING_EFFECTIVE_MAX_NUM_SCHEDULED_TOKENS
            and resources.get("minimum_shared_target_kv_blocks")
            == MINIMUM_SHARED_KV_BLOCKS
            and resources.get("shared_target_kv_block_capacity") == 24527
            and all(value is False for value in case.get("authorizations", {}).values())
            and case.get("claims", {}).get("performance_claim_allowed") is False
            and case.get("claims", {}).get("score_eligible") is False,
            f"V11 repaired GPU case drifted for {expected_case['case_id']}",
        )


def _validate_v11_package(
    package: Mapping[str, Any], *, require_output_absent: bool
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "package_id",
        "date",
        "status",
        "prior_authorization",
        "consumed_attempt",
        "variable_prefill_repair_validation",
        "chunked_prefill_probe",
        "chunked_prefill_contract",
        "source_artifacts",
        "run_contract",
        "decision",
        "claims",
        "authorizations",
        "execution_policy",
        "next_artifact",
    }
    _require(set(package) == expected_keys, "V11 authorization fields drifted")
    _require(
        package.get("schema_version") == 11
        and package.get("package_id") == V11_PACKAGE_ID
        and package.get("status")
        == "authorized_gpu4_repaired_variable_prefill_value_screen_only",
        "runner accepts only the reviewed V11 value-screen authority",
    )

    prior_ref = _file_reference(V10_AUTHORIZATION_PATH)
    _require(
        package["prior_authorization"]
        == {
            **prior_ref,
            "disposition": "consumed_variable_prefill_evidence_failure_attempt",
        },
        "V11 does not bind the consumed V10 authorization",
    )
    prior = _load_json(_repository_path(V10_AUTHORIZATION_PATH))
    _require(
        prior.get("package_id") == V10_PACKAGE_ID and prior.get("schema_version") == 10,
        "V11 prior package is not the immutable V10 authorization",
    )
    _validate_v11_consumed_attempt(package["consumed_attempt"])
    _validate_v11_repair_validation(package["variable_prefill_repair_validation"])
    _validate_v10_probe_evidence(package["chunked_prefill_probe"])
    _require(
        package["chunked_prefill_contract"] == _v10_chunked_prefill_contract(),
        "V11 bounded chunked-prefill contract drifted",
    )

    base_ref = _file_reference(BASE_AUTHORIZATION_PATH)
    base = _load_json(_repository_path(BASE_AUTHORIZATION_PATH))
    _require(
        base.get("package_id") == BASE_PACKAGE_ID and base.get("schema_version") == 2,
        "V11 base is not the immutable full V2 authorization",
    )
    expected_inputs = {
        "prompt_manifest": _file_reference(
            str(PROMPT_MANIFEST_PATH.relative_to(REPO_ROOT))
        ),
        "prompt_bundle": _file_reference(
            str(PROMPT_BUNDLE_PATH.relative_to(REPO_ROOT))
        ),
        "scorer_contract": _file_reference(
            str(SCORER_CONTRACT_PATH.relative_to(REPO_ROOT))
        ),
    }
    retained_contract = _load_json(_repository_path(V8_AUTHORIZATION_PATH))
    _require(
        retained_contract.get("package_id") == V8_PACKAGE_ID
        and retained_contract.get("frozen_inputs") == expected_inputs,
        "V11 frozen prompt or scorer input drifted",
    )

    source_paths = _v11_source_paths()
    references = package["source_artifacts"]
    _require(
        set(references) == set(source_paths),
        "V11 source closure is incomplete or inflated",
    )
    for role, path in source_paths.items():
        _require(
            references[role] == _file_reference(path),
            f"V11 source hash drifted for {role}",
        )

    invocation = {
        "runner_path": source_paths["matrix_runner"],
        "argv": [
            ".venv/bin/python",
            source_paths["matrix_runner"],
            "--authorization",
            V11_AUTHORIZATION_PATH,
            "--output-dir",
            V11_OUTPUT_PATH,
        ],
        "output_dir": V11_OUTPUT_PATH,
        "overwrite_allowed": False,
        "runner_exists": True,
        "launchable_now": True,
    }
    _require(
        package["run_contract"]
        == {"base_authorization": base_ref, "invocation": invocation},
        "V11 invocation differs from the create-new contract",
    )
    _require(
        package["decision"]
        == {
            "state": "approve",
            "scope": "gpu4_repaired_variable_prefill_value_screen_v11_only",
            "basis": [
                "v10_partial_attempt_preserved",
                "v10_output_reuse_forbidden",
                "variable_prefill_runtime_evidence_repaired",
                "isolated_r8_gpu_case_passed",
                "r5cot_to_r8_gpu_case_passed",
                "parent_rejection_classified_as_observer_cardinality",
                "bounded_chunked_prefill_cohort_barrier_retained",
                "current_execution_sources_hash_bound",
                "fresh_output_path_registered",
            ],
            "invalidated_by": [
                "approved_source_hash_drift",
                "fresh_output_directory_exists",
                "v10_attempt_artifact_drift",
                "repair_validation_artifact_drift",
                "v5_probe_artifact_drift",
                "chunked_prefill_contract_drift",
                "full_prefill_geometry_reenabled",
                "launch_dispatch_contract_drift",
                "decode_work_contract_drift",
                "request_id_contract_drift",
                "inprocess_engine_core_preflight_failure",
                "native_sampler_preflight_failure",
                "virtualenv_tool_preflight_failure",
                "gpu_identity_drift",
                "gpu_not_idle",
                "resource_floor_failure",
                "cohort_abort_or_incomplete_release",
                "matrix_or_contract_drift",
            ],
        },
        "V11 decision boundary drifted",
    )
    _require(
        package["claims"]
        == {
            "prior_value_screen_attempts_performed": 9,
            "latest_prior_complete_capture_count": 72,
            "latest_prior_empty_placeholder_count": 1,
            "latest_prior_gpu_executed": True,
            "latest_prior_score_emitted": False,
            "prior_outputs_preserved": True,
            "prior_outputs_reusable": False,
            "full_prefill_geometry_rejected": True,
            "chunked_prefill_cohort_cpu_proven": True,
            "chunked_prefill_k4_gpu_probe_passed": True,
            "variable_prefill_runtime_evidence_repaired": True,
            "isolated_r8_gpu_case_passed": True,
            "r5cot_to_r8_gpu_case_passed": True,
            "repair_parent_aggregate_passed": False,
            "repair_parent_rejection_is_observer_only": True,
            "launch_dispatch_tested": True,
            "decode_work_offset_tested": True,
            "all_action_capture_rollover_tested": True,
            "request_id_canonicalization_tested": True,
            "internal_request_id_randomization_retained": True,
            "inprocess_runner_wiring_complete": True,
            "executable_run_ready": True,
            "runtime_w512_switching_implemented": False,
            "action_admitted": False,
            "performance_claim_allowed": False,
        },
        "V11 claims drifted or overstate evidence",
    )
    _require(
        package["authorizations"]
        == {
            "capture_runner_conformance_engineering": True,
            "gpu_measurement": True,
            "v11_execution": True,
            "value_screen_scoring": True,
            "p4a_engineering": False,
            "action_admission": False,
            "production_value_claim": False,
        },
        "V11 authority exceeds one GPU value screen",
    )
    _require(
        package["execution_policy"]
        == {
            "physical_gpu_index": 4,
            "physical_gpu_uuid": GPU4_UUID,
            "fallback_gpu_authorized": False,
            "engine_core_mode": "in_process",
            "engine_core_class": INPROCESS_ENGINE_CORE_CLASS,
            "v1_multiprocessing": False,
            "physical_boot_count": 9,
            "capture_count": 432,
            "max_num_batched_tokens": SERVING_MAX_NUM_BATCHED_TOKENS,
            "effective_max_num_scheduled_tokens": (
                SERVING_EFFECTIVE_MAX_NUM_SCHEDULED_TOKENS
            ),
            "gpu_memory_utilization": SERVING_GPU_MEMORY_UTILIZATION,
            "capture_cohort_barrier_required": True,
            "create_new_output_required": True,
            "prior_output_reuse_allowed": False,
            "repair_validation_output_reuse_allowed": False,
            "partial_resume_allowed": False,
            "retry_allowed": False,
            "score_grants_authority": False,
            "on_any_failure": ("stop_without_scoring_and_require_fresh_authorization"),
        },
        "V11 execution policy drifted",
    )
    _require(
        package["next_artifact"]
        == {
            "kind": "p4_b0_value_screen_result",
            "requires_complete_capture_count": 432,
            "may_authorize_p4a": False,
            "may_admit_action": False,
        },
        "V11 post-run boundary drifted",
    )
    _require(
        _repository_path(V11_OUTPUT_PATH) != _repository_path(V10_OUTPUT_PATH)
        and _repository_path(V11_OUTPUT_PATH)
        != _repository_path(VARIABLE_PREFILL_REPAIR_AGGREGATE_PATH).parent,
        "V11 output aliases a consumed evidence directory",
    )
    if require_output_absent:
        _require(
            not _repository_path(V11_OUTPUT_PATH).exists(),
            "approved V11 create-new output already exists",
        )
    return base


def _validate_v10_package(
    package: Mapping[str, Any], *, require_output_absent: bool
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "package_id",
        "date",
        "status",
        "prior_authorization",
        "failed_attempt",
        "chunked_prefill_probe",
        "chunked_prefill_contract",
        "source_artifacts",
        "run_contract",
        "decision",
        "claims",
        "authorizations",
        "execution_policy",
        "next_artifact",
    }
    _require(set(package) == expected_keys, "V10 authorization fields drifted")
    _require(
        package.get("schema_version") == 10
        and package.get("package_id") == V10_PACKAGE_ID
        and package.get("status")
        == "authorized_gpu4_chunked_prefill_cohort_value_screen_only",
        "runner accepts only the reviewed V10 cohort value-screen authority",
    )

    prior_ref = _file_reference(V9_AUTHORIZATION_PATH)
    _require(
        package["prior_authorization"]
        == {
            **prior_ref,
            "disposition": "consumed_full_prefill_activation_oom_attempt",
        },
        "V10 does not bind the consumed V9 authorization",
    )
    prior = _load_json(_repository_path(V9_AUTHORIZATION_PATH))
    _require(
        prior.get("package_id") == V9_PACKAGE_ID and prior.get("schema_version") == 9,
        "V10 prior package is not the immutable V9 authorization",
    )

    failure_ref = _file_reference(V10_FAILED_ATTEMPT_PATH)
    _require(
        package["failed_attempt"]
        == {
            **failure_ref,
            "complete_capture_count": 8,
            "incomplete_capture_count": 0,
            "gpu_executed": True,
            "score_emitted": False,
            "preserve_without_resume": True,
            "failure_scope": "full_prefill_transient_activation_hbm_underbound",
        },
        "V10 does not bind the V9 full-prefill failure",
    )
    failure = _load_json(_repository_path(V10_FAILED_ATTEMPT_PATH))
    _require(
        failure.get("record_type") == "p4_b0_value_screen_execution_failure"
        and failure.get("authorization", {}).get("package_id") == V9_PACKAGE_ID
        and failure.get("attempt", {}).get("complete_captures_emitted") == 8
        and failure.get("attempt", {}).get("incomplete_captures_emitted") == 0
        and failure.get("attempt", {}).get("gpu_model_executed") is True
        and failure.get("attempt", {}).get("score_emitted") is False
        and failure.get("diagnostic", {}).get("scope")
        == "full_prefill_transient_activation_hbm_underbound"
        and failure.get("disposition", {}).get("v9_consumed") is True
        and failure.get("disposition", {}).get("retry_attempted") is False
        and failure.get("disposition", {}).get("partial_resume_attempted") is False,
        "V9 failure does not support a fresh bounded V10 authorization",
    )

    _validate_v10_probe_evidence(package["chunked_prefill_probe"])
    _require(
        package["chunked_prefill_contract"] == _v10_chunked_prefill_contract(),
        "V10 bounded chunked-prefill contract drifted",
    )

    base_ref = _file_reference(BASE_AUTHORIZATION_PATH)
    base = _load_json(_repository_path(BASE_AUTHORIZATION_PATH))
    _require(
        base.get("package_id") == BASE_PACKAGE_ID and base.get("schema_version") == 2,
        "V10 base is not the immutable full V2 authorization",
    )
    expected_inputs = {
        "prompt_manifest": _file_reference(
            str(PROMPT_MANIFEST_PATH.relative_to(REPO_ROOT))
        ),
        "prompt_bundle": _file_reference(
            str(PROMPT_BUNDLE_PATH.relative_to(REPO_ROOT))
        ),
        "scorer_contract": _file_reference(
            str(SCORER_CONTRACT_PATH.relative_to(REPO_ROOT))
        ),
    }
    retained_contract = _load_json(_repository_path(V8_AUTHORIZATION_PATH))
    _require(
        retained_contract.get("package_id") == V8_PACKAGE_ID
        and retained_contract.get("frozen_inputs") == expected_inputs,
        "V10 frozen prompt or scorer input drifted",
    )

    source_paths = _v10_source_paths()
    references = package["source_artifacts"]
    _require(
        set(references) == set(source_paths),
        "V10 source closure is incomplete or inflated",
    )
    for role, path in source_paths.items():
        _require(
            references[role] == _file_reference(path),
            f"V10 source hash drifted for {role}",
        )

    invocation = {
        "runner_path": source_paths["matrix_runner"],
        "argv": [
            ".venv/bin/python",
            source_paths["matrix_runner"],
            "--authorization",
            V10_AUTHORIZATION_PATH,
            "--output-dir",
            V10_OUTPUT_PATH,
        ],
        "output_dir": V10_OUTPUT_PATH,
        "overwrite_allowed": False,
        "runner_exists": True,
        "launchable_now": True,
    }
    _require(
        package["run_contract"]
        == {"base_authorization": base_ref, "invocation": invocation},
        "V10 invocation differs from the create-new contract",
    )
    _require(
        package["decision"]
        == {
            "state": "approve",
            "scope": "gpu4_chunked_prefill_cohort_value_screen_v10_only",
            "basis": [
                "v9_full_prefill_oom_preserved",
                "full_prefill_geometry_rejected",
                "bounded_chunked_prefill_cohort_barrier_cpu_proven",
                "target_matching_k4_chunked_prefill_gpu_probe_passed",
                "strict_request_id_canonicalization_retained",
                "decode_work_offset_retained",
                "current_execution_sources_hash_bound",
                "fresh_output_path_registered",
            ],
            "invalidated_by": [
                "approved_source_hash_drift",
                "fresh_output_directory_exists",
                "v5_probe_artifact_drift",
                "chunked_prefill_contract_drift",
                "full_prefill_geometry_reenabled",
                "launch_dispatch_contract_drift",
                "decode_work_contract_drift",
                "request_id_contract_drift",
                "inprocess_engine_core_preflight_failure",
                "native_sampler_preflight_failure",
                "virtualenv_tool_preflight_failure",
                "gpu_identity_drift",
                "gpu_not_idle",
                "resource_floor_failure",
                "cohort_abort_or_incomplete_release",
                "matrix_or_contract_drift",
            ],
        },
        "V10 decision boundary drifted",
    )
    _require(
        package["claims"]
        == {
            "prior_value_screen_attempts_performed": 8,
            "latest_prior_complete_capture_count": 8,
            "latest_prior_incomplete_capture_count": 0,
            "latest_prior_gpu_executed": True,
            "latest_prior_score_emitted": False,
            "full_prefill_geometry_rejected": True,
            "chunked_prefill_cohort_cpu_proven": True,
            "chunked_prefill_k4_gpu_probe_passed": True,
            "launch_dispatch_tested": True,
            "decode_work_offset_tested": True,
            "all_action_capture_rollover_tested": True,
            "request_id_canonicalization_tested": True,
            "internal_request_id_randomization_retained": True,
            "inprocess_runner_wiring_complete": True,
            "executable_run_ready": True,
            "runtime_w512_switching_implemented": False,
            "action_admitted": False,
            "performance_claim_allowed": False,
        },
        "V10 claims drifted or overstate evidence",
    )
    _require(
        package["authorizations"]
        == {
            "capture_runner_conformance_engineering": True,
            "gpu_measurement": True,
            "v10_execution": True,
            "value_screen_scoring": True,
            "p4a_engineering": False,
            "action_admission": False,
            "production_value_claim": False,
        },
        "V10 authority exceeds one GPU value screen",
    )
    _require(
        package["execution_policy"]
        == {
            "physical_gpu_index": 4,
            "physical_gpu_uuid": GPU4_UUID,
            "fallback_gpu_authorized": False,
            "engine_core_mode": "in_process",
            "engine_core_class": INPROCESS_ENGINE_CORE_CLASS,
            "v1_multiprocessing": False,
            "physical_boot_count": 9,
            "capture_count": 432,
            "max_num_batched_tokens": SERVING_MAX_NUM_BATCHED_TOKENS,
            "effective_max_num_scheduled_tokens": (
                SERVING_EFFECTIVE_MAX_NUM_SCHEDULED_TOKENS
            ),
            "gpu_memory_utilization": SERVING_GPU_MEMORY_UTILIZATION,
            "capture_cohort_barrier_required": True,
            "create_new_output_required": True,
            "prior_output_reuse_allowed": False,
            "partial_resume_allowed": False,
            "retry_allowed": False,
            "score_grants_authority": False,
            "on_any_failure": ("stop_without_scoring_and_require_fresh_authorization"),
        },
        "V10 execution policy drifted",
    )
    _require(
        package["next_artifact"]
        == {
            "kind": "p4_b0_value_screen_result",
            "requires_complete_capture_count": 432,
            "may_authorize_p4a": False,
            "may_admit_action": False,
        },
        "V10 post-run boundary drifted",
    )
    _require(
        _repository_path(V10_FAILED_ATTEMPT_PATH).parent
        != _repository_path(V10_OUTPUT_PATH),
        "V10 output aliases the V9 attempt",
    )
    if require_output_absent:
        _require(
            not _repository_path(V10_OUTPUT_PATH).exists(),
            "approved V10 create-new output already exists",
        )
    return base


def _validate_v9_package(
    package: Mapping[str, Any], *, require_output_absent: bool
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "package_id",
        "date",
        "status",
        "prior_authorization",
        "failed_attempt",
        "launch_dispatch_repair",
        "source_artifacts",
        "run_contract",
        "decision",
        "claims",
        "authorizations",
        "execution_policy",
        "next_artifact",
    }
    _require(set(package) == expected_keys, "V9 authorization fields drifted")
    _require(
        package.get("schema_version") == 9
        and package.get("package_id") == V9_PACKAGE_ID
        and package.get("status")
        == "authorized_gpu4_launch_dispatch_repair_value_screen_only",
        "runner accepts only the reviewed V9 launch-dispatch authorization",
    )

    prior_ref = _file_reference(V8_AUTHORIZATION_PATH)
    _require(
        package["prior_authorization"]
        == {
            **prior_ref,
            "disposition": "consumed_pre_gpu_launch_dispatch_refusal",
        },
        "V9 does not bind the consumed V8 authorization",
    )
    prior = _load_json(_repository_path(V8_AUTHORIZATION_PATH))
    _require(
        prior.get("package_id") == V8_PACKAGE_ID and prior.get("schema_version") == 8,
        "V9 prior package is not the immutable V8 authorization",
    )

    failure_ref = _file_reference(V9_FAILED_ATTEMPT_PATH)
    _require(
        package["failed_attempt"]
        == {
            **failure_ref,
            "complete_capture_count": 0,
            "incomplete_capture_count": 0,
            "gpu_executed": False,
            "score_emitted": False,
            "preserve_without_resume": True,
            "failure_scope": "reviewed_authorization_path_dispatch_omission",
        },
        "V9 does not bind the pre-GPU V8 launch refusal",
    )
    failure = _load_json(_repository_path(V9_FAILED_ATTEMPT_PATH))
    _require(
        failure.get("record_type") == "p4_b0_value_screen_execution_failure"
        and failure.get("authorization", {}).get("package_id") == V8_PACKAGE_ID
        and failure.get("attempt", {}).get("complete_captures_emitted") == 0
        and failure.get("attempt", {}).get("incomplete_captures_emitted") == 0
        and failure.get("attempt", {}).get("gpu_model_executed") is False
        and failure.get("attempt", {}).get("score_emitted") is False
        and failure.get("diagnostic", {}).get("scope")
        == "reviewed_authorization_path_dispatch_omission"
        and failure.get("disposition", {}).get("v8_consumed") is True
        and failure.get("disposition", {}).get("retry_attempted") is False
        and failure.get("output", {}).get("runner_output_created") is False,
        "V8 refusal does not support a fresh V9 authorization",
    )

    _require(
        package["launch_dispatch_repair"] == _v9_launch_dispatch_repair(),
        "V9 parent/child launch-dispatch contract drifted",
    )
    _require(
        prior["request_id_repair"] == _v7_request_id_repair(),
        "V9 prior request-ID contract drifted",
    )
    _require(
        prior["decode_work_repair"] == _v8_decode_work_repair(),
        "V9 prior decode-work contract drifted",
    )
    for section in ("atomic_ingress_proof", "resource_probe"):
        for value in prior["retained_evidence"][section].values():
            if isinstance(value, dict) and set(value) == {"path", "sha256"}:
                _require(
                    value == _file_reference(value["path"]),
                    f"V9 retained {section} artifact hash drifted",
                )

    base_ref = _file_reference(BASE_AUTHORIZATION_PATH)
    base = _load_json(_repository_path(BASE_AUTHORIZATION_PATH))
    _require(
        base.get("package_id") == BASE_PACKAGE_ID and base.get("schema_version") == 2,
        "V9 base is not the immutable full V2 authorization",
    )
    expected_inputs = {
        "prompt_manifest": _file_reference(
            str(PROMPT_MANIFEST_PATH.relative_to(REPO_ROOT))
        ),
        "prompt_bundle": _file_reference(
            str(PROMPT_BUNDLE_PATH.relative_to(REPO_ROOT))
        ),
        "scorer_contract": _file_reference(
            str(SCORER_CONTRACT_PATH.relative_to(REPO_ROOT))
        ),
    }
    _require(
        prior["frozen_inputs"] == expected_inputs,
        "V9 frozen prompt or scorer input drifted",
    )

    source_paths = _v9_source_paths()
    references = package["source_artifacts"]
    _require(
        set(references) == set(source_paths),
        "V9 source closure is incomplete or inflated",
    )
    for role, path in source_paths.items():
        _require(
            references[role] == _file_reference(path),
            f"V9 source hash drifted for {role}",
        )

    invocation = {
        "runner_path": source_paths["matrix_runner"],
        "argv": [
            ".venv/bin/python",
            source_paths["matrix_runner"],
            "--authorization",
            V9_AUTHORIZATION_PATH,
            "--output-dir",
            V9_OUTPUT_PATH,
        ],
        "output_dir": V9_OUTPUT_PATH,
        "overwrite_allowed": False,
        "runner_exists": True,
        "launchable_now": True,
    }
    _require(
        package["run_contract"]
        == {"base_authorization": base_ref, "invocation": invocation},
        "V9 invocation differs from the create-new contract",
    )
    _require(
        package["decision"]
        == {
            "state": "approve",
            "scope": "gpu4_launch_dispatch_repair_value_screen_v9_only",
            "basis": [
                "v8_pre_gpu_dispatch_failure_preserved",
                "shared_parent_child_dispatch_resolver",
                "parent_child_execution_path_regression_tested",
                "v8_decode_work_repair_retained",
                "strict_request_id_canonicalization_retained",
                "atomic_ingress_gpu_proof_retained",
                "full_prefill_resource_repair_retained",
                "current_execution_sources_hash_bound",
                "fresh_output_path_registered",
            ],
            "invalidated_by": [
                "approved_source_hash_drift",
                "fresh_output_directory_exists",
                "launch_dispatch_contract_drift",
                "decode_work_contract_drift",
                "request_id_contract_drift",
                "inprocess_engine_core_preflight_failure",
                "native_sampler_preflight_failure",
                "virtualenv_tool_preflight_failure",
                "gpu_identity_drift",
                "gpu_not_idle",
                "resource_floor_failure",
                "measurement_engine_override_drift",
                "serving_boundary_drift",
                "matrix_or_contract_drift",
            ],
        },
        "V9 decision boundary drifted",
    )
    _require(
        package["claims"]
        == {
            "prior_value_screen_attempts_performed": 7,
            "latest_prior_complete_capture_count": 0,
            "latest_prior_incomplete_capture_count": 0,
            "latest_prior_gpu_executed": False,
            "latest_prior_score_emitted": False,
            "launch_dispatch_repair_tested": True,
            "decode_work_offset_tested": True,
            "all_action_capture_rollover_tested": True,
            "request_id_canonicalization_tested": True,
            "internal_request_id_randomization_retained": True,
            "atomic_ingress_proof_passed": True,
            "inprocess_runner_wiring_complete": True,
            "executable_run_ready": True,
            "serving_chunked_prefill_validated": True,
            "runtime_w512_switching_implemented": False,
            "action_admitted": False,
            "performance_claim_allowed": False,
        },
        "V9 claims drifted or overstate evidence",
    )
    _require(
        package["authorizations"]
        == {
            "capture_runner_conformance_engineering": True,
            "gpu_measurement": True,
            "v9_execution": True,
            "serving_diagnosis": False,
            "p4a_engineering": False,
            "action_admission": False,
            "production_value_claim": False,
        },
        "V9 authority exceeds one GPU value screen",
    )
    _require(
        package["execution_policy"]
        == {
            "physical_gpu_index": 4,
            "physical_gpu_uuid": GPU4_UUID,
            "fallback_gpu_authorized": False,
            "engine_core_mode": "in_process",
            "engine_core_class": INPROCESS_ENGINE_CORE_CLASS,
            "v1_multiprocessing": False,
            "physical_boot_count": 9,
            "capture_count": 432,
            "create_new_output_required": True,
            "prior_output_reuse_allowed": False,
            "partial_resume_allowed": False,
            "retry_allowed": False,
            "score_grants_authority": False,
            "on_any_failure": "stop_without_scoring_and_require_fresh_authorization",
        },
        "V9 execution policy drifted",
    )
    _require(
        package["next_artifact"]
        == {
            "kind": "p4_b0_value_screen_result",
            "requires_complete_capture_count": 432,
            "may_authorize_p4a": False,
            "may_admit_action": False,
        },
        "V9 post-run boundary drifted",
    )
    _require(
        _repository_path(V9_FAILED_ATTEMPT_PATH).parent
        != _repository_path(V9_OUTPUT_PATH),
        "V9 output aliases the V8 attempt",
    )
    if require_output_absent:
        _require(
            not _repository_path(V9_OUTPUT_PATH).exists(),
            "approved V9 create-new output already exists",
        )
    return base


def resolve_authorization_package(
    package: Mapping[str, Any], *, require_output_absent: bool = True
) -> dict[str, Any]:
    """Materialize the frozen V2 contract under an additive retry review."""
    if package.get("package_id") == V11_PACKAGE_ID:
        authorization = copy.deepcopy(
            _validate_v11_package(
                package,
                require_output_absent=require_output_absent,
            )
        )
        authorization = apply_chunked_prefill_engine_contract(authorization)
        authorization["run_contract"]["environment"][V1_MULTIPROCESSING_ENV] = (
            V1_MULTIPROCESSING_VALUE
        )
        authorization["schema_version"] = 11
        authorization["package_id"] = V11_PACKAGE_ID
        authorization["status"] = (
            "authorized_gpu4_repaired_variable_prefill_value_screen_only"
        )
        authorization["run_contract"]["invocation"] = copy.deepcopy(
            package["run_contract"]["invocation"]
        )
        retained_contract = _load_json(_repository_path(V8_AUTHORIZATION_PATH))
        for field in (
            "request_id_repair",
            "decode_work_repair",
            "frozen_inputs",
        ):
            authorization[field] = copy.deepcopy(retained_contract[field])
        for field in (
            "prior_authorization",
            "consumed_attempt",
            "variable_prefill_repair_validation",
            "chunked_prefill_probe",
            "chunked_prefill_contract",
            "source_artifacts",
            "decision",
            "claims",
            "authorizations",
            "execution_policy",
            "next_artifact",
        ):
            authorization[field] = copy.deepcopy(package[field])
        return authorization
    if package.get("package_id") == V10_PACKAGE_ID:
        authorization = copy.deepcopy(
            _validate_v10_package(
                package,
                require_output_absent=require_output_absent,
            )
        )
        authorization = apply_chunked_prefill_engine_contract(authorization)
        authorization["run_contract"]["environment"][V1_MULTIPROCESSING_ENV] = (
            V1_MULTIPROCESSING_VALUE
        )
        authorization["schema_version"] = 10
        authorization["package_id"] = V10_PACKAGE_ID
        authorization["status"] = (
            "authorized_gpu4_chunked_prefill_cohort_value_screen_only"
        )
        authorization["run_contract"]["invocation"] = copy.deepcopy(
            package["run_contract"]["invocation"]
        )
        retained_contract = _load_json(_repository_path(V8_AUTHORIZATION_PATH))
        for field in (
            "request_id_repair",
            "decode_work_repair",
            "frozen_inputs",
        ):
            authorization[field] = copy.deepcopy(retained_contract[field])
        for field in (
            "prior_authorization",
            "failed_attempt",
            "chunked_prefill_probe",
            "chunked_prefill_contract",
            "source_artifacts",
            "decision",
            "claims",
            "authorizations",
            "execution_policy",
            "next_artifact",
        ):
            authorization[field] = copy.deepcopy(package[field])
        return authorization
    if package.get("package_id") == V9_PACKAGE_ID:
        authorization = copy.deepcopy(
            _validate_v9_package(package, require_output_absent=require_output_absent)
        )
        authorization = apply_full_prefill_engine_contract(authorization)
        authorization["run_contract"]["engine"]["gpu_memory_utilization"] = (
            MEASUREMENT_GPU_MEMORY_UTILIZATION
        )
        authorization["run_contract"]["environment"][V1_MULTIPROCESSING_ENV] = (
            V1_MULTIPROCESSING_VALUE
        )
        authorization["schema_version"] = 9
        authorization["package_id"] = V9_PACKAGE_ID
        authorization["status"] = (
            "authorized_gpu4_launch_dispatch_repair_value_screen_only"
        )
        authorization["run_contract"]["invocation"] = copy.deepcopy(
            package["run_contract"]["invocation"]
        )
        prior = _load_json(_repository_path(V8_AUTHORIZATION_PATH))
        for field in (
            "request_id_repair",
            "decode_work_repair",
            "retained_evidence",
            "frozen_inputs",
        ):
            authorization[field] = copy.deepcopy(prior[field])
        for field in (
            "launch_dispatch_repair",
            "source_artifacts",
            "decision",
            "claims",
            "authorizations",
            "execution_policy",
            "next_artifact",
        ):
            authorization[field] = copy.deepcopy(package[field])
        return authorization
    if package.get("package_id") == V8_PACKAGE_ID:
        authorization = copy.deepcopy(
            _validate_v8_package(package, require_output_absent=require_output_absent)
        )
        authorization = apply_full_prefill_engine_contract(authorization)
        authorization["run_contract"]["engine"]["gpu_memory_utilization"] = (
            MEASUREMENT_GPU_MEMORY_UTILIZATION
        )
        authorization["run_contract"]["environment"][V1_MULTIPROCESSING_ENV] = (
            V1_MULTIPROCESSING_VALUE
        )
        authorization["schema_version"] = 8
        authorization["package_id"] = V8_PACKAGE_ID
        authorization["status"] = "authorized_gpu4_decode_work_offset_value_screen_only"
        authorization["run_contract"]["invocation"] = copy.deepcopy(
            package["run_contract"]["invocation"]
        )
        for field in (
            "request_id_repair",
            "decode_work_repair",
            "retained_evidence",
            "frozen_inputs",
            "source_artifacts",
            "decision",
            "claims",
            "authorizations",
            "execution_policy",
            "next_artifact",
        ):
            authorization[field] = copy.deepcopy(package[field])
        return authorization
    if package.get("package_id") == V7_PACKAGE_ID:
        authorization = copy.deepcopy(
            _validate_v7_package(package, require_output_absent=require_output_absent)
        )
        authorization = apply_full_prefill_engine_contract(authorization)
        authorization["run_contract"]["engine"]["gpu_memory_utilization"] = (
            MEASUREMENT_GPU_MEMORY_UTILIZATION
        )
        authorization["run_contract"]["environment"][V1_MULTIPROCESSING_ENV] = (
            V1_MULTIPROCESSING_VALUE
        )
        authorization["schema_version"] = 7
        authorization["package_id"] = V7_PACKAGE_ID
        authorization["status"] = (
            "authorized_gpu4_request_id_canonicalization_value_screen_only"
        )
        authorization["run_contract"]["invocation"] = copy.deepcopy(
            package["run_contract"]["invocation"]
        )
        for field in (
            "request_id_repair",
            "retained_evidence",
            "frozen_inputs",
            "source_artifacts",
            "decision",
            "claims",
            "authorizations",
            "execution_policy",
            "next_artifact",
        ):
            authorization[field] = copy.deepcopy(package[field])
        return authorization
    if package.get("package_id") == V6_PACKAGE_ID:
        authorization = copy.deepcopy(
            _validate_v6_package(package, require_output_absent=require_output_absent)
        )
        authorization = apply_full_prefill_engine_contract(authorization)
        authorization["run_contract"]["engine"]["gpu_memory_utilization"] = (
            MEASUREMENT_GPU_MEMORY_UTILIZATION
        )
        authorization["run_contract"]["environment"][V1_MULTIPROCESSING_ENV] = (
            V1_MULTIPROCESSING_VALUE
        )
        authorization["schema_version"] = 6
        authorization["package_id"] = V6_PACKAGE_ID
        authorization["status"] = (
            "authorized_gpu4_inprocess_atomic_ingress_value_screen_only"
        )
        authorization["run_contract"]["invocation"] = copy.deepcopy(
            package["run_contract"]["invocation"]
        )
        for field in (
            "atomic_ingress_proof",
            "resource_probe",
            "frozen_inputs",
            "source_artifacts",
            "measurement_repair",
            "serving_chunked_prefill_boundary",
            "implementation_audit",
            "decision",
            "claims",
            "authorizations",
            "execution_policy",
            "next_artifact",
        ):
            authorization[field] = copy.deepcopy(package[field])
        return authorization
    if package.get("package_id") == V4_PACKAGE_ID:
        authorization = copy.deepcopy(
            _validate_retry_package(
                package, require_output_absent=require_output_absent
            )
        )
        authorization["schema_version"] = 4
        authorization["package_id"] = V4_PACKAGE_ID
        authorization["status"] = "authorized_gpu4_native_sampler_retry_only"
        authorization["run_contract"]["invocation"] = copy.deepcopy(
            package["run_contract"]["invocation"]
        )
        for role in ("matrix_runner", "runner_tests", "conformance_tests"):
            authorization["source_artifacts"][role] = copy.deepcopy(
                package["source_artifacts"][role]
            )
        return authorization
    if package.get("package_id") != APPROVED_PACKAGE_ID:
        return copy.deepcopy(dict(package))
    authorization = copy.deepcopy(
        _validate_v5_package(package, require_output_absent=require_output_absent)
    )
    authorization = apply_full_prefill_engine_contract(authorization)
    authorization["run_contract"]["engine"]["gpu_memory_utilization"] = (
        MEASUREMENT_GPU_MEMORY_UTILIZATION
    )
    authorization["schema_version"] = 5
    authorization["package_id"] = APPROVED_PACKAGE_ID
    authorization["status"] = "authorized_gpu4_full_prefill_resource_repair_retry_only"
    authorization["run_contract"]["invocation"] = copy.deepcopy(
        package["run_contract"]["invocation"]
    )
    for role in (
        "matrix_runner",
        "runner_tests",
        "conformance_tests",
        "validator",
    ):
        authorization["source_artifacts"][role] = copy.deepcopy(
            package["source_artifacts"][role]
        )
    for field in (
        "measurement_repair",
        "serving_chunked_prefill_boundary",
        "resource_probe",
    ):
        authorization[field] = copy.deepcopy(package[field])
    return authorization


def logical_weight_version(authorization: Mapping[str, Any]) -> str:
    """Derive the action-independent target-matching weight identifier."""
    model = authorization["run_contract"]["model"]
    identity = {
        "draft_weight_path": model["draft_weight_path"],
        "model_id": model["model_id"],
        "revision": model["revision"],
        "target_quantization": model["target_quantization"],
    }
    return LOGICAL_VERSION_PREFIX + _canonical_sha256(identity)


def apply_full_prefill_engine_contract(
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a copy with the measurement-only prefill budget applied."""
    effective = copy.deepcopy(dict(authorization))
    engine = effective["run_contract"]["engine"]
    engine["max_num_batched_tokens"] = FULL_PREFILL_MAX_NUM_BATCHED_TOKENS
    engine["enable_chunked_prefill"] = True
    return effective


def apply_chunked_prefill_engine_contract(
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a copy with the bounded cohort-barrier engine contract."""
    effective = copy.deepcopy(dict(authorization))
    engine = effective["run_contract"]["engine"]
    engine["max_num_batched_tokens"] = SERVING_MAX_NUM_BATCHED_TOKENS
    engine["enable_chunked_prefill"] = True
    engine["gpu_memory_utilization"] = SERVING_GPU_MEMORY_UTILIZATION
    return effective


def full_prefill_budget_evidence(
    manifest: Mapping[str, Any],
    prompt_rows: Mapping[str, Mapping[str, Any]],
    engine: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove every frozen prompt microbatch fits one scheduler event."""
    max_num_seqs = engine.get("max_num_seqs")
    max_num_batched_tokens = engine.get("max_num_batched_tokens")
    _require(type(max_num_seqs) is int, "full-prefill max_num_seqs is invalid")
    _require(
        type(max_num_batched_tokens) is int,
        "full-prefill max_num_batched_tokens is invalid",
    )
    _require(
        engine.get("enable_chunked_prefill") is True,
        "measurement runner must retain chunked prefill",
    )
    slot_reserve = max_num_seqs * SPECULATIVE_SLOT_RESERVE_PER_SEQUENCE
    effective_budget = max_num_batched_tokens - slot_reserve

    manifest_rows = manifest.get("prompts")
    _require(isinstance(manifest_rows, list), "prompt manifest rows are invalid")
    manifest_by_id = {row["record_id"]: row for row in manifest_rows}
    _require(
        set(manifest_by_id) == set(prompt_rows),
        "full-prefill prompt rows differ from the manifest",
    )
    for record_id, row in manifest_by_id.items():
        tokens = prompt_rows[record_id].get("token_ids")
        _require(
            isinstance(tokens, list) and len(tokens) == row.get("token_count"),
            f"full-prefill token count drifted for {record_id}",
        )

    regime_maxima: dict[str, int] = {}
    for regime in manifest["prompt_plan"]["regimes"]:
        regime_id = regime["regime_id"]
        batch = regime["batch"]
        _require(
            type(batch) is int and batch > 0 and 32 % batch == 0,
            f"full-prefill batch is invalid for {regime_id}",
        )
        totals = []
        for content_seed in CONTENT_SEEDS:
            prompt_ids = _prompt_group(manifest, regime_id, content_seed)
            for start in range(0, len(prompt_ids), batch):
                totals.append(
                    sum(
                        len(prompt_rows[record_id]["token_ids"])
                        for record_id in prompt_ids[start : start + batch]
                    )
                )
        regime_maxima[regime_id] = max(totals)

    max_prompt_tokens = max(regime_maxima.values())
    _require(
        max_prompt_tokens == EXPECTED_MAX_MICROBATCH_PROMPT_TOKENS,
        "frozen maximum prompt microbatch token count drifted",
    )
    _require(
        max_num_batched_tokens == FULL_PREFILL_MAX_NUM_BATCHED_TOKENS,
        "measurement runner full-prefill allocation drifted",
    )
    _require(
        effective_budget >= max_prompt_tokens,
        "full-prefill effective scheduler budget is too small",
    )
    return {
        "policy": "measurement_only_full_microbatch_prefill",
        "chunked_prefill_enabled": True,
        "max_num_batched_tokens": max_num_batched_tokens,
        "speculative_slot_reserve": slot_reserve,
        "effective_scheduler_token_budget": effective_budget,
        "max_microbatch_prompt_tokens": max_prompt_tokens,
        "headroom_tokens": effective_budget - max_prompt_tokens,
        "regime_max_microbatch_prompt_tokens": regime_maxima,
    }


def chunked_prefill_budget_evidence(
    manifest: Mapping[str, Any],
    prompt_rows: Mapping[str, Mapping[str, Any]],
    engine: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove the frozen matrix uses the bounded cohort-barrier geometry."""
    max_num_seqs = engine.get("max_num_seqs")
    max_num_batched_tokens = engine.get("max_num_batched_tokens")
    _require(type(max_num_seqs) is int, "chunked-prefill max_num_seqs is invalid")
    _require(
        type(max_num_batched_tokens) is int,
        "chunked-prefill max_num_batched_tokens is invalid",
    )
    _require(
        engine.get("enable_chunked_prefill") is True,
        "V10 must enable bounded chunked prefill",
    )
    _require(
        engine.get("gpu_memory_utilization") == SERVING_GPU_MEMORY_UTILIZATION,
        "V10 chunked-prefill GPU memory utilization drifted",
    )
    slot_reserve = max_num_seqs * SPECULATIVE_SLOT_RESERVE_PER_SEQUENCE
    effective_budget = max_num_batched_tokens - slot_reserve

    manifest_rows = manifest.get("prompts")
    _require(isinstance(manifest_rows, list), "prompt manifest rows are invalid")
    manifest_by_id = {row["record_id"]: row for row in manifest_rows}
    _require(
        set(manifest_by_id) == set(prompt_rows),
        "chunked-prefill prompt rows differ from the manifest",
    )
    for record_id, row in manifest_by_id.items():
        tokens = prompt_rows[record_id].get("token_ids")
        _require(
            isinstance(tokens, list) and len(tokens) == row.get("token_count"),
            f"chunked-prefill token count drifted for {record_id}",
        )

    regime_maxima: dict[str, int] = {}
    for regime in manifest["prompt_plan"]["regimes"]:
        regime_id = regime["regime_id"]
        batch = regime["batch"]
        _require(
            type(batch) is int and batch > 0 and 32 % batch == 0,
            f"chunked-prefill batch is invalid for {regime_id}",
        )
        totals = []
        for content_seed in CONTENT_SEEDS:
            prompt_ids = _prompt_group(manifest, regime_id, content_seed)
            for start in range(0, len(prompt_ids), batch):
                totals.append(
                    sum(
                        len(prompt_rows[record_id]["token_ids"])
                        for record_id in prompt_ids[start : start + batch]
                    )
                )
        regime_maxima[regime_id] = max(totals)

    max_prompt_tokens = max(regime_maxima.values())
    _require(
        max_prompt_tokens == EXPECTED_MAX_MICROBATCH_PROMPT_TOKENS,
        "frozen maximum prompt microbatch token count drifted",
    )
    _require(
        max_num_batched_tokens == SERVING_MAX_NUM_BATCHED_TOKENS
        and effective_budget == SERVING_EFFECTIVE_MAX_NUM_SCHEDULED_TOKENS,
        "V10 bounded chunked-prefill scheduler budget drifted",
    )
    _require(
        max_prompt_tokens > effective_budget,
        "V10 unexpectedly permits full-microbatch prefill",
    )
    return {
        "policy": "measurement_only_bounded_chunked_prefill_cohort_barrier",
        "chunked_prefill_enabled": True,
        "full_microbatch_prefill_allowed": False,
        "capture_cohort_barrier_required": True,
        "max_num_batched_tokens": max_num_batched_tokens,
        "speculative_slot_reserve": slot_reserve,
        "effective_scheduler_token_budget": effective_budget,
        "max_microbatch_prompt_tokens": max_prompt_tokens,
        "regime_max_microbatch_prompt_tokens": regime_maxima,
    }


def validate_preparation_contract(authorization: Mapping[str, Any]) -> None:
    """Validate the immutable matrix inputs without granting GPU authority."""
    run = authorization.get("run_contract", {})
    matrix = run.get("matrix", {})
    resources = authorization.get("resource_gates", {})
    engine = run.get("engine", {})
    _require(run.get("screen_id") == SCREEN_ID, "authorization screen id drifted")
    _require(
        authorization.get("authorizations", {}).get(
            "capture_runner_conformance_engineering"
        )
        is True,
        "capture-runner conformance engineering is not authorized",
    )
    _require(
        matrix.get("physical_boot_count") == 9
        and matrix.get("cells_per_boot") == 48
        and matrix.get("raw_capture_count") == 432,
        "authorization matrix does not close to 9 boots and 432 captures",
    )
    observed_orders = {
        row["block_id"]: tuple(row["action_order"])
        for row in matrix.get("boot_blocks", ())
    }
    _require(observed_orders == ACTION_ORDERS, "authorization action orders drifted")
    _require(
        resources.get("minimum_launch_capacity_blocks") == MINIMUM_SHARED_KV_BLOCKS,
        "authorization shared-KV floor drifted",
    )
    _require(
        resources.get("on_violation") == "stop_without_scoring",
        "authorization resource failure policy drifted",
    )
    if authorization.get("package_id") in {V10_PACKAGE_ID, V11_PACKAGE_ID}:
        _require(
            engine.get("max_num_batched_tokens") == SERVING_MAX_NUM_BATCHED_TOKENS
            and engine.get("enable_chunked_prefill") is True
            and engine.get("gpu_memory_utilization") == SERVING_GPU_MEMORY_UTILIZATION,
            "V10 does not bind the bounded chunked-prefill engine",
        )
    else:
        _require(
            engine.get("max_num_batched_tokens") == FULL_PREFILL_MAX_NUM_BATCHED_TOKENS
            and engine.get("enable_chunked_prefill") is True,
            "authorization does not bind the measurement-only full-prefill budget",
        )
    if authorization.get("package_id") in {
        V6_PACKAGE_ID,
        V7_PACKAGE_ID,
        V8_PACKAGE_ID,
        V9_PACKAGE_ID,
        V10_PACKAGE_ID,
        V11_PACKAGE_ID,
    }:
        _require(
            run.get("environment", {}).get(V1_MULTIPROCESSING_ENV)
            == V1_MULTIPROCESSING_VALUE,
            "authorization does not bind in-process EngineCore mode",
        )


def _validate_v6_execution_authority(authorization: Mapping[str, Any]) -> None:
    _require(
        authorization.get("schema_version") == 6
        and authorization.get("package_id") == V6_PACKAGE_ID
        and authorization.get("status")
        == "authorized_gpu4_inprocess_atomic_ingress_value_screen_only",
        "runner accepts only the reviewed V6 atomic-ingress authorization",
    )
    validate_preparation_contract(authorization)
    _require(
        authorization.get("authorizations")
        == {
            "capture_runner_conformance_engineering": True,
            "gpu_measurement": True,
            "v6_execution": True,
            "serving_diagnosis": False,
            "p4a_engineering": False,
            "action_admission": False,
            "production_value_claim": False,
        },
        "V6 authorities drifted beyond the value screen",
    )
    _require(
        authorization.get("claims", {}).get("executable_run_ready") is True
        and authorization.get("claims", {}).get("inprocess_runner_wiring_complete")
        is True
        and authorization.get("claims", {}).get("atomic_ingress_proof_passed") is True,
        "V6 does not claim complete atomic-ingress run readiness",
    )
    _require(
        authorization.get("decision", {}).get("state") == "approve"
        and authorization.get("decision", {}).get("scope")
        == "gpu4_inprocess_atomic_ingress_value_screen_v6_only",
        "V6 decision does not approve the narrow GPU-4 screen",
    )
    _require(
        authorization.get("execution_policy", {}).get("physical_gpu_index") == 4
        and authorization["execution_policy"].get("physical_gpu_uuid") == GPU4_UUID
        and authorization["execution_policy"].get("fallback_gpu_authorized") is False
        and authorization["execution_policy"].get("engine_core_class")
        == INPROCESS_ENGINE_CORE_CLASS
        and authorization["execution_policy"].get("v1_multiprocessing") is False
        and authorization["execution_policy"].get("retry_allowed") is False,
        "V6 execution policy lost its GPU or EngineCore boundary",
    )
    references = authorization.get("source_artifacts", {})
    source_paths = _v6_source_paths()
    _require(
        set(references) == set(source_paths),
        "V6 execution source closure is incomplete or inflated",
    )
    for role, path in source_paths.items():
        _require(
            references[role] == _file_reference(path),
            f"V6 execution source hash drifted for {role}",
        )

    base = _load_json(_repository_path(BASE_AUTHORIZATION_PATH))
    expected_run = copy.deepcopy(base["run_contract"])
    expected_run["engine"]["max_num_batched_tokens"] = (
        FULL_PREFILL_MAX_NUM_BATCHED_TOKENS
    )
    expected_run["engine"]["enable_chunked_prefill"] = True
    expected_run["engine"]["gpu_memory_utilization"] = (
        MEASUREMENT_GPU_MEMORY_UTILIZATION
    )
    expected_run["environment"][V1_MULTIPROCESSING_ENV] = V1_MULTIPROCESSING_VALUE
    expected_run["invocation"] = {
        "runner_path": source_paths["matrix_runner"],
        "argv": [
            ".venv/bin/python",
            source_paths["matrix_runner"],
            "--authorization",
            V6_AUTHORIZATION_PATH,
            "--output-dir",
            V6_OUTPUT_PATH,
        ],
        "output_dir": V6_OUTPUT_PATH,
        "overwrite_allowed": False,
        "runner_exists": True,
        "launchable_now": True,
    }
    _require(
        authorization.get("run_contract") == expected_run,
        "V6 model, engine, ingress, action, or matrix contract drifted",
    )
    _require(
        authorization.get("resource_gates") == base["resource_gates"],
        "V6 resource gates drifted",
    )
    _require(
        authorization.get("implementation_audit") == _v6_implementation_audit(base),
        "V6 implementation audit is incomplete",
    )
    _require(
        authorization.get("atomic_ingress_proof", {}).get("decision") == "pass"
        and authorization["atomic_ingress_proof"].get("engine_core_class")
        == INPROCESS_ENGINE_CORE_CLASS
        and authorization["atomic_ingress_proof"].get("first_decode_exclusion_reasons")
        == []
        and authorization["atomic_ingress_proof"].get("shared_target_kv_layer_count")
        == 36,
        "V6 atomic-ingress evidence is incomplete",
    )
    _require(
        authorization.get("measurement_repair", {}).get("state") == "pass"
        and authorization["measurement_repair"].get("measured_shared_kv_blocks")
        >= MINIMUM_SHARED_KV_BLOCKS,
        "V6 full-prefill resource repair is incomplete",
    )
    _require(
        authorization.get("serving_chunked_prefill_boundary")
        == _serving_chunked_prefill_boundary(),
        "V6 does not preserve the real-serving boundary",
    )
    _require(
        authorization.get("next_artifact")
        == {
            "kind": "p4_b0_value_screen_result",
            "requires_complete_capture_count": 432,
            "may_authorize_p4a": False,
            "may_admit_action": False,
        },
        "V6 post-run boundary drifted",
    )


def _validate_v11_execution_authority(authorization: Mapping[str, Any]) -> None:
    _require(
        authorization.get("schema_version") == 11
        and authorization.get("package_id") == V11_PACKAGE_ID
        and authorization.get("status")
        == "authorized_gpu4_repaired_variable_prefill_value_screen_only",
        "runner accepts only the reviewed V11 value-screen authority",
    )
    validate_preparation_contract(authorization)
    _require(
        authorization.get("request_id_repair") == _v7_request_id_repair(),
        "V11 request-ID canonicalization contract drifted",
    )
    _require(
        authorization.get("decode_work_repair") == _v8_decode_work_repair(),
        "V11 decode-work offset contract drifted",
    )
    _validate_v11_consumed_attempt(authorization.get("consumed_attempt", {}))
    _validate_v11_repair_validation(
        authorization.get("variable_prefill_repair_validation", {})
    )
    _validate_v10_probe_evidence(authorization.get("chunked_prefill_probe", {}))
    _require(
        authorization.get("chunked_prefill_contract")
        == _v10_chunked_prefill_contract(),
        "V11 chunked-prefill cohort contract drifted",
    )
    _require(
        authorization.get("authorizations")
        == {
            "capture_runner_conformance_engineering": True,
            "gpu_measurement": True,
            "v11_execution": True,
            "value_screen_scoring": True,
            "p4a_engineering": False,
            "action_admission": False,
            "production_value_claim": False,
        },
        "V11 authorities drifted beyond the value screen",
    )
    claims = authorization.get("claims", {})
    _require(
        claims.get("executable_run_ready") is True
        and claims.get("prior_outputs_reusable") is False
        and claims.get("full_prefill_geometry_rejected") is True
        and claims.get("chunked_prefill_cohort_cpu_proven") is True
        and claims.get("chunked_prefill_k4_gpu_probe_passed") is True
        and claims.get("variable_prefill_runtime_evidence_repaired") is True
        and claims.get("isolated_r8_gpu_case_passed") is True
        and claims.get("r5cot_to_r8_gpu_case_passed") is True
        and claims.get("repair_parent_aggregate_passed") is False
        and claims.get("repair_parent_rejection_is_observer_only") is True
        and claims.get("decode_work_offset_tested") is True
        and claims.get("all_action_capture_rollover_tested") is True
        and claims.get("request_id_canonicalization_tested") is True
        and claims.get("internal_request_id_randomization_retained") is True,
        "V11 does not claim the required tested repair path",
    )
    _require(
        authorization.get("decision", {}).get("state") == "approve"
        and authorization.get("decision", {}).get("scope")
        == "gpu4_repaired_variable_prefill_value_screen_v11_only",
        "V11 decision does not approve the narrow GPU-4 screen",
    )
    policy = authorization.get("execution_policy", {})
    _require(
        policy.get("physical_gpu_index") == 4
        and policy.get("physical_gpu_uuid") == GPU4_UUID
        and policy.get("fallback_gpu_authorized") is False
        and policy.get("engine_core_class") == INPROCESS_ENGINE_CORE_CLASS
        and policy.get("v1_multiprocessing") is False
        and policy.get("physical_boot_count") == 9
        and policy.get("capture_count") == 432
        and policy.get("max_num_batched_tokens") == SERVING_MAX_NUM_BATCHED_TOKENS
        and policy.get("effective_max_num_scheduled_tokens")
        == SERVING_EFFECTIVE_MAX_NUM_SCHEDULED_TOKENS
        and policy.get("gpu_memory_utilization") == SERVING_GPU_MEMORY_UTILIZATION
        and policy.get("capture_cohort_barrier_required") is True
        and policy.get("prior_output_reuse_allowed") is False
        and policy.get("repair_validation_output_reuse_allowed") is False
        and policy.get("retry_allowed") is False
        and policy.get("partial_resume_allowed") is False,
        "V11 execution policy lost its bounded GPU or fail-closed boundary",
    )

    references = authorization.get("source_artifacts", {})
    source_paths = _v11_source_paths()
    _require(
        set(references) == set(source_paths),
        "V11 execution source closure is incomplete or inflated",
    )
    for role, path in source_paths.items():
        _require(
            references[role] == _file_reference(path),
            f"V11 execution source hash drifted for {role}",
        )

    base = _load_json(_repository_path(BASE_AUTHORIZATION_PATH))
    expected_run = copy.deepcopy(base["run_contract"])
    expected_run["engine"]["max_num_batched_tokens"] = SERVING_MAX_NUM_BATCHED_TOKENS
    expected_run["engine"]["enable_chunked_prefill"] = True
    expected_run["engine"]["gpu_memory_utilization"] = SERVING_GPU_MEMORY_UTILIZATION
    expected_run["environment"][V1_MULTIPROCESSING_ENV] = V1_MULTIPROCESSING_VALUE
    expected_run["invocation"] = {
        "runner_path": source_paths["matrix_runner"],
        "argv": [
            ".venv/bin/python",
            source_paths["matrix_runner"],
            "--authorization",
            V11_AUTHORIZATION_PATH,
            "--output-dir",
            V11_OUTPUT_PATH,
        ],
        "output_dir": V11_OUTPUT_PATH,
        "overwrite_allowed": False,
        "runner_exists": True,
        "launchable_now": True,
    }
    _require(
        authorization.get("run_contract") == expected_run,
        "V11 model, engine, cohort, action, or matrix contract drifted",
    )
    _require(
        authorization.get("resource_gates") == base["resource_gates"],
        "V11 resource gates drifted",
    )
    _require(
        authorization.get("next_artifact")
        == {
            "kind": "p4_b0_value_screen_result",
            "requires_complete_capture_count": 432,
            "may_authorize_p4a": False,
            "may_admit_action": False,
        },
        "V11 post-run boundary drifted",
    )
    _require(
        _reviewed_output_path(_repository_path(V11_AUTHORIZATION_PATH))
        == _repository_path(V11_OUTPUT_PATH),
        "V11 launch dispatcher does not resolve the reviewed path pair",
    )


def _validate_v10_execution_authority(authorization: Mapping[str, Any]) -> None:
    _require(
        authorization.get("schema_version") == 10
        and authorization.get("package_id") == V10_PACKAGE_ID
        and authorization.get("status")
        == "authorized_gpu4_chunked_prefill_cohort_value_screen_only",
        "runner accepts only the reviewed V10 cohort value-screen authority",
    )
    validate_preparation_contract(authorization)
    _require(
        authorization.get("request_id_repair") == _v7_request_id_repair(),
        "V10 request-ID canonicalization contract drifted",
    )
    _require(
        authorization.get("decode_work_repair") == _v8_decode_work_repair(),
        "V10 decode-work offset contract drifted",
    )
    _validate_v10_probe_evidence(authorization.get("chunked_prefill_probe", {}))
    _require(
        authorization.get("chunked_prefill_contract")
        == _v10_chunked_prefill_contract(),
        "V10 chunked-prefill cohort contract drifted",
    )
    _require(
        authorization.get("authorizations")
        == {
            "capture_runner_conformance_engineering": True,
            "gpu_measurement": True,
            "v10_execution": True,
            "value_screen_scoring": True,
            "p4a_engineering": False,
            "action_admission": False,
            "production_value_claim": False,
        },
        "V10 authorities drifted beyond the value screen",
    )
    claims = authorization.get("claims", {})
    _require(
        claims.get("executable_run_ready") is True
        and claims.get("full_prefill_geometry_rejected") is True
        and claims.get("chunked_prefill_cohort_cpu_proven") is True
        and claims.get("chunked_prefill_k4_gpu_probe_passed") is True
        and claims.get("decode_work_offset_tested") is True
        and claims.get("all_action_capture_rollover_tested") is True
        and claims.get("request_id_canonicalization_tested") is True
        and claims.get("internal_request_id_randomization_retained") is True,
        "V10 does not claim the required tested cohort path",
    )
    _require(
        authorization.get("decision", {}).get("state") == "approve"
        and authorization.get("decision", {}).get("scope")
        == "gpu4_chunked_prefill_cohort_value_screen_v10_only",
        "V10 decision does not approve the narrow GPU-4 screen",
    )
    policy = authorization.get("execution_policy", {})
    _require(
        policy.get("physical_gpu_index") == 4
        and policy.get("physical_gpu_uuid") == GPU4_UUID
        and policy.get("fallback_gpu_authorized") is False
        and policy.get("engine_core_class") == INPROCESS_ENGINE_CORE_CLASS
        and policy.get("v1_multiprocessing") is False
        and policy.get("physical_boot_count") == 9
        and policy.get("capture_count") == 432
        and policy.get("max_num_batched_tokens") == SERVING_MAX_NUM_BATCHED_TOKENS
        and policy.get("effective_max_num_scheduled_tokens")
        == SERVING_EFFECTIVE_MAX_NUM_SCHEDULED_TOKENS
        and policy.get("gpu_memory_utilization") == SERVING_GPU_MEMORY_UTILIZATION
        and policy.get("capture_cohort_barrier_required") is True
        and policy.get("retry_allowed") is False
        and policy.get("partial_resume_allowed") is False,
        "V10 execution policy lost its bounded GPU or fail-closed boundary",
    )

    references = authorization.get("source_artifacts", {})
    source_paths = _v10_source_paths()
    _require(
        set(references) == set(source_paths),
        "V10 execution source closure is incomplete or inflated",
    )
    for role, path in source_paths.items():
        _require(
            references[role] == _file_reference(path),
            f"V10 execution source hash drifted for {role}",
        )

    base = _load_json(_repository_path(BASE_AUTHORIZATION_PATH))
    expected_run = copy.deepcopy(base["run_contract"])
    expected_run["engine"]["max_num_batched_tokens"] = SERVING_MAX_NUM_BATCHED_TOKENS
    expected_run["engine"]["enable_chunked_prefill"] = True
    expected_run["engine"]["gpu_memory_utilization"] = SERVING_GPU_MEMORY_UTILIZATION
    expected_run["environment"][V1_MULTIPROCESSING_ENV] = V1_MULTIPROCESSING_VALUE
    expected_run["invocation"] = {
        "runner_path": source_paths["matrix_runner"],
        "argv": [
            ".venv/bin/python",
            source_paths["matrix_runner"],
            "--authorization",
            V10_AUTHORIZATION_PATH,
            "--output-dir",
            V10_OUTPUT_PATH,
        ],
        "output_dir": V10_OUTPUT_PATH,
        "overwrite_allowed": False,
        "runner_exists": True,
        "launchable_now": True,
    }
    _require(
        authorization.get("run_contract") == expected_run,
        "V10 model, engine, cohort, action, or matrix contract drifted",
    )
    _require(
        authorization.get("resource_gates") == base["resource_gates"],
        "V10 resource gates drifted",
    )
    _require(
        authorization.get("next_artifact")
        == {
            "kind": "p4_b0_value_screen_result",
            "requires_complete_capture_count": 432,
            "may_authorize_p4a": False,
            "may_admit_action": False,
        },
        "V10 post-run boundary drifted",
    )
    _require(
        _reviewed_output_path(_repository_path(V10_AUTHORIZATION_PATH))
        == _repository_path(V10_OUTPUT_PATH),
        "V10 launch dispatcher does not resolve the reviewed path pair",
    )


def _validate_v9_execution_authority(authorization: Mapping[str, Any]) -> None:
    _require(
        authorization.get("schema_version") == 9
        and authorization.get("package_id") == V9_PACKAGE_ID
        and authorization.get("status")
        == "authorized_gpu4_launch_dispatch_repair_value_screen_only",
        "runner accepts only the reviewed V9 launch-dispatch authorization",
    )
    validate_preparation_contract(authorization)
    _require(
        authorization.get("request_id_repair") == _v7_request_id_repair(),
        "V9 request-ID canonicalization contract drifted",
    )
    _require(
        authorization.get("decode_work_repair") == _v8_decode_work_repair(),
        "V9 decode-work offset contract drifted",
    )
    _require(
        authorization.get("launch_dispatch_repair") == _v9_launch_dispatch_repair(),
        "V9 parent/child launch-dispatch contract drifted",
    )
    _require(
        authorization.get("authorizations")
        == {
            "capture_runner_conformance_engineering": True,
            "gpu_measurement": True,
            "v9_execution": True,
            "serving_diagnosis": False,
            "p4a_engineering": False,
            "action_admission": False,
            "production_value_claim": False,
        },
        "V9 authorities drifted beyond the value screen",
    )
    claims = authorization.get("claims", {})
    _require(
        claims.get("executable_run_ready") is True
        and claims.get("launch_dispatch_repair_tested") is True
        and claims.get("decode_work_offset_tested") is True
        and claims.get("all_action_capture_rollover_tested") is True
        and claims.get("request_id_canonicalization_tested") is True
        and claims.get("internal_request_id_randomization_retained") is True
        and claims.get("atomic_ingress_proof_passed") is True,
        "V9 does not claim the tested launch and decode-work repairs",
    )
    _require(
        authorization.get("decision", {}).get("state") == "approve"
        and authorization.get("decision", {}).get("scope")
        == "gpu4_launch_dispatch_repair_value_screen_v9_only",
        "V9 decision does not approve the narrow GPU-4 screen",
    )
    policy = authorization.get("execution_policy", {})
    _require(
        policy.get("physical_gpu_index") == 4
        and policy.get("physical_gpu_uuid") == GPU4_UUID
        and policy.get("fallback_gpu_authorized") is False
        and policy.get("engine_core_class") == INPROCESS_ENGINE_CORE_CLASS
        and policy.get("v1_multiprocessing") is False
        and policy.get("retry_allowed") is False
        and policy.get("partial_resume_allowed") is False,
        "V9 execution policy lost its GPU or fail-closed boundary",
    )

    references = authorization.get("source_artifacts", {})
    source_paths = _v9_source_paths()
    _require(
        set(references) == set(source_paths),
        "V9 execution source closure is incomplete or inflated",
    )
    for role, path in source_paths.items():
        _require(
            references[role] == _file_reference(path),
            f"V9 execution source hash drifted for {role}",
        )

    base = _load_json(_repository_path(BASE_AUTHORIZATION_PATH))
    expected_run = copy.deepcopy(base["run_contract"])
    expected_run["engine"]["max_num_batched_tokens"] = (
        FULL_PREFILL_MAX_NUM_BATCHED_TOKENS
    )
    expected_run["engine"]["enable_chunked_prefill"] = True
    expected_run["engine"]["gpu_memory_utilization"] = (
        MEASUREMENT_GPU_MEMORY_UTILIZATION
    )
    expected_run["environment"][V1_MULTIPROCESSING_ENV] = V1_MULTIPROCESSING_VALUE
    expected_run["invocation"] = {
        "runner_path": source_paths["matrix_runner"],
        "argv": [
            ".venv/bin/python",
            source_paths["matrix_runner"],
            "--authorization",
            V9_AUTHORIZATION_PATH,
            "--output-dir",
            V9_OUTPUT_PATH,
        ],
        "output_dir": V9_OUTPUT_PATH,
        "overwrite_allowed": False,
        "runner_exists": True,
        "launchable_now": True,
    }
    _require(
        authorization.get("run_contract") == expected_run,
        "V9 model, engine, ingress, action, or matrix contract drifted",
    )
    _require(
        authorization.get("resource_gates") == base["resource_gates"],
        "V9 resource gates drifted",
    )
    retained = authorization.get("retained_evidence", {})
    _require(
        retained.get("atomic_ingress_proof", {}).get("decision") == "pass"
        and retained.get("resource_probe", {}).get("decision") == "pass"
        and retained.get("serving_chunked_prefill_boundary")
        == _serving_chunked_prefill_boundary(),
        "V9 retained evidence is incomplete",
    )
    _require(
        authorization.get("next_artifact")
        == {
            "kind": "p4_b0_value_screen_result",
            "requires_complete_capture_count": 432,
            "may_authorize_p4a": False,
            "may_admit_action": False,
        },
        "V9 post-run boundary drifted",
    )
    _require(
        _reviewed_output_path(_repository_path(V9_AUTHORIZATION_PATH))
        == _repository_path(V9_OUTPUT_PATH),
        "V9 launch dispatcher does not resolve the reviewed path pair",
    )


def _validate_v8_execution_authority(authorization: Mapping[str, Any]) -> None:
    _require(
        authorization.get("schema_version") == 8
        and authorization.get("package_id") == V8_PACKAGE_ID
        and authorization.get("status")
        == "authorized_gpu4_decode_work_offset_value_screen_only",
        "runner accepts only the reviewed V8 decode-work authorization",
    )
    validate_preparation_contract(authorization)
    _require(
        authorization.get("request_id_repair") == _v7_request_id_repair(),
        "V8 request-ID canonicalization contract drifted",
    )
    _require(
        authorization.get("decode_work_repair") == _v8_decode_work_repair(),
        "V8 decode-work offset contract drifted",
    )
    _require(
        authorization.get("authorizations")
        == {
            "capture_runner_conformance_engineering": True,
            "gpu_measurement": True,
            "v8_execution": True,
            "serving_diagnosis": False,
            "p4a_engineering": False,
            "action_admission": False,
            "production_value_claim": False,
        },
        "V8 authorities drifted beyond the value screen",
    )
    claims = authorization.get("claims", {})
    _require(
        claims.get("executable_run_ready") is True
        and claims.get("decode_work_offset_tested") is True
        and claims.get("all_action_capture_rollover_tested") is True
        and claims.get("request_id_canonicalization_tested") is True
        and claims.get("internal_request_id_randomization_retained") is True
        and claims.get("atomic_ingress_proof_passed") is True,
        "V8 does not claim the tested decode-work repair",
    )
    _require(
        authorization.get("decision", {}).get("state") == "approve"
        and authorization.get("decision", {}).get("scope")
        == "gpu4_decode_work_offset_value_screen_v8_only",
        "V8 decision does not approve the narrow GPU-4 screen",
    )
    policy = authorization.get("execution_policy", {})
    _require(
        policy.get("physical_gpu_index") == 4
        and policy.get("physical_gpu_uuid") == GPU4_UUID
        and policy.get("fallback_gpu_authorized") is False
        and policy.get("engine_core_class") == INPROCESS_ENGINE_CORE_CLASS
        and policy.get("v1_multiprocessing") is False
        and policy.get("retry_allowed") is False
        and policy.get("partial_resume_allowed") is False,
        "V8 execution policy lost its GPU or fail-closed boundary",
    )

    references = authorization.get("source_artifacts", {})
    source_paths = _v8_source_paths()
    _require(
        set(references) == set(source_paths),
        "V8 execution source closure is incomplete or inflated",
    )
    for role, path in source_paths.items():
        _require(
            references[role] == _file_reference(path),
            f"V8 execution source hash drifted for {role}",
        )

    base = _load_json(_repository_path(BASE_AUTHORIZATION_PATH))
    expected_run = copy.deepcopy(base["run_contract"])
    expected_run["engine"]["max_num_batched_tokens"] = (
        FULL_PREFILL_MAX_NUM_BATCHED_TOKENS
    )
    expected_run["engine"]["enable_chunked_prefill"] = True
    expected_run["engine"]["gpu_memory_utilization"] = (
        MEASUREMENT_GPU_MEMORY_UTILIZATION
    )
    expected_run["environment"][V1_MULTIPROCESSING_ENV] = V1_MULTIPROCESSING_VALUE
    expected_run["invocation"] = {
        "runner_path": source_paths["matrix_runner"],
        "argv": [
            ".venv/bin/python",
            source_paths["matrix_runner"],
            "--authorization",
            V8_AUTHORIZATION_PATH,
            "--output-dir",
            V8_OUTPUT_PATH,
        ],
        "output_dir": V8_OUTPUT_PATH,
        "overwrite_allowed": False,
        "runner_exists": True,
        "launchable_now": True,
    }
    _require(
        authorization.get("run_contract") == expected_run,
        "V8 model, engine, ingress, action, or matrix contract drifted",
    )
    _require(
        authorization.get("resource_gates") == base["resource_gates"],
        "V8 resource gates drifted",
    )
    retained = authorization.get("retained_evidence", {})
    _require(
        retained.get("atomic_ingress_proof", {}).get("decision") == "pass"
        and retained.get("resource_probe", {}).get("decision") == "pass"
        and retained.get("serving_chunked_prefill_boundary")
        == _serving_chunked_prefill_boundary(),
        "V8 retained evidence is incomplete",
    )
    _require(
        authorization.get("next_artifact")
        == {
            "kind": "p4_b0_value_screen_result",
            "requires_complete_capture_count": 432,
            "may_authorize_p4a": False,
            "may_admit_action": False,
        },
        "V8 post-run boundary drifted",
    )


def _validate_v7_execution_authority(authorization: Mapping[str, Any]) -> None:
    _require(
        authorization.get("schema_version") == 7
        and authorization.get("package_id") == V7_PACKAGE_ID
        and authorization.get("status")
        == "authorized_gpu4_request_id_canonicalization_value_screen_only",
        "runner accepts only the reviewed V7 request-ID authorization",
    )
    validate_preparation_contract(authorization)
    _require(
        authorization.get("request_id_repair") == _v7_request_id_repair(),
        "V7 request-ID canonicalization contract drifted",
    )
    _require(
        authorization.get("authorizations")
        == {
            "capture_runner_conformance_engineering": True,
            "gpu_measurement": True,
            "v7_execution": True,
            "serving_diagnosis": False,
            "p4a_engineering": False,
            "action_admission": False,
            "production_value_claim": False,
        },
        "V7 authorities drifted beyond the value screen",
    )
    claims = authorization.get("claims", {})
    _require(
        claims.get("executable_run_ready") is True
        and claims.get("request_id_canonicalization_tested") is True
        and claims.get("internal_request_id_randomization_retained") is True
        and claims.get("atomic_ingress_proof_passed") is True,
        "V7 does not claim the tested request-ID repair",
    )
    _require(
        authorization.get("decision", {}).get("state") == "approve"
        and authorization.get("decision", {}).get("scope")
        == "gpu4_request_id_canonicalization_value_screen_v7_only",
        "V7 decision does not approve the narrow GPU-4 screen",
    )
    policy = authorization.get("execution_policy", {})
    _require(
        policy.get("physical_gpu_index") == 4
        and policy.get("physical_gpu_uuid") == GPU4_UUID
        and policy.get("fallback_gpu_authorized") is False
        and policy.get("engine_core_class") == INPROCESS_ENGINE_CORE_CLASS
        and policy.get("v1_multiprocessing") is False
        and policy.get("retry_allowed") is False
        and policy.get("partial_resume_allowed") is False,
        "V7 execution policy lost its GPU or fail-closed boundary",
    )

    references = authorization.get("source_artifacts", {})
    source_paths = _v7_source_paths()
    _require(
        set(references) == set(source_paths),
        "V7 execution source closure is incomplete or inflated",
    )
    for role, path in source_paths.items():
        _require(
            references[role] == _file_reference(path),
            f"V7 execution source hash drifted for {role}",
        )

    base = _load_json(_repository_path(BASE_AUTHORIZATION_PATH))
    expected_run = copy.deepcopy(base["run_contract"])
    expected_run["engine"]["max_num_batched_tokens"] = (
        FULL_PREFILL_MAX_NUM_BATCHED_TOKENS
    )
    expected_run["engine"]["enable_chunked_prefill"] = True
    expected_run["engine"]["gpu_memory_utilization"] = (
        MEASUREMENT_GPU_MEMORY_UTILIZATION
    )
    expected_run["environment"][V1_MULTIPROCESSING_ENV] = V1_MULTIPROCESSING_VALUE
    expected_run["invocation"] = {
        "runner_path": source_paths["matrix_runner"],
        "argv": [
            ".venv/bin/python",
            source_paths["matrix_runner"],
            "--authorization",
            V7_AUTHORIZATION_PATH,
            "--output-dir",
            V7_OUTPUT_PATH,
        ],
        "output_dir": V7_OUTPUT_PATH,
        "overwrite_allowed": False,
        "runner_exists": True,
        "launchable_now": True,
    }
    _require(
        authorization.get("run_contract") == expected_run,
        "V7 model, engine, ingress, action, or matrix contract drifted",
    )
    _require(
        authorization.get("resource_gates") == base["resource_gates"],
        "V7 resource gates drifted",
    )
    retained = authorization.get("retained_evidence", {})
    _require(
        retained.get("atomic_ingress_proof", {}).get("decision") == "pass"
        and retained.get("resource_probe", {}).get("decision") == "pass"
        and retained.get("serving_chunked_prefill_boundary")
        == _serving_chunked_prefill_boundary(),
        "V7 retained evidence is incomplete",
    )
    _require(
        authorization.get("next_artifact")
        == {
            "kind": "p4_b0_value_screen_result",
            "requires_complete_capture_count": 432,
            "may_authorize_p4a": False,
            "may_admit_action": False,
        },
        "V7 post-run boundary drifted",
    )


def validate_execution_authority(
    authorization: Mapping[str, Any],
) -> None:
    """Require a future source-bound approval before any GPU child starts."""
    if authorization.get("package_id") == V11_PACKAGE_ID:
        _validate_v11_execution_authority(authorization)
        return
    if authorization.get("package_id") == V10_PACKAGE_ID:
        _validate_v10_execution_authority(authorization)
        return
    if authorization.get("package_id") == V9_PACKAGE_ID:
        _validate_v9_execution_authority(authorization)
        return
    if authorization.get("package_id") == V8_PACKAGE_ID:
        _validate_v8_execution_authority(authorization)
        return
    if authorization.get("package_id") == V7_PACKAGE_ID:
        _validate_v7_execution_authority(authorization)
        return
    if authorization.get("package_id") == V6_PACKAGE_ID:
        _validate_v6_execution_authority(authorization)
        return
    if authorization.get("package_id") == CURRENT_HOLD_PACKAGE_ID:
        validate_preparation_contract(authorization)
        raise P4RunnerError("the reviewed HOLD package cannot execute GPU measurement")
    _require(
        authorization.get("package_id") == APPROVED_PACKAGE_ID
        and authorization.get("schema_version") == 5
        and authorization.get("status")
        == "authorized_gpu4_full_prefill_resource_repair_retry_only",
        "runner accepts only the reviewed V5 GPU-4 retry authorization",
    )
    validate_preparation_contract(authorization)
    _require(
        authorization.get("authorizations", {}).get("gpu_measurement") is True,
        "GPU measurement is not explicitly authorized",
    )
    _require(
        authorization.get("decision", {}).get("state") in {"approve", "authorized"},
        "authorization decision does not approve execution",
    )
    _require(
        authorization.get("claims", {}).get("executable_run_ready") is True,
        "authorization does not claim executable readiness",
    )
    invocation = authorization["run_contract"]["invocation"]
    _require(
        invocation.get("runner_exists") is True
        and invocation.get("launchable_now") is True,
        "authorization invocation remains held",
    )
    audit = authorization.get("implementation_audit", {})
    checks = audit.get("checks", ())
    _require(
        audit.get("state") == "pass"
        and len(checks) == 6
        and all(row.get("status") == "pass" for row in checks),
        "capture-runner conformance audit is not complete",
    )
    reference = authorization.get("source_artifacts", {}).get(
        "capture_runner_conformance"
    )
    _require(
        isinstance(reference, Mapping),
        "future authorization must bind the conformance artifact",
    )
    expected_path = str(CONFORMANCE_PATH.relative_to(REPO_ROOT))
    _require(
        reference.get("path") == expected_path and CONFORMANCE_PATH.is_file(),
        "future authorization points to another conformance artifact",
    )
    actual_hash = hashlib.sha256(CONFORMANCE_PATH.read_bytes()).hexdigest()
    _require(
        reference.get("sha256") == actual_hash,
        "capture-runner conformance hash differs from the approved source",
    )
    conformance = _load_json(CONFORMANCE_PATH)
    _require(
        conformance.get("status") == "pass_cpu_conformance_reauthorization_required"
        and len(conformance.get("implementation_checks", ())) == 6
        and all(
            row.get("status") == "pass" for row in conformance["implementation_checks"]
        )
        and not any(conformance.get("authorizations", {}).values()),
        "approved conformance artifact is not a narrow six-check pass",
    )
    expected_sources = {
        "capture_runner_conformance": {
            "path": expected_path,
            "sha256": actual_hash,
        },
        **conformance["source_artifacts"],
    }
    for role in (
        "matrix_runner",
        "runner_tests",
        "conformance_tests",
        "validator",
    ):
        expected_sources[role] = authorization["source_artifacts"][role]
    references = authorization["source_artifacts"]
    _require(
        set(references) == set(expected_sources),
        "authorization source bindings are incomplete or inflated",
    )
    for role, expected in expected_sources.items():
        approved = references[role]
        _require(
            approved == expected,
            f"authorization source binding drifted for {role}",
        )
        path = _repository_path(expected["path"])
        _require(path.is_file(), f"approved source is missing for {role}: {path}")
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        _require(
            source_hash == expected["sha256"],
            f"approved source hash drifted for {role}",
        )

    historical = _load_json(
        _repository_path(expected_sources["held_authorization"]["path"])
    )
    historical_hash = expected_sources["held_authorization"]["sha256"]
    _require(
        authorization.get("supersedes")
        == {
            "path": expected_sources["held_authorization"]["path"],
            "sha256": historical_hash,
            "historical_decision": "hold",
        },
        "approved package does not supersede the exact historical HOLD",
    )
    run = authorization["run_contract"]
    historical_run = historical["run_contract"]
    expected_run = copy.deepcopy(historical_run)
    expected_run["engine"]["max_num_batched_tokens"] = (
        FULL_PREFILL_MAX_NUM_BATCHED_TOKENS
    )
    expected_run["engine"]["enable_chunked_prefill"] = True
    expected_run["engine"]["gpu_memory_utilization"] = (
        MEASUREMENT_GPU_MEMORY_UTILIZATION
    )
    expected_run["invocation"] = {
        "runner_path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        "argv": [
            ".venv/bin/python",
            str(Path(__file__).resolve().relative_to(REPO_ROOT)),
            "--authorization",
            APPROVED_AUTHORIZATION_PATH,
            "--output-dir",
            APPROVED_OUTPUT_PATH,
        ],
        "output_dir": APPROVED_OUTPUT_PATH,
        "overwrite_allowed": False,
        "runner_exists": True,
        "launchable_now": True,
    }
    _require(
        run == expected_run,
        "approved model, GPU, measurement engine, action, or matrix contract drifted",
    )
    _require(
        authorization.get("resource_gates") == historical["resource_gates"],
        "approved resource gates drifted",
    )
    _require(
        audit
        == {
            "state": "pass",
            "checks": conformance["implementation_checks"],
        },
        "approved implementation audit differs from conformance",
    )
    _require(
        authorization.get("evidence_readiness")
        == {
            "state": "complete",
            "historical_hold_validated": True,
            "capture_runner_conformance_status": "pass",
            "implementation_checks_passed": 6,
        },
        "approved evidence-readiness summary drifted",
    )
    _require(
        authorization.get("measurement_repair", {}).get("state") == "pass"
        and authorization["measurement_repair"]["allowed_engine_overrides"]
        == {
            "max_num_batched_tokens": FULL_PREFILL_MAX_NUM_BATCHED_TOKENS,
            "enable_chunked_prefill": True,
            "gpu_memory_utilization": MEASUREMENT_GPU_MEMORY_UTILIZATION,
        }
        and authorization["measurement_repair"]["measured_shared_kv_blocks"]
        >= MINIMUM_SHARED_KV_BLOCKS,
        "approved full-prefill resource repair is incomplete",
    )
    _require(
        authorization.get("serving_chunked_prefill_boundary")
        == _serving_chunked_prefill_boundary(),
        "approved package does not preserve real-serving chunked prefill",
    )
    _require(
        authorization.get("decision")
        == {
            "state": "approve",
            "scope": "gpu4_complete_boot_static_value_screen_only",
            "basis": [
                "frozen_evidence_chain_valid",
                "six_capture_runner_checks_pass",
                "all_execution_sources_hash_bound",
                "runtime_resource_floor_fail_closed",
            ],
            "invalidated_by": [
                "approved_source_hash_drift",
                "output_directory_exists",
                "gpu_identity_drift",
                "resource_floor_failure",
                "matrix_or_contract_drift",
            ],
        },
        "approved decision boundary drifted",
    )
    _require(
        authorization.get("claims")
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
        "approved claims drifted or overstate results",
    )
    _require(
        authorization.get("authorizations")
        == {
            "capture_runner_conformance_engineering": True,
            "gpu_measurement": True,
            "p4a_engineering": False,
            "action_admission": False,
            "production_value_claim": False,
        },
        "approved authorities drifted beyond the value screen",
    )
    _require(
        authorization.get("execution_policy")
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
        "approved execution policy drifted",
    )
    _require(
        authorization.get("next_artifact")
        == {
            "kind": "p4_b0_value_screen_result",
            "requires_complete_capture_count": 432,
            "may_authorize_p4a": False,
            "may_admit_action": False,
        },
        "approved post-run boundary drifted",
    )


def _load_prompt_rows() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = _load_json(PROMPT_MANIFEST_PATH)
    try:
        with gzip.open(PROMPT_BUNDLE_PATH, "rt", encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise P4RunnerError(f"cannot load frozen prompt bundle: {exc}") from exc
    _require(len(rows) == 384, "prompt bundle does not contain 384 records")
    by_id = {row["record_id"]: row for row in rows}
    _require(len(by_id) == len(rows), "prompt bundle repeats a record id")
    _require(
        set(by_id) == {row["record_id"] for row in manifest["prompts"]},
        "prompt bundle ids differ from the frozen manifest",
    )
    return manifest, by_id


def _prompt_group(
    manifest: Mapping[str, Any], regime_id: str, content_seed: int
) -> list[str]:
    rows = [
        row
        for row in manifest["prompts"]
        if row["regime_id"] == regime_id and row["content_seed"] == content_seed
    ]
    rows.sort(key=lambda row: row["prompt_index"])
    _require(
        len(rows) == 32 and [row["prompt_index"] for row in rows] == list(range(32)),
        f"prompt group {regime_id}/seed{content_seed} is not canonical",
    )
    return [row["record_id"] for row in rows]


def _capture_template(
    authorization: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    logical_version: str,
    block_id: int,
    action_position: int,
    action_id: str,
    boot_id: str,
    regime: Mapping[str, Any],
    content_seed: int,
    round_index: int,
) -> dict[str, Any]:
    regime_id = regime["regime_id"]
    prompt_ids = _prompt_group(manifest, regime_id, content_seed)
    slug = ACTION_SLUGS[action_id]
    capture_id = (
        f"capture-b{block_id}-p{action_position}-{slug}-"
        f"{regime_id.lower()}-s{content_seed}-r{round_index}"
    )
    run = authorization["run_contract"]
    model = run["model"]
    gpu = run["gpu_assignment"]
    return {
        "schema_version": 1,
        "capture_contract_id": "p4-b0-same-event-capture-v1",
        "capture_id": capture_id,
        "scored": False,
        "complete": False,
        "warmup_complete": True,
        "runner": {
            "preregistration_id": SCREEN_ID,
            "prompt_manifest_id": manifest["manifest_id"],
            "prompt_manifest_sha256": hashlib.sha256(
                PROMPT_MANIFEST_PATH.read_bytes()
            ).hexdigest(),
            "prompt_bundle_sha256": manifest["bundle"]["sha256"],
            "target_checkpoint_revision": model["revision"],
            "target_quantization": model["target_quantization"],
            "target_kv_dtype": model["target_kv_dtype"],
            "draft_weight_version": logical_version,
            "shared_kv_binding_id": "pending-live-proof",
            "shared_kv_alias_proven": True,
            "true_slot_mapping_id": "pending-live-proof",
            "true_slot_identity_proven": True,
            "hardware_id": gpu["uuid"],
            "parallel_layout": "tp1-pp1",
            "kernel_backend": "vllm-cuda",
            "graph_grade": "fullcg-piecewise",
            "warmup_policy": "registered-four-round",
            "measurement_currency": "S_dec",
        },
        "matrix": {
            "boot_block_id": block_id,
            "boot_id": boot_id,
            "action_id": action_id,
            "action_position": action_position,
            "action_realization": ACTION_REALIZATIONS[action_id],
            "regime_id": regime_id,
            "content_seed": content_seed,
            "round_index": round_index,
        },
        "generation": {
            "prompt_record_ids": prompt_ids,
            "generation_seed": manifest["prompt_plan"]["generation_seed"],
            "batch": regime["batch"],
            "max_output_tokens": regime["max_output_tokens"],
            "temperature": regime["temperature"],
            "ignore_eos": manifest["prompt_plan"]["ignore_eos"],
            "requested_output_tokens": 32 * regime["max_output_tokens"],
        },
        "events": [],
    }


def build_boot_specs(
    authorization: Mapping[str, Any], output_dir: Path
) -> list[dict[str, Any]]:
    """Build the exact nine boot specs and 48-cell plans in memory."""
    validate_preparation_contract(authorization)
    manifest, prompt_rows = _load_prompt_rows()
    regimes = {row["regime_id"]: row for row in manifest["prompt_plan"]["regimes"]}
    _require(set(regimes) == set(REGIME_ORDER), "prompt regimes drifted")
    bounded_chunked_prefill = authorization.get("package_id") in {
        V10_PACKAGE_ID,
        V11_PACKAGE_ID,
    }
    if bounded_chunked_prefill:
        budget_key = "chunked_prefill_budget"
        budget_evidence = chunked_prefill_budget_evidence(
            manifest,
            prompt_rows,
            authorization["run_contract"]["engine"],
        )
    else:
        budget_key = "full_prefill_budget"
        budget_evidence = full_prefill_budget_evidence(
            manifest,
            prompt_rows,
            authorization["run_contract"]["engine"],
        )
    logical_version = logical_weight_version(authorization)
    action_boots = {
        row["action_id"]: row for row in authorization["run_contract"]["action_boots"]
    }
    specs = []
    for block_id, action_order in ACTION_ORDERS.items():
        for action_position, action_id in enumerate(action_order, start=1):
            slug = ACTION_SLUGS[action_id]
            boot_id = f"p4-b0-b{block_id}-p{action_position}-{slug}"
            cells = [
                _capture_template(
                    authorization,
                    manifest,
                    logical_version=logical_version,
                    block_id=block_id,
                    action_position=action_position,
                    action_id=action_id,
                    boot_id=boot_id,
                    regime=regimes[regime_id],
                    content_seed=content_seed,
                    round_index=round_index,
                )
                for regime_id in REGIME_ORDER
                for content_seed in CONTENT_SEEDS
                for round_index in ROUNDS
            ]
            plan = {
                "schema_version": 1,
                "capture_plan_contract_id": PLAN_CONTRACT_ID,
                "boot_id": boot_id,
                "boot_action_id": action_id,
                "logical_draft_weight_version": logical_version,
                "minimum_shared_kv_blocks": MINIMUM_SHARED_KV_BLOCKS,
                "cells": cells,
            }
            boot = action_boots[action_id]
            plan_path = output_dir / "plans" / f"{boot_id}.json"
            capture_dir = output_dir / "captures" / boot_id
            base_environment = dict(authorization["run_contract"]["environment"])
            if base_environment.get("VLLM_SELF_SPEC_DRAFT_PARTIAL_REPLICA") == "0":
                base_environment["VLLM_SELF_SPEC_DRAFT_PARTIAL_REPLICA"] = ""
            _require(
                NATIVE_SAMPLER_ENV not in base_environment,
                "frozen environment unexpectedly defines the V4 sampler policy",
            )
            _require(
                str(
                    base_environment.get(
                        V1_MULTIPROCESSING_ENV, V1_MULTIPROCESSING_VALUE
                    )
                )
                == V1_MULTIPROCESSING_VALUE,
                "authorization attempts to enable multiprocess EngineCore",
            )
            base_environment[V1_MULTIPROCESSING_ENV] = V1_MULTIPROCESSING_VALUE
            env = {
                **base_environment,
                NATIVE_SAMPLER_ENV: NATIVE_SAMPLER_VALUE,
                "VLLM_SELF_SPEC_DRAFT_KV_WINDOW": str(boot["window_tokens"]),
                "VLLM_SELF_SPEC_DRAFT_KV_SINKS": str(boot["sink_tokens"]),
                "VLLM_SELF_SPEC_P4_CAPTURE_CONFIG": str(plan_path.resolve()),
                "VLLM_SELF_SPEC_P4_CAPTURE_OUTPUT": str(capture_dir.resolve()),
                "VLLM_SELF_SPEC_P4_BOOT_ACTION": action_id,
                "VLLM_SELF_SPEC_P4_LOGICAL_WEIGHT_VERSION": logical_version,
                "VLLM_SELF_SPEC_P4_MIN_KV_BLOCKS": str(MINIMUM_SHARED_KV_BLOCKS),
            }
            specs.append(
                {
                    "schema_version": 1,
                    "boot_id": boot_id,
                    "boot_block_id": block_id,
                    "action_position": action_position,
                    "action_id": action_id,
                    "dynamic_k_schedule": boot["dynamic_k_schedule"],
                    "logical_draft_weight_version": logical_version,
                    "minimum_shared_kv_blocks": MINIMUM_SHARED_KV_BLOCKS,
                    budget_key: budget_evidence,
                    "plan_path": str(plan_path.resolve()),
                    "capture_dir": str(capture_dir.resolve()),
                    "environment": env,
                    "engine": authorization["run_contract"]["engine"],
                    "model": authorization["run_contract"]["model"],
                    "plan": plan,
                }
            )
    _require(len(specs) == 9, "boot-spec construction did not close to nine")
    return specs


def _validate_plan(plan: Mapping[str, Any]) -> None:
    from vllm.v1.spec_decode.koff_runtime import P4SameEventRecorder

    P4SameEventRecorder._validate_plan(
        plan,
        boot_action_id=plan["boot_action_id"],
        logical_weight_version=plan["logical_draft_weight_version"],
        minimum_shared_kv_blocks=plan["minimum_shared_kv_blocks"],
    )


def prepare_run(
    authorization: Mapping[str, Any], output_dir: Path
) -> list[dict[str, Any]]:
    """Create a CPU-only, create-new plan package for all nine boots."""
    _require(not output_dir.exists(), f"refusing to overwrite {output_dir}")
    specs = build_boot_specs(authorization, output_dir)
    output_dir.mkdir(parents=True)
    (output_dir / "plans").mkdir()
    (output_dir / "captures").mkdir()
    (output_dir / "boot_specs").mkdir()
    manifest_rows = []
    for spec in specs:
        plan = spec.pop("plan")
        _validate_plan(plan)
        plan_path = Path(spec["plan_path"])
        plan_path.write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        Path(spec["capture_dir"]).mkdir()
        spec_path = output_dir / "boot_specs" / f"{spec['boot_id']}.json"
        spec_path.write_text(
            json.dumps(spec, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_rows.append(
            {
                "boot_id": spec["boot_id"],
                "action_id": spec["action_id"],
                "boot_spec_path": str(spec_path.resolve()),
                "plan_path": spec["plan_path"],
                "capture_dir": spec["capture_dir"],
            }
        )
    budget_key = (
        "chunked_prefill_budget"
        if "chunked_prefill_budget" in specs[0]
        else "full_prefill_budget"
    )
    preparation = {
        "schema_version": 1,
        "record_type": "p4_b0_capture_runner_preparation",
        "gpu_executed": False,
        "gpu_authority_granted": False,
        "logical_draft_weight_version": logical_weight_version(authorization),
        "minimum_shared_kv_blocks": MINIMUM_SHARED_KV_BLOCKS,
        budget_key: specs[0][budget_key],
        "generation_work_contract": {
            "measurement_currency": "S_dec",
            "measured_decode_tokens_field": "generation.max_output_tokens",
            "prefill_sampled_tokens_per_request": (PREFILL_SAMPLED_TOKENS_PER_REQUEST),
            "sampling_max_tokens_formula": (
                "generation.max_output_tokens + prefill_sampled_tokens_per_request"
            ),
        },
        "sampler_backend": "pytorch_native",
        "flashinfer_sampler_enabled": False,
        "engine_core_mode": "in_process",
        "engine_core_class": INPROCESS_ENGINE_CORE_CLASS,
        "v1_multiprocessing": False,
        "queue_all_before_first_step": True,
        "physical_boot_count": len(specs),
        "cells_per_boot": 48,
        "raw_capture_count": len(specs) * 48,
        "boots": manifest_rows,
    }
    (output_dir / "preparation.json").write_text(
        json.dumps(preparation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return specs


def _boot_child_environment(
    spec_environment: Mapping[str, Any],
    *,
    executable_dir: Path | None = None,
) -> dict[str, str]:
    """Build a child environment with verified tools and sampler policy."""
    _require("PATH" not in spec_environment, "boot spec must not override PATH")
    _require(
        str(spec_environment.get(NATIVE_SAMPLER_ENV, NATIVE_SAMPLER_VALUE))
        == NATIVE_SAMPLER_VALUE,
        "boot spec must not enable the FlashInfer sampler",
    )
    _require(
        str(spec_environment.get(V1_MULTIPROCESSING_ENV, V1_MULTIPROCESSING_VALUE))
        == V1_MULTIPROCESSING_VALUE,
        "boot spec must not enable multiprocess EngineCore",
    )
    executable_dir = executable_dir or Path(sys.prefix) / "bin"
    _require(
        executable_dir.is_dir(),
        f"active environment executable directory is missing: {executable_dir}",
    )
    child_env = os.environ.copy()
    inherited_path = child_env.get("PATH", "")
    child_env["PATH"] = str(executable_dir) + (
        os.pathsep + inherited_path if inherited_path else ""
    )
    child_env[NATIVE_SAMPLER_ENV] = NATIVE_SAMPLER_VALUE
    child_env[V1_MULTIPROCESSING_ENV] = V1_MULTIPROCESSING_VALUE
    ninja = shutil.which("ninja", path=child_env["PATH"])
    _require(ninja is not None, "active environment does not provide ninja")
    ninja_path = Path(ninja)
    _require(
        ninja_path.resolve() == (executable_dir / "ninja").resolve()
        and os.access(ninja_path, os.X_OK),
        "ninja does not resolve to the active environment executable",
    )
    try:
        subprocess.run(
            [ninja, "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise P4RunnerError(
            f"active environment ninja preflight failed: {exc}"
        ) from exc
    child_env.update({key: str(value) for key, value in spec_environment.items()})
    return child_env


def _preflight_native_sampler(
    child_env: Mapping[str, str],
    *,
    python_executable: Path | None = None,
) -> dict[str, Any]:
    """Prove the exact child environment binds the native sampler.

    Args:
        child_env: Environment that will be passed to every physical boot.
        python_executable: Optional interpreter override for CPU-only tests.

    Returns:
        Parsed subprocess evidence for the native sampler binding.

    Raises:
        P4RunnerError: If policy, subprocess execution, or binding drifts.
    """
    _require(
        child_env.get(NATIVE_SAMPLER_ENV) == NATIVE_SAMPLER_VALUE,
        "native-sampler preflight requires VLLM_USE_FLASHINFER_SAMPLER=0",
    )
    executable = python_executable or Path(sys.executable)
    script = "\n".join(
        (
            "import json",
            "from vllm import envs",
            "from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler",
            "sampler = TopKTopPSampler()",
            "bound = getattr(sampler.forward, '__func__', None)",
            "if envs.VLLM_USE_FLASHINFER_SAMPLER:",
            "    raise RuntimeError('FlashInfer sampler remains enabled')",
            "if bound is not TopKTopPSampler.forward_native:",
            "    raise RuntimeError('TopKTopPSampler did not bind forward_native')",
            (
                "print(json.dumps({'backend': 'native', "
                "'flashinfer_enabled': False}, sort_keys=True))"
            ),
        )
    )
    environment = dict(child_env)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        result = subprocess.run(
            [str(executable), "-c", script],
            cwd=REPO_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise P4RunnerError(
            f"native-sampler subprocess preflight failed: {detail.strip()}"
        ) from exc
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    _require(lines, "native-sampler preflight emitted no evidence")
    try:
        evidence = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise P4RunnerError(
            "native-sampler preflight emitted malformed evidence"
        ) from exc
    expected = {"backend": "native", "flashinfer_enabled": False}
    _require(evidence == expected, "native-sampler preflight evidence drifted")
    return evidence


def _preflight_inprocess_engine_core(
    child_env: Mapping[str, str],
    *,
    python_executable: Path | None = None,
) -> dict[str, Any]:
    """Prove that the child imports vLLM with the in-process EngineCore policy."""
    _require(
        child_env.get(V1_MULTIPROCESSING_ENV) == V1_MULTIPROCESSING_VALUE,
        "in-process preflight requires VLLM_ENABLE_V1_MULTIPROCESSING=0",
    )
    executable = python_executable or Path(sys.executable)
    script = "\n".join(
        (
            "import json",
            "from vllm import envs",
            "from vllm.v1.engine.core_client import InprocClient",
            "if envs.VLLM_ENABLE_V1_MULTIPROCESSING:",
            "    raise RuntimeError('V1 multiprocessing remains enabled')",
            (
                "print(json.dumps({'engine_core_class': InprocClient.__name__, "
                "'v1_multiprocessing': False}, sort_keys=True))"
            ),
        )
    )
    environment = dict(child_env)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        result = subprocess.run(
            [str(executable), "-c", script],
            cwd=REPO_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise P4RunnerError(
            f"in-process EngineCore preflight failed: {detail.strip()}"
        ) from exc
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    _require(lines, "in-process EngineCore preflight emitted no evidence")
    try:
        evidence = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise P4RunnerError(
            "in-process EngineCore preflight emitted malformed evidence"
        ) from exc
    expected = {
        "engine_core_class": INPROCESS_ENGINE_CORE_CLASS,
        "v1_multiprocessing": False,
    }
    _require(evidence == expected, "in-process EngineCore evidence drifted")
    return evidence


def _preflight_gpu_identity_and_idle(
    physical_index: int, expected_uuid: str
) -> dict[str, Any]:
    """Fail before output creation if the authorized GPU changed or is busy."""
    _require(
        type(physical_index) is int and physical_index >= 0,
        "authorized physical GPU index is invalid",
    )
    _require(
        isinstance(expected_uuid, str) and expected_uuid.startswith("GPU-"),
        "authorized physical GPU UUID is invalid",
    )
    try:
        identity = subprocess.run(
            [
                "nvidia-smi",
                f"--id={physical_index}",
                "--query-gpu=uuid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        applications = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError) as exc:
        raise P4RunnerError(f"GPU {physical_index} preflight failed: {exc}") from exc
    _require(
        identity == expected_uuid,
        f"GPU {physical_index} identity drifted",
    )
    active_pids = []
    for row in applications:
        fields = [field.strip() for field in row.split(",", maxsplit=1)]
        if len(fields) == 2 and fields[0] == expected_uuid:
            active_pids.append(fields[1])
    _require(
        not active_pids,
        f"GPU {physical_index} has active compute processes: {active_pids}",
    )
    return {
        "physical_gpu_index": physical_index,
        "gpu_uuid": identity,
        "active_compute_processes": 0,
    }


def _preflight_gpu4_identity_and_idle() -> dict[str, Any]:
    """Retain the exact historical GPU-4 preflight entry point."""
    return _preflight_gpu_identity_and_idle(4, GPU4_UUID)


def _total_output_tokens(cell: Mapping[str, Any]) -> int:
    """Return frontend output work including the unmeasured prefill sample."""
    generation = cell.get("generation")
    _require(isinstance(generation, Mapping), "capture cell has no generation work")
    measured_tokens = generation.get("max_output_tokens")
    _require(
        type(measured_tokens) is int and measured_tokens > 0,
        "capture cell measured decode work is invalid",
    )
    return measured_tokens + PREFILL_SAMPLED_TOKENS_PER_REQUEST


def _capture_cohort_metadata(
    cell: Mapping[str, Any], request_ids: Sequence[str]
) -> dict[str, Any]:
    """Build the exact measurement-only scheduler marker for one prompt slice."""
    frozen_request_ids = list(request_ids)
    _require(
        frozen_request_ids and len(set(frozen_request_ids)) == len(frozen_request_ids),
        "capture cohort members are empty or duplicated",
    )
    capture_id = cell.get("capture_id")
    matrix = cell.get("matrix")
    generation = cell.get("generation")
    _require(isinstance(capture_id, str) and capture_id, "capture cell has no id")
    _require(isinstance(matrix, Mapping), "capture cell has no action")
    _require(isinstance(generation, Mapping), "capture cell has no generation work")
    digest = hashlib.sha256(
        json.dumps(frozen_request_ids, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return {
        "contract_id": "p4-capture-cohort-barrier-v1",
        "measurement_only": True,
        "cohort_id": f"{capture_id}:cohort:{digest}",
        "frozen_request_ids": frozen_request_ids,
        "action_id": matrix.get("action_id"),
        "measured_decode_tokens": generation.get("max_output_tokens"),
        "unmeasured_prefill_tokens": PREFILL_SAMPLED_TOKENS_PER_REQUEST,
    }


def _sampling_params(cell: Mapping[str, Any], request_ids: Sequence[str]):
    from vllm import SamplingParams
    from vllm.v1.spec_decode.koff_runtime import P4_CAPTURE_COHORT_EXTRA_ARGS_KEY

    generation = cell["generation"]
    return SamplingParams(
        temperature=generation["temperature"],
        max_tokens=_total_output_tokens(cell),
        ignore_eos=True,
        seed=generation["generation_seed"],
        extra_args={
            P4_CAPTURE_COHORT_EXTRA_ARGS_KEY: _capture_cohort_metadata(
                cell, request_ids
            )
        },
    )


def _engine_args(spec: Mapping[str, Any]):
    from vllm import EngineArgs

    engine = spec["engine"]
    model = spec["model"]
    speculative_config = {
        "method": "draft_model",
        "model": model["snapshot_path"],
        "num_speculative_tokens": engine["num_speculative_tokens"],
        "num_speculative_tokens_per_batch_size": spec["dynamic_k_schedule"],
        "draft_tensor_parallel_size": 1,
    }
    return EngineArgs(
        model=model["snapshot_path"],
        speculative_config=speculative_config,
        tensor_parallel_size=engine["tensor_parallel_size"],
        pipeline_parallel_size=engine["pipeline_parallel_size"],
        max_model_len=engine["max_model_len"],
        max_num_batched_tokens=engine["max_num_batched_tokens"],
        max_num_seqs=engine["max_num_seqs"],
        enable_chunked_prefill=engine["enable_chunked_prefill"],
        gpu_memory_utilization=engine["gpu_memory_utilization"],
        enable_prefix_caching=engine["prefix_caching"],
        async_scheduling=engine["async_scheduling"],
        enforce_eager=engine["enforce_eager"],
        enable_flashinfer_autotune=engine["flashinfer_autotune"],
        seed=engine["generation_seed"],
        disable_log_stats=True,
        generation_config="vllm",
    )


def _run_prompt_chunk(
    engine: Any,
    cell: Mapping[str, Any],
    request_ids: Sequence[str],
    prompt_rows: Mapping[str, Mapping[str, Any]],
) -> None:
    total_output_tokens = _total_output_tokens(cell)
    params = _sampling_params(cell, request_ids)
    for request_id in request_ids:
        engine.add_request(
            request_id,
            {"prompt_token_ids": prompt_rows[request_id]["token_ids"]},
            params,
        )
    _require(
        engine.get_num_unfinished_requests() == len(request_ids),
        "frontend did not queue the complete capture microbatch",
    )
    finished: dict[str, int] = {}
    step_limit = total_output_tokens + 64
    steps = 0
    while engine.has_unfinished_requests():
        for output in engine.step():
            if output.finished:
                _require(output.outputs, f"request {output.request_id} has no output")
                finished[output.request_id] = len(output.outputs[0].token_ids)
        steps += 1
        _require(steps <= step_limit, "capture chunk exceeded its engine-step limit")
    _require(set(finished) == set(request_ids), "capture chunk lost a request")
    _require(
        all(value == total_output_tokens for value in finished.values()),
        "capture chunk did not complete exact frontend work",
    )


def run_boot_child(spec_path: Path) -> None:
    """Run all 48 cells in one engine process after parent authorization."""
    spec = _load_json(spec_path)
    plan = _load_json(Path(spec["plan_path"]))
    _validate_plan(plan)
    _require(
        os.environ.get("VLLM_SELF_SPEC_P4_CAPTURE_CONFIG") == spec["plan_path"],
        "child capture-plan environment differs from its boot spec",
    )
    _require(
        spec["environment"].get(NATIVE_SAMPLER_ENV) == NATIVE_SAMPLER_VALUE
        and os.environ.get(NATIVE_SAMPLER_ENV) == NATIVE_SAMPLER_VALUE,
        "child execution did not preserve the native-sampler policy",
    )
    _require(
        spec["environment"].get(V1_MULTIPROCESSING_ENV) == V1_MULTIPROCESSING_VALUE
        and os.environ.get(V1_MULTIPROCESSING_ENV) == V1_MULTIPROCESSING_VALUE,
        "child execution did not preserve in-process EngineCore mode",
    )
    manifest, prompt_rows = _load_prompt_rows()
    if "chunked_prefill_budget" in spec:
        _require(
            spec["chunked_prefill_budget"]
            == chunked_prefill_budget_evidence(
                manifest,
                prompt_rows,
                spec["engine"],
            ),
            "child chunked-prefill evidence differs from its boot spec",
        )
        _require(
            "full_prefill_budget" not in spec,
            "V10 child boot spec retained the rejected full-prefill contract",
        )
    else:
        _require(
            spec.get("full_prefill_budget")
            == full_prefill_budget_evidence(manifest, prompt_rows, spec["engine"]),
            "child full-prefill evidence differs from its boot spec",
        )
    from vllm import LLMEngine, envs
    from vllm.v1.engine.core_client import InprocClient

    _require(
        not envs.VLLM_ENABLE_V1_MULTIPROCESSING,
        "child imported vLLM with V1 multiprocessing enabled",
    )
    engine = LLMEngine.from_engine_args(_engine_args(spec))
    _require(
        isinstance(engine.engine_core, InprocClient),
        "child engine did not construct InprocClient",
    )
    try:
        for cell in plan["cells"]:
            prompt_ids = cell["generation"]["prompt_record_ids"]
            batch = cell["generation"]["batch"]
            for start in range(0, len(prompt_ids), batch):
                _run_prompt_chunk(
                    engine,
                    cell,
                    prompt_ids[start : start + batch],
                    prompt_rows,
                )
    finally:
        engine.engine_core.shutdown()
    captures = sorted(Path(spec["capture_dir"]).glob("*.json"))
    _require(len(captures) == 48, "physical boot did not emit 48 captures")
    _require(
        all(path.stat().st_size > 0 for path in captures),
        "physical boot left an empty capture placeholder",
    )


def _adapt_and_score(output_dir: Path) -> None:
    from adapt_p4_b0_same_event import adapt_capture
    from score_p4_b0 import score_rounds

    captures = sorted((output_dir / "captures").glob("*/*.json"))
    _require(len(captures) == 432, "runner did not produce exactly 432 captures")
    rounds = [adapt_capture(_load_json(path)) for path in captures]
    rounds_path = output_dir / "adapted_rounds.jsonl"
    rounds_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            + "\n"
            for row in rounds
        ),
        encoding="utf-8",
    )
    result = score_rounds(_load_json(SCORER_CONTRACT_PATH), rounds)
    _require(
        result.get("decision", {}).get("authority_granted") is False,
        "the frozen score attempted to grant authority",
    )
    (output_dir / "score.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _reviewed_output_path(authorization_path: Path) -> Path:
    """Resolve an exact reviewed authorization path to its paired output."""
    resolved_authorization_path = authorization_path.resolve()
    reviewed_pairs = (
        (V11_AUTHORIZATION_PATH, V11_OUTPUT_PATH),
        (V10_AUTHORIZATION_PATH, V10_OUTPUT_PATH),
        (V9_AUTHORIZATION_PATH, V9_OUTPUT_PATH),
        (V8_AUTHORIZATION_PATH, V8_OUTPUT_PATH),
        (V7_AUTHORIZATION_PATH, V7_OUTPUT_PATH),
        (V6_AUTHORIZATION_PATH, V6_OUTPUT_PATH),
    )
    for reviewed_authorization, reviewed_output in reviewed_pairs:
        if resolved_authorization_path == _repository_path(reviewed_authorization):
            return _repository_path(reviewed_output)
    raise P4RunnerError("execution must use a reviewed authorization path")


def execute_run(
    authorization_path: Path,
    authorization: Mapping[str, Any],
    output_dir: Path,
) -> None:
    """Launch exactly one child process for each authorized physical boot."""
    reviewed_output_path = _reviewed_output_path(authorization_path)
    _require(
        output_dir.resolve() == reviewed_output_path,
        "execution output differs from the reviewed create-new path",
    )
    validate_execution_authority(authorization)
    base_child_env = _boot_child_environment({})
    _preflight_native_sampler(base_child_env)
    _preflight_inprocess_engine_core(base_child_env)
    gpu_policy = authorization.get("execution_policy", {})
    physical_gpu_index = gpu_policy.get("physical_gpu_index", 4)
    physical_gpu_uuid = gpu_policy.get("physical_gpu_uuid", GPU4_UUID)
    if physical_gpu_index == 4 and physical_gpu_uuid == GPU4_UUID:
        _preflight_gpu4_identity_and_idle()
    else:
        _preflight_gpu_identity_and_idle(
            physical_gpu_index,
            physical_gpu_uuid,
        )
    specs = prepare_run(authorization, output_dir)
    script_path = Path(__file__).resolve()
    for spec in specs:
        spec_path = output_dir / "boot_specs" / f"{spec['boot_id']}.json"
        child_env = base_child_env.copy()
        child_env.update(
            {key: str(value) for key, value in spec["environment"].items()}
        )
        subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--authorization",
                str(authorization_path.resolve()),
                "--output-dir",
                str(output_dir.resolve()),
                "--child-spec",
                str(spec_path.resolve()),
            ],
            cwd=REPO_ROOT,
            env=child_env,
            check=True,
        )
    _adapt_and_score(output_dir)


def parse_args() -> argparse.Namespace:
    """Parse the held runner interface and CPU-only preparation option."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--child-spec", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    """Prepare without GPU, or execute only under a future explicit approval."""
    args = parse_args()
    package = _load_json(args.authorization)
    authorization = resolve_authorization_package(
        package,
        require_output_absent=args.child_spec is None,
    )
    if args.prepare_only:
        _require(args.child_spec is None, "child mode cannot prepare a new run")
        validate_preparation_contract(authorization)
        prepare_run(authorization, args.output_dir)
        print(
            json.dumps(
                {
                    "status": "pass",
                    "mode": "prepare_only",
                    "gpu_executed": False,
                    "gpu_authority_granted": False,
                    "output_dir": str(args.output_dir),
                },
                sort_keys=True,
            )
        )
        return 0

    validate_execution_authority(authorization)
    if args.child_spec is not None:
        approved_output = _reviewed_output_path(args.authorization)
        _require(
            args.output_dir.resolve() == approved_output,
            "child execution output differs from the reviewed path",
        )
        _require(
            args.child_spec.resolve().parent == approved_output / "boot_specs",
            "child spec is outside the reviewed matrix package",
        )
        run_boot_child(args.child_spec)
    else:
        execute_run(args.authorization, authorization, args.output_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except P4RunnerError as exc:
        print(f"P4 runner refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
