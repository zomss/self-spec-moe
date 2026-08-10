#!/usr/bin/env python3
"""Validate the budget-repaired one-shot non-scored P4 probe V3 authority."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from validate_p4_b0_chunked_prefill_probe_authorization import (
    P4ChunkedPrefillAuthorizationError,
    _load_json,
    _repo_path,
    _require,
    _sha256,
    _validate_contract,
)

PHASE_DIR = Path(__file__).resolve().parents[1]
AUTHORIZATION_PATH = (
    PHASE_DIR / "data" / "p4" / "p4_b0_chunked_prefill_probe_authorization_v3.json"
)
OUTPUT_DIR = PHASE_DIR / "data" / "p4" / "run_b0_chunked_prefill_probe_v3"
VALIDATION_PATH = (
    PHASE_DIR
    / "data"
    / "p4"
    / "p4_b0_chunked_prefill_probe_authorization_v3_validation.json"
)

EXPECTED_SOURCE_PATHS = {
    "authorization_schema": (
        "research/97_composition_runtime/schemas/"
        "p4_b0_chunked_prefill_probe_authorization_v3.schema.json"
    ),
    "authorization_tests": (
        "research/97_composition_runtime/tests/"
        "test_p4_b0_chunked_prefill_probe_authorization.py"
    ),
    "authorization_validator": (
        "research/97_composition_runtime/scripts/"
        "validate_p4_b0_chunked_prefill_probe_authorization_v3.py"
    ),
    "authorization_validator_base": (
        "research/97_composition_runtime/scripts/"
        "validate_p4_b0_chunked_prefill_probe_authorization.py"
    ),
    "capture_runner": (
        "research/97_composition_runtime/scripts/run_p4_b0_value_screen.py"
    ),
    "capture_runner_tests": (
        "research/97_composition_runtime/tests/test_p4_b0_capture_runner.py"
    ),
    "cohort_barrier_cpu_proof": (
        "research/97_composition_runtime/data/p4/"
        "p4_b0_chunked_prefill_cohort_barrier_design_and_cpu_proof.json"
    ),
    "cohort_barrier_implementation": "vllm/v1/spec_decode/koff_runtime.py",
    "gpu_worker": "vllm/v1/worker/gpu_worker.py",
    "probe_runner": (
        "research/97_composition_runtime/scripts/run_p4_b0_chunked_prefill_probe.py"
    ),
    "prompt_bundle": (
        "research/97_composition_runtime/data/p4/p4_b0_prompt_tokens.jsonl.gz"
    ),
    "prompt_manifest": (
        "research/97_composition_runtime/data/p4/p4_b0_prompt_manifest.json"
    ),
    "scheduler": "vllm/v1/core/sched/scheduler.py",
    "scheduler_test_utils": "tests/v1/core/utils.py",
    "scheduler_tests": "tests/v1/core/test_scheduler.py",
    "transient_bound": (
        "research/97_composition_runtime/data/p4/"
        "p4_b0_full_prefill_transient_bound.json"
    ),
    "v2_authorization": (
        "research/97_composition_runtime/data/p4/"
        "p4_b0_chunked_prefill_probe_authorization_v2.json"
    ),
    "v2_consumed_failure": (
        "research/97_composition_runtime/data/p4/"
        "run_b0_chunked_prefill_probe_v2/failure.json"
    ),
    "v9_consumed_failure": (
        "research/97_composition_runtime/data/p4/run_b0_value_screen_v8/failure.json"
    ),
}

LEGACY_PASS_GATES = {
    "actual_cuda_graph_memory_bytes_positive": True,
    "minimum_shared_target_kv_blocks": 21682,
    "pure_first_measured_decode_every_cohort": True,
    "zero_preemption": True,
    "zero_recomputation": True,
    "zero_invalid_spec_tokens": True,
    "exact_one_plus_one_token_accounting": True,
}
EXPECTED_PASS_GATES = {
    **LEGACY_PASS_GATES,
    "configured_max_num_batched_tokens": 8192,
    "effective_max_num_scheduled_tokens": 8160,
}


def _validate_sources(value: Mapping[str, Any]) -> dict[str, str]:
    _require(
        set(value) == set(EXPECTED_SOURCE_PATHS),
        "V3 probe source roles drifted",
    )
    observed = {}
    for role, expected_path in EXPECTED_SOURCE_PATHS.items():
        reference = value[role]
        _require(
            isinstance(reference, Mapping) and set(reference) == {"path", "sha256"},
            f"V3 probe source reference is malformed: {role}",
        )
        _require(
            reference["path"] == expected_path,
            f"V3 probe source path drifted: {role}",
        )
        path = _repo_path(expected_path)
        _require(path.is_file(), f"V3 probe source is missing: {role}")
        actual = _sha256(path)
        _require(
            reference["sha256"] == actual,
            f"V3 probe source hash drifted: {role}",
        )
        observed[role] = actual
    return observed


def _validate_v2_failure(source_hashes: Mapping[str, str]) -> None:
    failure = _load_json(_repo_path(EXPECTED_SOURCE_PATHS["v2_consumed_failure"]))
    authorization = failure.get("authorization", {})
    attempt = failure.get("attempt", {})
    diagnostic = failure.get("diagnostic", {})
    budgets = diagnostic.get("budget_evidence", {})
    disposition = failure.get("disposition", {})
    _require(
        failure.get("record_type") == "p4_b0_chunked_prefill_probe_execution_failure"
        and authorization.get("package_id")
        == "p4-b0-chunked-prefill-probe-authorization-v2"
        and authorization.get("sha256") == source_hashes["v2_authorization"],
        "V3 does not bind the exact consumed V2 authority",
    )
    _require(
        attempt.get("invocation_started") is True
        and attempt.get("gpu_model_executed") is True
        and attempt.get("engine_initialization_completed") is True
        and attempt.get("physical_boots_started") == 1
        and attempt.get("first_request_registration_attempted") is True
        and attempt.get("first_request_scheduled") is False
        and attempt.get("complete_cohorts_emitted") == 0
        and attempt.get("probe_result_emitted") is False
        and attempt.get("score_emitted") is False,
        "V3 source failure is not the consumed post-boot V2 refusal",
    )
    _require(
        diagnostic.get("scope")
        == "configured_vs_effective_chunked_prefill_budget_confusion"
        and budgets.get("configured_max_num_batched_tokens") == 8192
        and budgets.get("speculative_slot_reserve_tokens") == 32
        and budgets.get("effective_max_num_scheduled_tokens") == 8160,
        "V3 does not bind the exact V2 budget diagnosis",
    )
    _require(
        disposition.get("authorization_consumed") is True
        and disposition.get("requires_fresh_authorization") is True
        and disposition.get("retry_attempted") is False
        and disposition.get("resume_attempted") is False
        and disposition.get("fallback_gpu_used") is False
        and disposition.get("scoring_allowed") is False
        and disposition.get("v10_authorized") is False,
        "V3 source failure permits reuse, retry, scoring, or V10",
    )


def _validate_v3_contract(
    contract: Mapping[str, Any],
    *,
    authorization_path: Path,
    output_dir: Path,
) -> None:
    _require(
        contract.get("pass_gates") == EXPECTED_PASS_GATES,
        "V3 probe pass gates do not separate configured and effective budgets",
    )
    legacy_contract = dict(contract)
    legacy_contract["pass_gates"] = LEGACY_PASS_GATES
    _validate_contract(
        legacy_contract,
        authorization_path=authorization_path,
        output_dir=output_dir,
    )


def validate_authorization(
    authorization: Mapping[str, Any],
    *,
    authorization_path: Path = AUTHORIZATION_PATH,
    output_dir: Path = OUTPUT_DIR,
    require_output_absent: bool = True,
) -> dict[str, Any]:
    """Validate exact V3 authority without executing a GPU command."""
    _require(
        authorization_path.resolve() == AUTHORIZATION_PATH.resolve(),
        "V3 probe must use the reviewed authorization path",
    )
    _require(
        output_dir.resolve() == OUTPUT_DIR.resolve(),
        "V3 probe must use the reviewed create-new output path",
    )
    if require_output_absent:
        _require(
            not output_dir.exists(),
            "V3 probe output already exists; retry forbidden",
        )
    _require(
        set(authorization)
        == {
            "schema_version",
            "package_id",
            "date",
            "status",
            "source_artifacts",
            "run_contract",
            "execution_policy",
            "decision",
            "claims",
            "authorizations",
            "next_artifact",
        },
        "V3 probe authorization fields drifted",
    )
    _require(
        authorization["schema_version"] == 3
        and authorization["package_id"]
        == "p4-b0-chunked-prefill-probe-authorization-v3"
        and authorization["date"] == "2026-08-09"
        and authorization["status"]
        == "authorized_gpu4_non_scored_chunked_prefill_probe_v3_only",
        "V3 probe authorization identity drifted",
    )
    source_hashes = _validate_sources(authorization["source_artifacts"])
    _validate_v2_failure(source_hashes)
    _validate_v3_contract(
        authorization["run_contract"],
        authorization_path=authorization_path,
        output_dir=output_dir,
    )
    _require(
        authorization["execution_policy"]
        == {
            "create_new_output_only": True,
            "single_parent_launch": True,
            "single_child_boot": True,
            "retry_allowed": False,
            "resume_allowed": False,
            "reuse_v1_output_allowed": False,
            "reuse_v2_output_allowed": False,
            "reuse_v9_captures_allowed": False,
            "fallback_gpu_allowed": False,
            "score_output": False,
        },
        "V3 execution policy permits retry, reuse, fallback, or scoring",
    )
    _require(
        authorization["decision"]
        == {
            "decision": "authorize",
            "scope": "one_non_scored_gpu4_probe_v3",
            "reason": "v2_budget_guard_repaired_and_real_config_cpu_proven",
        },
        "V3 probe decision is not narrowly authorized",
    )
    _require(
        authorization["claims"]
        == {
            "live_engine_wired": True,
            "relative_path_dispatch_repaired": True,
            "configured_effective_budget_repaired": True,
            "real_draft_model_normalization_tested": True,
            "ordinary_serving_unchanged": True,
            "gpu_resource_fit_proven": False,
            "value_screen_run_ready": False,
            "p4a_ready": False,
            "performance_claim_allowed": False,
        },
        "V3 probe claims drifted",
    )
    _require(
        authorization["authorizations"]
        == {
            "gpu4_non_scored_probe_v3": True,
            "v10_value_screen": False,
            "p4a_engineering": False,
            "action_admission": False,
            "production_value_claim": False,
        },
        "V3 package grants authority beyond one non-scored GPU-4 probe",
    )
    _require(
        authorization["next_artifact"]
        == {
            "kind": "p4_b0_chunked_prefill_gpu_probe_result_v3",
            "path": (
                "research/97_composition_runtime/data/p4/"
                "run_b0_chunked_prefill_probe_v3/probe_result.json"
            ),
            "on_pass": "draft_separate_source_bound_v10_authorization",
            "on_fail": "diagnose_without_retry_or_fallback",
            "v10_authorized_here": False,
        },
        "V3 probe next-artifact boundary drifted",
    )
    return {
        "artifact_id": authorization["package_id"],
        "status": "pass",
        "source_hashes": source_hashes,
        "v2_failure_bound": True,
        "configured_max_num_batched_tokens": 8192,
        "effective_max_num_scheduled_tokens": 8160,
        "gpu_authority_granted": True,
        "gpu_executed": False,
        "scoring_authorized": False,
        "v10_authorized": False,
        "output_absent": not output_dir.exists(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, default=AUTHORIZATION_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = validate_authorization(
            _load_json(args.authorization),
            authorization_path=args.authorization,
            output_dir=args.output_dir,
        )
    except P4ChunkedPrefillAuthorizationError as exc:
        raise SystemExit(f"P4 chunked-prefill probe V3 rejected: {exc}") from exc
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        print(payload, end="")
    else:
        args.out.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
