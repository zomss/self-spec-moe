#!/usr/bin/env python3
"""Validate the request-ID-repaired one-shot non-scored P4 probe V5 authority."""

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
    PHASE_DIR / "data" / "p4" / "p4_b0_chunked_prefill_probe_authorization_v5.json"
)
OUTPUT_DIR = PHASE_DIR / "data" / "p4" / "run_b0_chunked_prefill_probe_v5"
VALIDATION_PATH = (
    PHASE_DIR
    / "data"
    / "p4"
    / "p4_b0_chunked_prefill_probe_authorization_v5_validation.json"
)

EXPECTED_SOURCE_PATHS = {
    "authorization_schema": (
        "research/97_composition_runtime/schemas/"
        "p4_b0_chunked_prefill_probe_authorization_v5.schema.json"
    ),
    "authorization_tests": (
        "research/97_composition_runtime/tests/"
        "test_p4_b0_chunked_prefill_probe_authorization.py"
    ),
    "authorization_validator": (
        "research/97_composition_runtime/scripts/"
        "validate_p4_b0_chunked_prefill_probe_authorization_v5.py"
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
    "cohort_barrier_implementation_tests": (
        "tests/v1/spec_decode/test_koff_runtime.py"
    ),
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
    "v4_authorization": (
        "research/97_composition_runtime/data/p4/"
        "p4_b0_chunked_prefill_probe_authorization_v4.json"
    ),
    "v4_consumed_failure": (
        "research/97_composition_runtime/data/p4/"
        "run_b0_chunked_prefill_probe_v4/failure.json"
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
    "pure_decode_draft_step0_query_width": 1,
    "prefill_draft_step0_query_width_positive": True,
    "non_decode_k4_requires_pure_prefill_cohort_arm": True,
    "canonical_probe_result_request_ids": True,
    "randomized_internal_request_ids_preserved": True,
}


def _validate_sources(value: Mapping[str, Any]) -> dict[str, str]:
    _require(
        set(value) == set(EXPECTED_SOURCE_PATHS),
        "V5 probe source roles drifted",
    )
    observed = {}
    for role, expected_path in EXPECTED_SOURCE_PATHS.items():
        reference = value[role]
        _require(
            isinstance(reference, Mapping) and set(reference) == {"path", "sha256"},
            f"V5 probe source reference is malformed: {role}",
        )
        _require(
            reference["path"] == expected_path,
            f"V5 probe source path drifted: {role}",
        )
        path = _repo_path(expected_path)
        _require(path.is_file(), f"V5 probe source is missing: {role}")
        actual = _sha256(path)
        _require(
            reference["sha256"] == actual,
            f"V5 probe source hash drifted: {role}",
        )
        observed[role] = actual
    return observed


def _validate_bound_file(reference: Mapping[str, Any], role: str) -> None:
    _require(
        set(reference) >= {"path", "sha256"},
        f"V4 failure artifact reference is malformed: {role}",
    )
    path = _repo_path(reference["path"])
    _require(path.is_file(), f"V4 failure artifact is missing: {role}")
    _require(
        _sha256(path) == reference["sha256"],
        f"V4 failure artifact hash drifted: {role}",
    )


def _validate_v4_failure(source_hashes: Mapping[str, str]) -> None:
    prior = _load_json(_repo_path(EXPECTED_SOURCE_PATHS["v4_authorization"]))
    _require(
        prior.get("schema_version") == 4
        and prior.get("package_id") == "p4-b0-chunked-prefill-probe-authorization-v4"
        and prior.get("authorizations", {}).get("gpu4_non_scored_probe_v4") is True,
        "V5 does not bind the exact consumed V4 authority",
    )

    failure = _load_json(_repo_path(EXPECTED_SOURCE_PATHS["v4_consumed_failure"]))
    authorization = failure.get("authorization", {})
    attempt = failure.get("attempt", {})
    diagnostic = failure.get("diagnostic", {})
    runtime = diagnostic.get("runtime_evidence", {})
    progress = diagnostic.get("probe_validator_progress", {})
    identity = diagnostic.get("r4_identity_evidence", {})
    disposition = failure.get("disposition", {})
    output = failure.get("output", {})

    _require(
        failure.get("record_type") == "p4_b0_chunked_prefill_probe_execution_failure"
        and authorization.get("package_id")
        == "p4-b0-chunked-prefill-probe-authorization-v4"
        and authorization.get("sha256") == source_hashes["v4_authorization"],
        "V5 failure does not reference the bound V4 authorization",
    )
    _require(
        attempt.get("invocation_started") is True
        and attempt.get("exit_code") == 1
        and attempt.get("gpu_model_executed") is True
        and attempt.get("engine_initialization_completed") is True
        and attempt.get("physical_boots_started") == 1
        and attempt.get("completed_probe_boots") == 0
        and attempt.get("all_three_cohorts_submitted") is True
        and attempt.get("complete_cohort_histories_observed_in_child") == 3
        and attempt.get("same_event_records_observed_in_child") == 3
        and attempt.get("probe_result_emitted") is False
        and attempt.get("score_emitted") is False,
        "V5 source failure is not the consumed post-cohort V4 refusal",
    )
    _require(
        diagnostic.get("scope") == "probe_result_request_id_canonicalization_omission"
        and diagnostic.get("last_stage")
        == "post_execution_probe_evidence_validation_after_three_complete_cohorts",
        "V5 source failure has another diagnostic scope",
    )
    _require(
        runtime.get("configured_max_num_batched_tokens") == 8192
        and runtime.get("effective_max_num_scheduled_tokens") == 8160
        and runtime.get("max_num_seqs") == 32
        and runtime.get("trace_record_count") == 40
        and runtime.get("trace_engine_step_count") == 39
        and runtime.get("gpu_kv_cache_blocks") == 24527
        and runtime.get("minimum_shared_kv_blocks") == 21682
        and runtime.get("resource_precheck_passed") is True
        and runtime.get("total_preemptions") == 0
        and runtime.get("total_recomputed_tokens") == 0
        and runtime.get("all_trace_closure_checks_passed") is True,
        "V5 lost the V4 resource or execution diagnostics",
    )
    cohort_evidence = diagnostic.get("cohort_evidence")
    _require(
        isinstance(cohort_evidence, list)
        and [row.get("regime_id") for row in cohort_evidence] == ["R4", "R5", "R5cot"]
        and [row.get("decode_engine_step_index") for row in cohort_evidence]
        == [8, 23, 38]
        and all(
            row.get("verified_action_id") == "target-matching-k4"
            and row.get("pure_decode") is True
            and row.get("draft_step0_query_width") == 1
            and row.get("draft_output_width") == 4
            and row.get("request_count") == 8
            and row.get("preemptions") == 0
            and row.get("recomputed_tokens") == 0
            and row.get("closure_holds") is True
            for row in cohort_evidence
        ),
        "V5 lost the three V4 cohort diagnostics",
    )
    _require(
        progress.get("expected_history_count_passed") is True
        and progress.get("expected_event_count_passed") is True
        and progress.get("r4_complete_history_contract_passed") is True
        and progress.get("r4_exact_committed_work_passed") is True
        and progress.get("r4_pure_decode_quality_passed") is True
        and progress.get("exception_message")
        == "probe cohort R4 target rows changed identity"
        and identity.get("all_suffixes_match_eight_lowercase_hex") is True
        and identity.get("canonicalized_set_matches_frozen_set") is True
        and identity.get("identity_or_work_was_actually_lost") is False,
        "V5 does not bind the exact V4 request-ID diagnosis",
    )

    artifacts = failure.get("artifacts", {})
    for role in ("preparation", "child_log", "koff_trace"):
        reference = artifacts.get(role)
        _require(isinstance(reference, Mapping), f"V4 failure lost {role}")
        _validate_bound_file(reference, role)
    _require(
        artifacts.get("probe_result_present") is False
        and disposition.get("authorization_consumed") is True
        and disposition.get("requires_fresh_authorization") is True
        and disposition.get("retry_attempted") is False
        and disposition.get("resume_attempted") is False
        and disposition.get("fallback_gpu_used") is False
        and disposition.get("scoring_allowed") is False
        and disposition.get("v10_authorized") is False
        and disposition.get("p4a_authorized") is False
        and output.get("probe_result_created") is False
        and output.get("score_created") is False,
        "V4 failure permits reuse, retry, fallback, scoring, or V10",
    )
    _require(
        not (_repo_path(output["path"]) / "probe_result.json").exists(),
        "consumed V4 output unexpectedly acquired a probe result",
    )


def _validate_v5_contract(
    contract: Mapping[str, Any],
    *,
    authorization_path: Path,
    output_dir: Path,
) -> None:
    _require(
        contract.get("pass_gates") == EXPECTED_PASS_GATES,
        "V5 probe pass gates do not preserve separate budgets, query phases, "
        "and canonical IDs",
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
    """Validate exact V5 authority without executing a GPU command."""
    _require(
        authorization_path.resolve() == AUTHORIZATION_PATH.resolve(),
        "V5 probe must use the reviewed authorization path",
    )
    _require(
        output_dir.resolve() == OUTPUT_DIR.resolve(),
        "V5 probe must use the reviewed create-new output path",
    )
    if require_output_absent:
        _require(
            not output_dir.exists(),
            "V5 probe output already exists; retry forbidden",
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
        "V5 probe authorization fields drifted",
    )
    _require(
        authorization["schema_version"] == 5
        and authorization["package_id"]
        == "p4-b0-chunked-prefill-probe-authorization-v5"
        and authorization["date"] == "2026-08-09"
        and authorization["status"]
        == "authorized_gpu4_non_scored_chunked_prefill_probe_v5_only",
        "V5 probe authorization identity drifted",
    )
    source_hashes = _validate_sources(authorization["source_artifacts"])
    _validate_v4_failure(source_hashes)
    _validate_v5_contract(
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
            "reuse_v3_output_allowed": False,
            "reuse_v4_output_allowed": False,
            "reuse_v9_captures_allowed": False,
            "fallback_gpu_allowed": False,
            "score_output": False,
        },
        "V5 execution policy permits retry, reuse, fallback, or scoring",
    )
    _require(
        authorization["decision"]
        == {
            "decision": "authorize",
            "scope": "one_non_scored_gpu4_probe_v5",
            "reason": (
                "v4_probe_result_request_id_canonicalization_repaired_and_cpu_proven"
            ),
        },
        "V5 probe decision is not narrowly authorized",
    )
    _require(
        authorization["claims"]
        == {
            "live_engine_wired": True,
            "relative_path_dispatch_repaired": True,
            "configured_effective_budget_repaired": True,
            "prefill_query_width_phase_repaired": True,
            "prefill_query_width_execution_path_tested": True,
            "pure_decode_query_width_fail_closed": True,
            "probe_result_request_id_canonicalization_repaired": True,
            "randomized_internal_request_ids_preserved": True,
            "probe_result_request_id_fail_closed_tested": True,
            "v4_three_cohort_diagnostic_evidence": True,
            "real_draft_model_normalization_tested": True,
            "ordinary_serving_unchanged": True,
            "gpu_resource_fit_proven": False,
            "value_screen_run_ready": False,
            "p4a_ready": False,
            "performance_claim_allowed": False,
        },
        "V5 probe claims drifted",
    )
    _require(
        authorization["authorizations"]
        == {
            "gpu4_non_scored_probe_v5": True,
            "v10_value_screen": False,
            "p4a_engineering": False,
            "action_admission": False,
            "production_value_claim": False,
        },
        "V5 package grants authority beyond one non-scored GPU-4 probe",
    )
    _require(
        authorization["next_artifact"]
        == {
            "kind": "p4_b0_chunked_prefill_gpu_probe_result_v5",
            "path": (
                "research/97_composition_runtime/data/p4/"
                "run_b0_chunked_prefill_probe_v5/probe_result.json"
            ),
            "on_pass": "draft_separate_source_bound_v10_authorization",
            "on_fail": "diagnose_without_retry_or_fallback",
            "v10_authorized_here": False,
        },
        "V5 probe next-artifact boundary drifted",
    )
    return {
        "artifact_id": authorization["package_id"],
        "status": "pass",
        "source_hashes": source_hashes,
        "v4_failure_bound": True,
        "v4_complete_cohort_histories": 3,
        "configured_max_num_batched_tokens": 8192,
        "effective_max_num_scheduled_tokens": 8160,
        "pure_decode_draft_step0_query_width": 1,
        "prefill_query_width_phase_repaired": True,
        "probe_result_request_id_canonicalization_repaired": True,
        "randomized_internal_request_ids_preserved": True,
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
        raise SystemExit(f"P4 chunked-prefill probe V5 rejected: {exc}") from exc
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        print(payload, end="")
    else:
        args.out.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
