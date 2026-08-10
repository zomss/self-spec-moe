#!/usr/bin/env python3
"""Validate the Phase 97 chunked-prefill capture-cohort barrier proof."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from vllm.v1.spec_decode.koff_runtime import (
    K4_ACTION_ID,
    OFF_ACTION_ID,
    W512_ACTION_ID,
    KOffRuntimeError,
    P4CaptureCohortBarrier,
    P4CohortRequestState,
)

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]

MAX_NUM_BATCHED_TOKENS = 8192
MAX_NUM_SEQS = 32
SPECULATIVE_SLOT_RESERVE = 32
EFFECTIVE_SCHEDULER_TOKEN_BUDGET = 8160
MINIMUM_SHARED_KV_BLOCKS = 21682
CPU_PROOF_DECODE_TOKENS = 512
PREFILL_SAMPLE_TOKENS = 1
PROMPT_COUNTS = (6, 10, 14)
FROZEN_IDS = ("prompt-a", "prompt-b", "prompt-c")
ACTION_WIDTHS = {
    OFF_ACTION_ID: 1,
    K4_ACTION_ID: 5,
    W512_ACTION_ID: 5,
}

EXPECTED_SOURCE_PATHS = {
    "transient_bound": (
        "research/97_composition_runtime/data/p4/"
        "p4_b0_full_prefill_transient_bound.json"
    ),
    "v9_authorization": (
        "research/97_composition_runtime/data/p4/p4_b0_run_authorization_v9.json"
    ),
    "v9_failure": (
        "research/97_composition_runtime/data/p4/run_b0_value_screen_v8/failure.json"
    ),
    "prompt_manifest": (
        "research/97_composition_runtime/data/p4/p4_b0_prompt_manifest.json"
    ),
    "serving_chunked_prefill_diagnosis": (
        "research/97_composition_runtime/data/p4/"
        "run_b0_serving_chunked_prefill_diagnosis_v1/diagnosis.json"
    ),
    "cohort_barrier_implementation": "vllm/v1/spec_decode/koff_runtime.py",
    "cpu_proof_validator": (
        "research/97_composition_runtime/scripts/"
        "validate_p4_b0_chunked_prefill_cohort_barrier.py"
    ),
    "cpu_proof_tests": (
        "research/97_composition_runtime/tests/"
        "test_p4_b0_chunked_prefill_cohort_barrier.py"
    ),
}

EXPECTED_TOP_LEVEL_KEYS = {
    "schema_version",
    "artifact_id",
    "date",
    "status",
    "evidence_grade",
    "source_artifacts",
    "rejected_mechanism",
    "bounded_prefill_contract",
    "cohort_membership_contract",
    "state_machine",
    "decode_release_contract",
    "accounting_contract",
    "failure_policy",
    "ordinary_serving_boundary",
    "cpu_proof",
    "decision",
    "claims",
    "authorizations",
    "next_artifact",
}


class CohortBarrierProofError(ValueError):
    """Raised when the cohort-barrier design or proof fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CohortBarrierProofError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CohortBarrierProofError(
            f"cannot load JSON artifact {path}: {exc}"
        ) from exc
    _require(isinstance(value, dict), f"JSON artifact must be an object: {path}")
    return value


def _repository_path(relative_path: str) -> Path:
    path = (REPO_ROOT / relative_path).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise CohortBarrierProofError(f"artifact escapes repository: {path}") from exc
    return path


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise CohortBarrierProofError(
            f"cannot hash source artifact {path}: {exc}"
        ) from exc


def _runtime_ids(tag: int = 0) -> tuple[str, ...]:
    return tuple(
        f"{frozen_id}-{tag * len(FROZEN_IDS) + index:08x}"
        for index, frozen_id in enumerate(FROZEN_IDS)
    )


def _states(
    runtime_ids: Sequence[str],
    computed: Sequence[int],
    outputs: Sequence[int],
    *,
    preemptions: Sequence[int] | None = None,
) -> list[P4CohortRequestState]:
    preemption_counts = preemptions or [0] * len(runtime_ids)
    return [
        P4CohortRequestState(
            request_id=request_id,
            num_prompt_tokens=prompt_tokens,
            num_computed_tokens=computed_tokens,
            num_output_tokens=output_tokens,
            num_preemptions=preemption_count,
        )
        for (
            request_id,
            prompt_tokens,
            computed_tokens,
            output_tokens,
            preemption_count,
        ) in zip(
            runtime_ids,
            PROMPT_COUNTS,
            computed,
            outputs,
            preemption_counts,
            strict=True,
        )
    ]


def _new_barrier(
    action_id: str = OFF_ACTION_ID,
    *,
    measured_decode_tokens: int = CPU_PROOF_DECODE_TOKENS,
    tag: int = 0,
) -> tuple[P4CaptureCohortBarrier, tuple[str, ...]]:
    runtime_ids = _runtime_ids(tag)
    barrier = P4CaptureCohortBarrier(
        f"cpu-proof-{action_id}-{tag}",
        FROZEN_IDS,
        action_id=action_id,
        measured_decode_tokens=measured_decode_tokens,
    )
    for request_id in runtime_ids:
        barrier.register(request_id)
    barrier.seal()
    return barrier, runtime_ids


def _open_release(
    barrier: P4CaptureCohortBarrier,
    runtime_ids: Sequence[str],
) -> dict[str, Any]:
    gate = barrier.begin_step(
        _states(runtime_ids, PROMPT_COUNTS, [PREFILL_SAMPLE_TOKENS] * 3)
    )
    _require(gate["release"] is True, "complete cohort did not arm release")
    _require(gate["held_request_ids"] == (), "release retained a held request")
    return gate


def _run_positive_action(action_id: str, tag: int) -> dict[str, Any]:
    barrier, runtime_ids = _new_barrier(action_id, tag=tag)

    gate = barrier.begin_step(_states(runtime_ids, [0, 0, 0], [0, 0, 0]))
    _require(gate["held_request_ids"] == (), "initial cohort unexpectedly held")
    barrier.end_step(
        scheduled_request_ids=runtime_ids,
        pure_decode=False,
        action_id=None,
        target_query_widths={},
        committed_tokens={},
        prefill_progress_tokens=12,
    )

    gate = barrier.begin_step(_states(runtime_ids, [6, 4, 2], [1, 0, 0]))
    _require(
        gate["held_request_ids"] == (runtime_ids[0],),
        "first early prefill was not held",
    )
    barrier.end_step(
        scheduled_request_ids=runtime_ids[1:],
        pure_decode=False,
        action_id=None,
        target_query_widths={},
        committed_tokens={},
        prefill_progress_tokens=8,
    )

    gate = barrier.begin_step(_states(runtime_ids, [6, 10, 10], [1, 1, 0]))
    _require(
        gate["held_request_ids"] == runtime_ids[:2],
        "second early prefill was not held",
    )
    barrier.end_step(
        scheduled_request_ids=runtime_ids[2:],
        pure_decode=False,
        action_id=None,
        target_query_widths={},
        committed_tokens={},
        prefill_progress_tokens=4,
    )

    _open_release(barrier, runtime_ids)
    width = ACTION_WIDTHS[action_id]
    committed = 0
    first_release_widths: dict[str, int] | None = None
    while committed < CPU_PROOF_DECODE_TOKENS:
        event_commit = min(width, CPU_PROOF_DECODE_TOKENS - committed)
        widths = dict.fromkeys(runtime_ids, width)
        if first_release_widths is None:
            first_release_widths = widths
        barrier.end_step(
            scheduled_request_ids=runtime_ids,
            pure_decode=True,
            action_id=action_id,
            target_query_widths=widths,
            committed_tokens=dict.fromkeys(runtime_ids, event_commit),
        )
        committed += event_commit
        if committed < CPU_PROOF_DECODE_TOKENS:
            barrier.begin_step(
                _states(
                    runtime_ids,
                    [prompt + committed for prompt in PROMPT_COUNTS],
                    [PREFILL_SAMPLE_TOKENS + committed] * len(runtime_ids),
                )
            )

    for request_id in runtime_ids:
        barrier.finish_request(
            request_id,
            PREFILL_SAMPLE_TOKENS + CPU_PROOF_DECODE_TOKENS,
        )
    barrier.close()
    summary = barrier.summary()
    _require(summary["state"] == "complete", "positive cohort did not complete")
    _require(
        set(summary["committed_by_request"].values()) == {CPU_PROOF_DECODE_TOKENS},
        "positive cohort measured work drifted",
    )
    return {
        "case_id": f"{action_id}_early_hold_release_rollover",
        "status": "pass",
        "action_id": action_id,
        "target_query_width": width,
        "prefill_steps": summary["prefill_step_count"],
        "decode_steps": summary["decode_step_count"],
        "request_count": summary["expected_request_count"],
        "unmeasured_prefill_tokens_per_request": PREFILL_SAMPLE_TOKENS,
        "measured_decode_tokens_per_request": CPU_PROOF_DECODE_TOKENS,
        "frontend_output_tokens_per_request": (
            PREFILL_SAMPLE_TOKENS + CPU_PROOF_DECODE_TOKENS
        ),
        "first_release_scheduled_all_members": True,
        "first_release_pure_decode": True,
        "first_release_query_widths": sorted(set(first_release_widths.values())),
    }


def _expect_rejection(
    case_id: str,
    failure_class: str,
    operation: Callable[[], Any],
    match: str,
) -> dict[str, str]:
    try:
        operation()
    except KOffRuntimeError as exc:
        _require(match in str(exc), f"{case_id} failed for another reason: {exc}")
        return {
            "case_id": case_id,
            "status": "pass_rejected",
            "failure_class": failure_class,
        }
    raise CohortBarrierProofError(f"{case_id} did not fail closed")


def _negative_cases() -> list[dict[str, str]]:
    cases: list[dict[str, str]] = []

    barrier = P4CaptureCohortBarrier(
        "missing-member",
        FROZEN_IDS,
        action_id=OFF_ACTION_ID,
        measured_decode_tokens=CPU_PROOF_DECODE_TOKENS,
    )
    for request_id in _runtime_ids()[:2]:
        barrier.register(request_id)
    cases.append(
        _expect_rejection(
            "seal_missing_member",
            "membership",
            barrier.seal,
            "before every frozen request",
        )
    )

    barrier = P4CaptureCohortBarrier(
        "unknown-member",
        FROZEN_IDS,
        action_id=OFF_ACTION_ID,
        measured_decode_tokens=CPU_PROOF_DECODE_TOKENS,
    )
    cases.append(
        _expect_rejection(
            "reject_unknown_identity",
            "identity",
            lambda: barrier.register("foreign-00000000"),
            "neither frozen nor",
        )
    )

    barrier, runtime_ids = _new_barrier()
    barrier.begin_step(_states(runtime_ids, [6, 4, 2], [1, 0, 0]))
    cases.append(
        _expect_rejection(
            "reject_early_decode_release",
            "early_release",
            lambda: barrier.end_step(
                scheduled_request_ids=runtime_ids,
                pure_decode=False,
                action_id=None,
                target_query_widths={},
                committed_tokens={},
                prefill_progress_tokens=8,
            ),
            "before cohort release",
        )
    )

    barrier, runtime_ids = _new_barrier()
    cases.append(
        _expect_rejection(
            "reject_second_prefill_sample",
            "prefill_offset",
            lambda: barrier.begin_step(_states(runtime_ids, [6, 4, 2], [2, 0, 0])),
            "pre-release output tokens",
        )
    )

    barrier, runtime_ids = _new_barrier()
    _open_release(barrier, runtime_ids)
    cases.append(
        _expect_rejection(
            "reject_partial_first_release",
            "atomic_release",
            lambda: barrier.end_step(
                scheduled_request_ids=runtime_ids[:2],
                pure_decode=True,
                action_id=OFF_ACTION_ID,
                target_query_widths=dict.fromkeys(runtime_ids[:2], 1),
                committed_tokens=dict.fromkeys(runtime_ids[:2], 1),
            ),
            "every unfinished member",
        )
    )

    barrier, runtime_ids = _new_barrier()
    _open_release(barrier, runtime_ids)
    cases.append(
        _expect_rejection(
            "reject_mixed_first_release",
            "pure_decode",
            lambda: barrier.end_step(
                scheduled_request_ids=runtime_ids,
                pure_decode=False,
                action_id=OFF_ACTION_ID,
                target_query_widths=dict.fromkeys(runtime_ids, 1),
                committed_tokens=dict.fromkeys(runtime_ids, 1),
            ),
            "mixed prefill/decode",
        )
    )

    barrier, runtime_ids = _new_barrier()
    _open_release(barrier, runtime_ids)
    cases.append(
        _expect_rejection(
            "reject_quality_corruption",
            "quality",
            lambda: barrier.end_step(
                scheduled_request_ids=runtime_ids,
                pure_decode=True,
                action_id=OFF_ACTION_ID,
                target_query_widths=dict.fromkeys(runtime_ids, 1),
                committed_tokens=dict.fromkeys(runtime_ids, 1),
                recomputed_tokens=1,
            ),
            "preemption, recomputation",
        )
    )

    barrier, runtime_ids = _new_barrier(K4_ACTION_ID, measured_decode_tokens=3)
    _open_release(barrier, runtime_ids)
    cases.append(
        _expect_rejection(
            "reject_decode_overcommit",
            "accounting",
            lambda: barrier.end_step(
                scheduled_request_ids=runtime_ids,
                pure_decode=True,
                action_id=K4_ACTION_ID,
                target_query_widths=dict.fromkeys(runtime_ids, 5),
                committed_tokens=dict.fromkeys(runtime_ids, 4),
            ),
            "exceeded measured decode work",
        )
    )

    barrier, runtime_ids = _new_barrier()
    barrier.begin_step(_states(runtime_ids, [6, 4, 2], [1, 0, 0]))
    barrier.end_step(
        scheduled_request_ids=(),
        pure_decode=False,
        action_id=None,
        target_query_widths={},
        committed_tokens={},
    )
    barrier.begin_step(_states(runtime_ids, [6, 4, 2], [1, 0, 0]))
    cases.append(
        _expect_rejection(
            "abort_no_progress_deadlock",
            "deadlock",
            lambda: barrier.end_step(
                scheduled_request_ids=(),
                pure_decode=False,
                action_id=None,
                target_query_widths={},
                committed_tokens={},
            ),
            "would deadlock",
        )
    )

    barrier, runtime_ids = _new_barrier()
    _open_release(barrier, runtime_ids)
    barrier.end_step(
        scheduled_request_ids=runtime_ids,
        pure_decode=True,
        action_id=OFF_ACTION_ID,
        target_query_widths=dict.fromkeys(runtime_ids, 1),
        committed_tokens=dict.fromkeys(runtime_ids, 1),
    )
    cases.append(
        _expect_rejection(
            "reject_premature_frontend_finish",
            "frontend_work",
            lambda: barrier.finish_request(runtime_ids[0], 2),
            "before exact measured decode work",
        )
    )
    return cases


def _abort_case() -> dict[str, Any]:
    barrier, runtime_ids = _new_barrier()
    abort_ids = barrier.abort_member(runtime_ids[1])
    _require(barrier.state == "aborted", "member abort did not terminate cohort")
    _require(abort_ids == runtime_ids, "member abort did not return the full cohort")
    return {
        "case_id": "member_abort_cancels_complete_cohort",
        "status": "pass",
        "abort_request_count": len(abort_ids),
        "partial_capture_allowed": False,
        "score_allowed": False,
    }


def run_cpu_proof() -> dict[str, Any]:
    """Run deterministic lifecycle, rollover, and fail-closed proof cases."""
    positive = [
        _run_positive_action(action_id, tag)
        for tag, action_id in enumerate(ACTION_WIDTHS)
    ]
    negative = _negative_cases()
    abort = _abort_case()
    return {
        "status": "pass",
        "gpu_executed": False,
        "positive_cases": positive,
        "rejection_cases": negative,
        "abort_cases": [abort],
        "positive_case_count": len(positive),
        "rejection_case_count": len(negative),
        "abort_case_count": 1,
        "action_rollover_order": list(ACTION_WIDTHS),
        "state_leak_across_cohorts": False,
        "all_cases_passed": True,
    }


def _validate_sources(artifact: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    references = artifact["source_artifacts"]
    _require(
        set(references) == set(EXPECTED_SOURCE_PATHS),
        "cohort-barrier source closure drifted",
    )
    loaded: dict[str, dict[str, Any]] = {}
    for role, relative_path in EXPECTED_SOURCE_PATHS.items():
        reference = references[role]
        _require(reference.get("path") == relative_path, f"source path drifted: {role}")
        path = _repository_path(relative_path)
        _require(path.is_file(), f"source artifact is missing: {path}")
        _require(
            reference.get("sha256") == _sha256(path),
            f"source hash drifted: {role}",
        )
        if path.suffix == ".json":
            loaded[role] = _load_json(path)
    return loaded


def _validate_upstream(sources: Mapping[str, Mapping[str, Any]]) -> None:
    transient = sources["transient_bound"]
    _require(
        transient.get("status") == "reject_current_full_prefill_geometry"
        and transient.get("next_artifact", {}).get("kind")
        == "p4_b0_chunked_prefill_cohort_barrier_design_and_cpu_proof",
        "transient-bound handoff drifted",
    )
    authorization = sources["v9_authorization"]
    _require(
        authorization.get("package_id") == "p4-b0-value-screen-run-authorization-v9",
        "source authorization is not consumed V9",
    )
    failure = sources["v9_failure"]
    _require(
        failure.get("disposition", {}).get("v9_consumed") is True
        and failure.get("attempt", {}).get("complete_captures_emitted") == 8
        and failure.get("attempt", {}).get("score_emitted") is False,
        "V9 failure disposition drifted",
    )
    serving = sources["serving_chunked_prefill_diagnosis"]
    _require(
        serving.get("status") == "pass"
        and serving.get("engine", {}).get("max_num_batched_tokens")
        == MAX_NUM_BATCHED_TOKENS
        and serving["engine"].get("effective_scheduler_token_budget")
        == EFFECTIVE_SCHEDULER_TOKEN_BUDGET
        and serving.get("invariants", {}).get("mixed_steps_force_q1_off") is True
        and serving["invariants"].get("mixed_steps_dispatch_no_draft") is True,
        "ordinary-serving chunked-prefill evidence drifted",
    )


def _max_microbatch_tokens(manifest: Mapping[str, Any], regime_id: str) -> int:
    regimes = {row["regime_id"]: row for row in manifest["prompt_plan"]["regimes"]}
    batch = regimes[regime_id]["batch"]
    maxima: list[int] = []
    for seed in manifest["prompt_plan"]["content_seeds"]:
        rows = [
            row
            for row in manifest["prompts"]
            if row["regime_id"] == regime_id and row["content_seed"] == seed
        ]
        rows.sort(key=lambda row: row["prompt_index"])
        for start in range(0, len(rows), batch):
            maxima.append(
                sum(row["token_count"] for row in rows[start : start + batch])
            )
    return max(maxima)


def _validate_design(
    artifact: Mapping[str, Any], sources: Mapping[str, Mapping[str, Any]]
) -> None:
    manifest = sources["prompt_manifest"]
    r5_tokens = _max_microbatch_tokens(manifest, "R5")
    r5cot_tokens = _max_microbatch_tokens(manifest, "R5cot")
    expected_bounded = {
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "max_num_seqs": MAX_NUM_SEQS,
        "speculative_slot_reserve_tokens": SPECULATIVE_SLOT_RESERVE,
        "effective_scheduler_token_budget": EFFECTIVE_SCHEDULER_TOKEN_BUDGET,
        "full_microbatch_prefill_allowed": False,
        "r5_max_microbatch_prompt_tokens": r5_tokens,
        "r5cot_max_microbatch_prompt_tokens": r5cot_tokens,
        "r5_minimum_prefill_events": math.ceil(
            r5_tokens / EFFECTIVE_SCHEDULER_TOKEN_BUDGET
        ),
        "r5cot_minimum_prefill_events": math.ceil(
            r5cot_tokens / EFFECTIVE_SCHEDULER_TOKEN_BUDGET
        ),
        "minimum_shared_target_kv_blocks": MINIMUM_SHARED_KV_BLOCKS,
        "gpu_memory_utilization": "deferred_to_separate_gpu_probe_authorization",
    }
    _require(
        artifact["bounded_prefill_contract"] == expected_bounded,
        "bounded-prefill contract drifted",
    )
    _require(
        artifact["cohort_membership_contract"]
        == {
            "source": "active_capture_cell_prompt_slice",
            "queue_policy": "all_exact_members_before_first_engine_step",
            "identity_policy": (
                "frozen_id_or_frozen_id_plus_exactly_eight_lowercase_hex"
            ),
            "order_policy": "exact_frozen_queue_order",
            "foreign_requests_allowed": False,
            "duplicate_requests_allowed": False,
            "seal_before_complete_registration_allowed": False,
        },
        "cohort membership contract drifted",
    )
    _require(
        artifact["state_machine"]
        == {
            "states": [
                "registering",
                "prefilling",
                "released",
                "complete",
                "aborted",
            ],
            "hold_predicate": (
                "num_computed_tokens==num_prompt_tokens_and_num_output_tokens==1"
            ),
            "hold_effect": "skip_decode_scheduling_while_retaining_target_owned_kv",
            "release_predicate": "every_exact_member_satisfies_hold_predicate",
            "release_boundary": "next_scheduler_step_only",
            "member_finish_before_release_allowed": False,
            "request_refill_or_reprefill_allowed": False,
            "private_draft_kv_allowed": False,
            "terminal_states": ["complete", "aborted"],
        },
        "cohort state-machine contract drifted",
    )
    _require(
        artifact["decode_release_contract"]
        == {
            "first_measured_event": "pure_decode",
            "scheduled_members": "every_unfinished_cohort_member",
            "action_query_widths": ACTION_WIDTHS,
            "mixed_prefill_decode_allowed": False,
            "preemptions": 0,
            "recomputed_tokens": 0,
            "invalid_spec_tokens": 0,
            "steady_state_action_required": True,
        },
        "decode release contract drifted",
    )
    _require(
        artifact["accounting_contract"]
        == {
            "measurement_currency": "S_dec",
            "unmeasured_prefill_tokens_per_request": PREFILL_SAMPLE_TOKENS,
            "measured_decode_tokens_source": (
                "capture_cell_generation.max_output_tokens"
            ),
            "cpu_proof_measured_decode_tokens_per_request": CPU_PROOF_DECODE_TOKENS,
            "cpu_proof_frontend_output_tokens_per_request": (
                PREFILL_SAMPLE_TOKENS + CPU_PROOF_DECODE_TOKENS
            ),
            "prefill_events_contribute_measured_commits": False,
            "overcommit_allowed": False,
            "incomplete_work_score_allowed": False,
        },
        "cohort accounting contract drifted",
    )
    _require(
        artifact["failure_policy"]
        == {
            "on_member_abort": "abort_every_unfinished_cohort_member",
            "on_identity_or_order_drift": "abort_without_capture_or_score",
            "on_early_decode": "abort_without_capture_or_score",
            "on_mixed_first_release": "abort_without_capture_or_score",
            "on_quality_violation": "abort_without_capture_or_score",
            "on_two_no_progress_steps": "abort_without_capture_or_score",
            "on_incomplete_close": "abort_without_capture_or_score",
            "partial_resume_allowed": False,
        },
        "cohort failure policy drifted",
    )
    _require(
        artifact["ordinary_serving_boundary"]
        == {
            "cohort_barrier_enabled": False,
            "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
            "effective_scheduler_token_budget": EFFECTIVE_SCHEDULER_TOKEN_BUDGET,
            "gpu_memory_utilization": 0.9,
            "mixed_steps_force_q1_off": True,
            "mixed_steps_dispatch_draft": False,
            "serving_behavior_changed_by_this_artifact": False,
        },
        "ordinary-serving boundary drifted",
    )


def validate_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Validate source closure, design, CPU proof, and authority boundaries.

    Args:
        artifact: Parsed cohort-barrier design and proof.

    Returns:
        Stable validation summary.

    Raises:
        CohortBarrierProofError: If any source or invariant differs.
    """
    _require(
        set(artifact) == EXPECTED_TOP_LEVEL_KEYS,
        "cohort-barrier top-level fields drifted",
    )
    _require(
        artifact.get("schema_version") == 1
        and artifact.get("artifact_id")
        == "p4-b0-chunked-prefill-cohort-barrier-design-and-cpu-proof-v1"
        and artifact.get("status") == "pass_cpu_proof_gpu_unauthorized"
        and artifact.get("evidence_grade")
        == "executable_cpu_state_machine_proof_no_gpu_execution",
        "cohort-barrier artifact identity drifted",
    )
    sources = _validate_sources(artifact)
    _validate_upstream(sources)
    _validate_design(artifact, sources)
    _require(
        artifact["rejected_mechanism"]
        == {
            "mechanism": "114688_0.96_full_microbatch_prefill",
            "disposition": "remains_rejected_no_retry_or_resume",
            "v9_output": (
                "research/97_composition_runtime/data/p4/run_b0_value_screen_v8"
            ),
            "v9_consumed": True,
        },
        "rejected full-prefill disposition drifted",
    )
    observed_proof = run_cpu_proof()
    _require(artifact["cpu_proof"] == observed_proof, "checked CPU proof drifted")
    _require(
        artifact["decision"]
        == {
            "state": "pass",
            "scope": "cpu_only_cohort_barrier_design",
            "live_engine_wiring_ready_for_review": True,
            "gpu_execution_authorized": False,
            "v10_authorized": False,
        },
        "CPU-only decision boundary drifted",
    )
    _require(
        artifact["claims"]
        == {
            "bounded_prefill_design_complete": True,
            "cohort_barrier_cpu_proven": True,
            "ordinary_serving_unchanged": True,
            "live_engine_wired": False,
            "gpu_resource_fit_proven": False,
            "value_screen_run_ready": False,
            "p4a_ready": False,
            "performance_claim_allowed": False,
        },
        "cohort-barrier claims drifted",
    )
    _require(
        not any(artifact["authorizations"].values()),
        "CPU proof must not grant downstream authority",
    )
    _require(
        artifact["next_artifact"]
        == {
            "kind": (
                "p4_b0_chunked_prefill_cohort_barrier_live_wiring_and_"
                "gpu_probe_authorization"
            ),
            "probe_scope": "one_non_scored_boot_gpu4",
            "probe_regimes": ["R4", "R5", "R5cot"],
            "must_measure_actual_graph_memory": True,
            "minimum_shared_target_kv_blocks": MINIMUM_SHARED_KV_BLOCKS,
            "must_prove_no_preemption_or_recomputation": True,
            "must_prove_pure_first_measured_decode": True,
            "separate_source_bound_authorization_required": True,
            "separate_v10_authorization_required": True,
        },
        "cohort-barrier next-artifact boundary drifted",
    )
    return {
        "status": "pass",
        "artifact_id": artifact["artifact_id"],
        "bounded_prefill_budget": MAX_NUM_BATCHED_TOKENS,
        "effective_scheduler_token_budget": EFFECTIVE_SCHEDULER_TOKEN_BUDGET,
        "r5_minimum_prefill_events": artifact["bounded_prefill_contract"][
            "r5_minimum_prefill_events"
        ],
        "r5cot_minimum_prefill_events": artifact["bounded_prefill_contract"][
            "r5cot_minimum_prefill_events"
        ],
        "cpu_positive_cases": observed_proof["positive_case_count"],
        "cpu_rejection_cases": observed_proof["rejection_case_count"],
        "cpu_abort_cases": observed_proof["abort_case_count"],
        "gpu_executed": False,
        "gpu_probe_authorized": False,
        "v10_authorized": False,
    }


def parse_args() -> argparse.Namespace:
    """Parse the validator command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Validate the checked artifact and print a stable JSON summary."""
    args = parse_args()
    try:
        result = validate_artifact(_load_json(args.artifact))
    except CohortBarrierProofError as exc:
        raise SystemExit(f"P4 cohort-barrier proof rejected: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
