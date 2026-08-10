#!/usr/bin/env python3
"""Verify the Phase 97 P3b mixed-boundary K4 abort repair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

K4_ACTION = "target-matching-k4"
OFF_ACTION = "off"
LONG_REQUEST_ID = "long"
SHORT_REQUEST_ID = "short"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _steps(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    headers = [r for r in records if r.get("record_type") == "koff_runtime_header"]
    steps = [r for r in records if r.get("record_type") == "koff_engine_step"]
    _require(len(headers) == 1, f"expected one trace header, got {len(headers)}")
    _require(bool(steps), "trace contains no engine steps")
    _require(
        headers[0].get("contract") == "phase97-minimal-b0-k4-off",
        "trace has the wrong runtime contract",
    )
    return steps


def _diagnostic(step: dict[str, Any]) -> dict[str, Any]:
    diagnostic = step.get("execution", {}).get("diagnostic")
    _require(isinstance(diagnostic, dict), "step is missing target diagnostic data")
    return diagnostic


def _first_mixed_off_step(steps: list[dict[str, Any]]) -> dict[str, Any]:
    for step in steps:
        diagnostic = step.get("execution", {}).get("diagnostic")
        if (
            step.get("verified_action_id") == OFF_ACTION
            and step.get("next_action_id") == OFF_ACTION
            and isinstance(diagnostic, dict)
            and diagnostic.get("num_scheduled_tokens") == [1, 10]
        ):
            return step
    raise RuntimeError("trace has no mixed q=[1,10] OFF step")


def _stable_k4_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        step
        for step in steps
        if step.get("eligible_for_p3_replay")
        and step.get("verified_action_id") == K4_ACTION
        and step.get("next_action_id") == K4_ACTION
    ]


def _accepted_prefix_length(draft: list[int], target: list[int]) -> int:
    accepted = 0
    for draft_token, target_token in zip(draft, target, strict=True):
        if draft_token != target_token:
            break
        accepted += 1
    return accepted


def _check_target_local_k4(step: dict[str, Any]) -> None:
    diagnostic = _diagnostic(step)
    _require(
        diagnostic["num_scheduled_tokens"] == [5],
        "target-local K4 check requires one q=5 decode row",
    )
    _require(diagnostic["num_draft_tokens"] == [4], "K4 draft width is not 4")
    logits_indices = diagnostic["logits_indices"]
    argmax_ids = diagnostic["logit_argmax_token_ids"]
    argmax_by_row = dict(zip(logits_indices, argmax_ids, strict=True))
    target_ids = [argmax_by_row[index] for index in diagnostic["target_logits_indices"]]
    draft_ids = diagnostic["draft_token_ids"]
    _require(len(draft_ids) == 4, "K4 diagnostic has the wrong draft width")
    expected_accepted = _accepted_prefix_length(draft_ids, target_ids)
    counters = step["counters"]
    _require(counters["H_target_steps"] == 1, "K4 step must contain one target row")
    _require(counters["D_armed"] == 1, "K4 step must count one armed row")
    _require(
        counters["A_accepted"] == expected_accepted,
        "accepted K4 count does not match the target-verifier argmax prefix",
    )


def _check_aliases_and_closure(steps: list[dict[str, Any]]) -> dict[str, Any]:
    binding_ids = {step["resources"]["binding_id"] for step in steps}
    pool_ids = {step["resources"]["pool_id"] for step in steps}
    slot_ids = {step["resources"]["true_slot_mapping_id"] for step in steps}
    _require(len(binding_ids) == 1, "live shared-KV binding changed")
    _require(len(pool_ids) == 1, "live target KV pool changed")
    _require(len(slot_ids) == 1, "canonical target slot buffer changed")
    for step in steps:
        evidence = step["execution"]
        _require(evidence["shared_kv_layer_count"] == 36, "expected 36 KV aliases")
        _require(
            evidence["shared_kv_storage_alias_count"] == 36,
            "not every draft KV layer aliases target storage",
        )
        _require(
            evidence["shared_weight_parameter_count"] == 291,
            "expected 291 shared target/draft parameters",
        )
        _require(
            evidence["target_weight_version_id"] == evidence["draft_weight_version_id"],
            "target/draft weight identity changed",
        )
        _require(step["counters"]["closure_holds"], "H/D/A/C/E closure failed")
    return {
        "binding_id": next(iter(binding_ids)),
        "pool_id": next(iter(pool_ids)),
        "true_slot_mapping_id": next(iter(slot_ids)),
    }


def _token_mismatches(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, dict[str, int | None]]:
    mismatches: dict[str, dict[str, int | None]] = {}
    request_ids = set(left["requests"]) | set(right["requests"])
    for request_id in sorted(request_ids):
        left_ids = left.get("requests", {}).get(request_id, {}).get("token_ids", [])
        right_ids = right.get("requests", {}).get(request_id, {}).get("token_ids", [])
        first_index = next(
            (
                index
                for index, (left_id, right_id) in enumerate(
                    zip(left_ids, right_ids, strict=False)
                )
                if left_id != right_id
            ),
            None,
        )
        if first_index is None and len(left_ids) != len(right_ids):
            first_index = min(len(left_ids), len(right_ids))
        if first_index is not None:
            mismatches[request_id] = {
                "first_index": first_index,
                "left_token_id": (
                    left_ids[first_index] if first_index < len(left_ids) else None
                ),
                "right_token_id": (
                    right_ids[first_index] if first_index < len(right_ids) else None
                ),
            }
    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynamic", type=Path, required=True)
    parser.add_argument("--dynamic-trace", type=Path, required=True)
    parser.add_argument("--off-control", type=Path, required=True)
    parser.add_argument("--off-trace", type=Path, required=True)
    parser.add_argument("--k4-control", type=Path, required=True)
    parser.add_argument("--k4-trace", type=Path, required=True)
    parser.add_argument("--ar-diagnostic", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")

    dynamic = _load_json(args.dynamic)
    off_control = _load_json(args.off_control)
    k4_control = _load_json(args.k4_control)
    dynamic_steps = _steps(_load_jsonl(args.dynamic_trace))
    off_steps = _steps(_load_jsonl(args.off_trace))
    k4_steps = _steps(_load_jsonl(args.k4_trace))

    abort_steps = [
        step for step in dynamic_steps if step.get("aborted_action_id") == K4_ACTION
    ]
    _require(len(abort_steps) == 1, f"expected one K4 abort, got {len(abort_steps)}")
    abort_step = abort_steps[0]
    abort_index = dynamic_steps.index(abort_step)
    _require(abort_step["verified_action_id"] == OFF_ACTION, "abort did not verify OFF")
    _require(abort_step["next_action_id"] == OFF_ACTION, "abort did not remain OFF")
    _require(abort_step["selection_intent"] == "force_off", "abort was not forced")
    _require(abort_step["discarded_draft_width"] == 4, "abort width is not K4")
    _require(abort_step["aborted_draft_request_count"] == 1, "wrong abort row count")
    _require(not abort_step["eligible_for_p3_replay"], "abort step was replay-eligible")
    _require(
        "aborted_mixed_boundary_draft" in abort_step["exclusion_reasons"],
        "abort exclusion reason is missing",
    )
    abort_counters = abort_step["counters"]
    _require(abort_counters["D_armed"] == 0, "aborted K4 counted as armed")
    _require(abort_counters["A_accepted"] == 0, "aborted K4 counted as accepted")
    abort_execution = abort_step["execution"]
    _require(abort_execution["target_query_width"] == 1, "abort target width is not 1")
    _require(abort_execution["produced_draft_width"] == 0, "abort produced drafts")
    _require(not abort_execution["draft_dispatched"], "abort dispatched a draft")
    abort_diagnostic = _diagnostic(abort_step)
    _require(
        abort_diagnostic["num_scheduled_tokens"] == [1, 10],
        "mixed abort geometry is not q=[1,10]",
    )
    _require(
        abort_diagnostic["num_draft_tokens"] == [0, 0],
        "mixed abort retained verifier draft rows",
    )
    _require(
        abort_diagnostic["query_start_loc"] == [0, 1, 11],
        "mixed abort query starts are wrong",
    )
    for slots in abort_diagnostic["slot_mappings_by_group"].values():
        _require(len(slots) == 11, "mixed abort slot row has the wrong length")
        _require(slots[0] not in slots[1:], "decode and prefill target slots overlap")

    dynamic_stable = _stable_k4_steps(dynamic_steps)
    _require(
        any(dynamic_steps.index(step) < abort_index for step in dynamic_stable),
        "K4 was not stable before the abort",
    )
    off_to_k4_index = next(
        (
            index
            for index, step in enumerate(dynamic_steps)
            if index > abort_index
            and step.get("verified_action_id") == OFF_ACTION
            and step.get("next_action_id") == K4_ACTION
            and step.get("eligible_for_p3_replay")
        ),
        None,
    )
    _require(off_to_k4_index is not None, "trace has no pure OFF-to-K4 re-entry")
    bootstrap = dynamic_steps[off_to_k4_index]
    bootstrap_execution = bootstrap["execution"]
    _require(
        bootstrap_execution["target_query_width"] == 1,
        "bootstrap target is not q1",
    )
    _require(
        bootstrap_execution["draft_step0_query_width"] == 1,
        "bootstrap draft step 0 is not q1",
    )
    _require(bootstrap_execution["produced_draft_width"] == 4, "bootstrap is not K4")
    _require(
        any(
            index > off_to_k4_index and step.get("verified_action_id") == K4_ACTION
            for index, step in enumerate(dynamic_steps)
        ),
        "K4 was not verified after re-entry",
    )

    for step in dynamic_steps + k4_steps:
        if step.get("verified_action_id") == K4_ACTION:
            _check_target_local_k4(step)

    off_mixed = _first_mixed_off_step(off_steps)
    _require(off_mixed.get("aborted_action_id") is None, "OFF control reports an abort")
    off_diagnostic = _diagnostic(off_mixed)
    matched_fields = (
        "num_scheduled_tokens",
        "num_draft_tokens",
        "query_start_loc",
        "input_token_ids",
        "positions",
        "seq_lens",
        "logits_indices",
    )
    for field in matched_fields:
        _require(
            abort_diagnostic[field] == off_diagnostic[field],
            f"mixed abort differs from the shape-matched OFF control in {field}",
        )
    _require(
        abort_execution["target_runtime_mode"]
        == off_mixed["execution"]["target_runtime_mode"],
        "mixed abort and OFF control used different target runtime modes",
    )
    for dynamic_ids, off_ids in zip(
        abort_diagnostic["logit_top_token_ids"],
        off_diagnostic["logit_top_token_ids"],
        strict=True,
    ):
        _require(
            set(dynamic_ids) == set(off_ids),
            "mixed abort and OFF control have different top candidate sets",
        )
    max_top_logit_delta = max(
        abs(dynamic_value - off_value)
        for dynamic_row, off_row in zip(
            abort_diagnostic["logit_top_values"],
            off_diagnostic["logit_top_values"],
            strict=True,
        )
        for dynamic_value, off_value in zip(dynamic_row, off_row, strict=True)
    )
    _require(
        max_top_logit_delta <= 0.125,
        "shape-matched OFF logits differ by more than one BF16 step",
    )

    fixed_k4_stable = _stable_k4_steps(k4_steps)
    _require(bool(fixed_k4_stable), "fixed-K4 control has no stable K4 step")
    dynamic_stable_diagnostic = _diagnostic(dynamic_stable[0])
    fixed_k4_diagnostic = _diagnostic(fixed_k4_stable[0])
    for field in (
        "num_scheduled_tokens",
        "num_draft_tokens",
        "input_token_ids",
        "positions",
        "seq_lens",
        "draft_token_ids",
        "target_logits_indices",
        "logit_argmax_token_ids",
    ):
        _require(
            dynamic_stable_diagnostic[field] == fixed_k4_diagnostic[field],
            f"stable K4 differs from its fixed control in {field}",
        )
    injection_tokens = dynamic["injection"]["after_long_output_tokens"]
    _require(
        off_mixed["engine"]["generated_suffix_min"]
        == abort_step["engine"]["generated_suffix_min"],
        "OFF control does not match the boundary output position",
    )
    _require(
        k4_control["injection"]["after_long_output_tokens"] == injection_tokens,
        "fixed-K4 control injection point differs",
    )
    dynamic_long = dynamic["requests"][LONG_REQUEST_ID]["token_ids"]
    off_long = off_control["requests"][LONG_REQUEST_ID]["token_ids"]
    k4_long = k4_control["requests"][LONG_REQUEST_ID]["token_ids"]
    _require(
        dynamic_long[:injection_tokens] == k4_long[:injection_tokens],
        "pre-boundary stable K4 prefix differs from the fixed-K4 control",
    )
    boundary_output_index = abort_step["engine"]["generated_suffix_min"]
    _require(
        dynamic_long[boundary_output_index]
        == abort_diagnostic["logit_argmax_token_ids"][0],
        "mixed-boundary output is not its target-forward argmax",
    )
    _require(
        off_long[boundary_output_index] == off_diagnostic["logit_argmax_token_ids"][0],
        "OFF-control output is not its target-forward argmax",
    )
    _require(
        dynamic["requests"][SHORT_REQUEST_ID]["token_ids"]
        == off_control["requests"][SHORT_REQUEST_ID]["token_ids"],
        "mixed prefill/decode short output differs from the OFF control",
    )

    identities = _check_aliases_and_closure(dynamic_steps)
    ar_mismatches = None
    if args.ar_diagnostic is not None:
        ar_mismatches = _token_mismatches(dynamic, _load_json(args.ar_diagnostic))

    report = {
        "schema_version": 1,
        "record_type": "p3b_mixed_abort_verification",
        "scored": False,
        "status": "pass",
        "checks": {
            "mixed_pending_k4_aborted": True,
            "mixed_decode_query_width_one": True,
            "discarded_draft_not_counted": True,
            "shape_matched_off_boundary_geometry": True,
            "shape_matched_off_top_candidate_set": True,
            "fixed_k4_steady_decode": True,
            "target_local_verifier_correctness": True,
            "off_to_k4_q1_bootstrap": True,
            "stable_live_kv_aliases": True,
            "stable_live_weight_aliases": True,
            "stable_target_slot_mapping": True,
            "h_d_a_c_e_closure": True,
            "cross_boot_ar_identity_diagnostic": not ar_mismatches,
        },
        "transition": {
            "abort_engine_step_index": abort_step["engine_step_index"],
            "discarded_draft_width": abort_step["discarded_draft_width"],
            "aborted_draft_request_count": abort_step["aborted_draft_request_count"],
            "off_to_k4_engine_step_index": bootstrap["engine_step_index"],
            "injection_after_long_output_tokens": injection_tokens,
            "boundary_output_index": boundary_output_index,
            "dynamic_boundary_token_id": dynamic_long[boundary_output_index],
            "off_control_boundary_token_id": off_long[boundary_output_index],
            "boundary_argmax_identity_diagnostic": (
                dynamic_long[boundary_output_index] == off_long[boundary_output_index]
            ),
            "max_shape_matched_top_logit_delta": max_top_logit_delta,
        },
        "identities": identities,
        "cross_boot_ar_mismatches": ar_mismatches,
    }
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
