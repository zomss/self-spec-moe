#!/usr/bin/env python3
"""Verify the Phase 97 live K4/OFF smoke against its greedy AR oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

K4_ACTION = "target-matching-k4"
OFF_ACTION = "off"


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


def _token_ids(result: dict[str, Any]) -> dict[str, list[int]]:
    return {
        request_id: request["token_ids"]
        for request_id, request in result["requests"].items()
    }


def _token_mismatches(
    spec_tokens: dict[str, list[int]], ar_tokens: dict[str, list[int]]
) -> dict[str, dict[str, int | None]]:
    mismatches = {}
    for request_id in sorted(spec_tokens.keys() | ar_tokens.keys()):
        spec_row = spec_tokens.get(request_id, [])
        ar_row = ar_tokens.get(request_id, [])
        first_index = next(
            (
                index
                for index, (spec_token, ar_token) in enumerate(
                    zip(spec_row, ar_row, strict=False)
                )
                if spec_token != ar_token
            ),
            None,
        )
        if first_index is None and len(spec_row) != len(ar_row):
            first_index = min(len(spec_row), len(ar_row))
        if first_index is not None:
            mismatches[request_id] = {
                "first_index": first_index,
                "spec_token_id": (
                    spec_row[first_index] if first_index < len(spec_row) else None
                ),
                "ar_token_id": (
                    ar_row[first_index] if first_index < len(ar_row) else None
                ),
            }
    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--ar", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    spec = _load_json(args.spec)
    ar = _load_json(args.ar)
    trace = _load_jsonl(args.trace)
    _require(spec["mode"] == "spec", "spec output has the wrong mode")
    _require(ar["mode"] == "ar", "AR output has the wrong mode")
    spec_tokens = _token_ids(spec)
    ar_tokens = _token_ids(ar)
    token_mismatches = _token_mismatches(spec_tokens, ar_tokens)
    token_identity = not token_mismatches

    headers = [r for r in trace if r.get("record_type") == "koff_runtime_header"]
    steps = [r for r in trace if r.get("record_type") == "koff_engine_step"]
    _require(len(headers) == 1, f"expected one trace header, got {len(headers)}")
    _require(bool(steps), "live trace contains no engine steps")
    header = headers[0]
    _require(header["scored"] is False, "live smoke must remain non-scored")
    _require(
        header["contract"] == "phase97-minimal-b0-k4-off",
        "wrong runtime contract",
    )
    environment = header["environment"]
    _require(
        environment["target_model"] == environment["draft_model"],
        "target and draft checkpoints differ",
    )

    pairs = [(step.get("verified_action_id"), step["next_action_id"]) for step in steps]
    k4_to_off = [
        index for index, pair in enumerate(pairs) if pair == (K4_ACTION, OFF_ACTION)
    ]
    off_to_k4 = [
        index for index, pair in enumerate(pairs) if pair == (OFF_ACTION, K4_ACTION)
    ]
    _require(bool(k4_to_off), "trace has no K4-to-OFF transition")
    _require(bool(off_to_k4), "trace has no OFF-to-K4 transition")
    k4_to_off_index = k4_to_off[0]
    off_to_k4_index = next(
        (index for index in off_to_k4 if index > k4_to_off_index),
        None,
    )
    _require(off_to_k4_index is not None, "trace order is not K4-to-OFF-to-K4")
    _require(
        any(pair == (K4_ACTION, K4_ACTION) for pair in pairs[:k4_to_off_index]),
        "K4 was not stable before switching OFF",
    )
    _require(
        any(pair[0] == K4_ACTION for pair in pairs[(off_to_k4_index or 0) + 1 :]),
        "K4 was not verified after re-entry",
    )
    _require(
        all(step["target_step_boundary"] for step in steps),
        "an action was recorded away from a target-step boundary",
    )

    bootstrap = steps[off_to_k4_index]
    execution = bootstrap["execution"]
    _require(
        bootstrap["eligible_for_p3_replay"],
        "OFF-to-K4 bootstrap is not a pure eligible decode step",
    )
    _require(execution["target_query_width"] == 1, "bootstrap target width is not 1")
    _require(
        execution["draft_step0_query_width"] == 1,
        "bootstrap draft step-0 width is not 1",
    )
    _require(execution["produced_draft_width"] == 4, "bootstrap did not produce K4")
    _require(execution["draft_dispatched"], "bootstrap did not dispatch the draft")

    binding_ids = {step["resources"]["binding_id"] for step in steps}
    pool_ids = {step["resources"]["pool_id"] for step in steps}
    slot_ids = {step["resources"]["true_slot_mapping_id"] for step in steps}
    _require(len(binding_ids) == 1, "live shared-KV binding changed")
    _require(len(pool_ids) == 1, "live target KV pool changed")
    _require(len(slot_ids) == 1, "canonical target slot buffer changed")
    for step in steps:
        evidence = step["execution"]
        _require(evidence["shared_kv_layer_count"] > 0, "no shared KV layers")
        _require(
            evidence["shared_kv_layer_count"]
            == evidence["shared_kv_storage_alias_count"],
            "not every draft KV layer aliases target storage",
        )
        _require(
            evidence["shared_weight_parameter_count"] > 0,
            "no live shared-weight aliases",
        )
        _require(
            evidence["target_weight_version_id"] == evidence["draft_weight_version_id"],
            "target/draft weight identity changed",
        )
        _require(step["counters"]["closure_holds"], "H/D/A/C/E closure failed")

    eligible = [step for step in steps if step["eligible_for_p3_replay"]]
    _require(bool(eligible), "trace has no replay-eligible records")
    _require(
        any(step["verified_action_id"] == OFF_ACTION for step in eligible),
        "trace has no eligible OFF record",
    )
    _require(
        any(step["verified_action_id"] == K4_ACTION for step in eligible),
        "trace has no eligible K4 record",
    )
    aggregate = {
        name: sum(step["counters"][name] for step in eligible)
        for name in (
            "H_target_steps",
            "D_armed",
            "A_accepted",
            "C_clipped",
            "E_committed",
        )
    }
    _require(
        aggregate["E_committed"] + aggregate["C_clipped"]
        == aggregate["A_accepted"] + aggregate["H_target_steps"],
        "aggregate H/D/A/C/E closure failed",
    )

    report = {
        "schema_version": 1,
        "record_type": "p3_live_smoke_verification",
        "scored": False,
        "status": "pass" if token_identity else "fail",
        "checks": {
            "greedy_token_identity": token_identity,
            "ordered_k4_off_k4_transition": True,
            "off_to_k4_q1_bootstrap": True,
            "target_step_boundary_only": True,
            "stable_live_kv_aliases": True,
            "stable_live_weight_aliases": True,
            "stable_target_slot_mapping": True,
            "no_active_private_draft_kv_pool": True,
            "h_d_a_c_e_closure": True,
        },
        "token_counts": {
            request_id: len(tokens) for request_id, tokens in spec_tokens.items()
        },
        "token_mismatches": token_mismatches,
        "trace": {
            "engine_step_records": len(steps),
            "eligible_records": len(eligible),
            "k4_to_off_engine_step_index": steps[k4_to_off_index]["engine_step_index"],
            "off_to_k4_engine_step_index": bootstrap["engine_step_index"],
            "bootstrap_target_runtime_mode": execution["target_runtime_mode"],
            "bootstrap_draft_step0_runtime_mode": execution["draft_step0_runtime_mode"],
            "bootstrap_draft_chain_runtime_mode": execution["draft_chain_runtime_mode"],
            "binding_id": next(iter(binding_ids)),
            "pool_id": next(iter(pool_ids)),
            "true_slot_mapping_id": next(iter(slot_ids)),
            "shared_kv_layer_count": execution["shared_kv_layer_count"],
            "shared_weight_parameter_count": execution["shared_weight_parameter_count"],
        },
        "eligible_counter_aggregate": aggregate,
    }
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))
    if not token_identity:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
