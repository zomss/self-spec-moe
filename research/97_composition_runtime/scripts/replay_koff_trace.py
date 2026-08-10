#!/usr/bin/env python3
"""Validate and replay the Phase 97 minimal-B0 K/OFF trace."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from plan_boot_class import (
    plan_boot_classes,
    validate_candidate,
    validate_environment,
    validate_workload,
)
from validate_shared_kv import validate_runtime_snapshot

PHASE_DIR = Path(__file__).resolve().parents[1]
TRACE_SCHEMA = PHASE_DIR / "schemas" / "passive_koff_trace.schema.json"


class KOffReplayError(ValueError):
    """Raised when the minimal-B0 package or trace violates its contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise KOffReplayError(message)


def _format_json_path(parts: Sequence[Any]) -> str:
    path = "$"
    for part in parts:
        if isinstance(part, int):
            path += f"[{part}]"
        else:
            path += f".{part}"
    return path


def _validate_trace_schema(trace: Mapping[str, Any]) -> None:
    schema = json.loads(TRACE_SCHEMA.read_text())
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise KOffReplayError(f"invalid passive trace schema: {exc}") from exc
    errors = sorted(
        Draft202012Validator(schema).iter_errors(trace),
        key=lambda error: [str(part) for part in error.absolute_path],
    )
    if not errors:
        return
    first = errors[0]
    path = _format_json_path(list(first.absolute_path))
    raise KOffReplayError(f"passive trace schema rejected {path}: {first.message}")


def interval_tie_set(
    speed_intervals: Mapping[str, tuple[float, float]],
    epsilon: float,
) -> list[str]:
    """Return the interval-dominance epsilon tie-set, including OFF.

    Args:
        speed_intervals: Action id to lower/upper speedup bounds.
        epsilon: Selection tolerance.

    Returns:
        Sorted action ids whose optimistic bound is not dominated.
    """
    pool = dict(speed_intervals)
    pool["off"] = (1.0, 1.0)
    best_lower = max(lower for lower, _ in pool.values())
    threshold = (1.0 - epsilon) * best_lower
    return sorted(
        action_id for action_id, (_, upper) in pool.items() if upper >= threshold
    )


def validate_p3_package(
    environment: Mapping[str, Any],
    workload: Mapping[str, Any],
    candidate: Mapping[str, Any],
    boot: Mapping[str, Any],
    registry: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> None:
    """Bind P1 shared-KV and P2 resource evidence to minimal B0.

    Args:
        environment: Fixed environment manifest.
        workload: Resource workload envelope.
        candidate: Exact measured B0 resource candidate.
        boot: Strict shared-KV boot manifest.
        registry: OFF plus one target-matching K4 action.
        runtime: Synthetic shared-KV binding snapshot.

    Raises:
        KOffReplayError: If any cross-artifact binding fails.
    """
    try:
        validate_environment(environment)
        validate_workload(environment, workload)
        validate_candidate(candidate)
        plan = plan_boot_classes(environment, workload, [candidate])
        validate_runtime_snapshot(boot, registry, runtime)
    except Exception as exc:
        raise KOffReplayError(f"upstream P1/P2 validation failed: {exc}") from exc

    _require(
        plan["admitted_candidate_ids"] == [candidate["candidate_id"]],
        "P2 resource planner did not admit the bound candidate",
    )
    _require(candidate["capability_class"] == "B0", "P3 supports only B0")
    _require(
        candidate["evidence"]["grade"] == "measured_exact",
        "P3 requires exact post-capture resource evidence",
    )
    _require(boot["capability_class"] == "B0", "P3 boot must be B0")
    _require(boot["max_k"] == 4, "minimal B0 must declare max_k=4")
    _require(
        boot["fixed_environment_id"] == environment["environment_id"],
        "boot fixed-environment id mismatch",
    )

    candidate_paths = {
        (path["path_id"], path["kind"])
        for path in candidate["resident_objects"]["draft_weight_paths"]
    }
    boot_paths = {
        (path["path_id"], path["kind"]) for path in boot["draft_weight_paths"]
    }
    _require(
        boot_paths == candidate_paths,
        "boot weight paths differ from exact resource evidence",
    )
    _require(
        set(boot["resident_graph_ids"])
        == set(candidate["resident_objects"]["graph_ids"]),
        "boot graph pool differs from exact resource evidence",
    )
    _require(
        boot["admitted_windows"] == [0],
        "minimal B0 cannot admit a window action",
    )
    _require(
        boot["admitted_skip_sets"] == [[]],
        "minimal B0 cannot admit a layer-skip action",
    )

    environment_kv = environment["kv"]
    boot_spec = boot["kv"]["cache_spec"]
    _require(boot_spec["dtype"] == environment_kv["dtype"], "KV dtype mismatch")
    _require(
        boot_spec["block_size"] == environment_kv["block_size_tokens"],
        "KV block-size mismatch",
    )
    _require(boot_spec["layout"] == environment_kv["layout"], "KV layout mismatch")
    _require(
        boot_spec["spec_fingerprint"] == environment_kv["spec_fingerprint"],
        "KV spec fingerprint mismatch",
    )

    resources = boot["resources"]
    capacity = candidate["evidence"]["capacity"]
    for boot_name, candidate_name in (
        ("usable_hbm_bytes", "usable_hbm_bytes"),
        ("shared_target_kv_blocks", "available_shared_target_kv_blocks"),
        ("kv_block_size_tokens", "kv_block_size_tokens"),
    ):
        _require(
            resources[boot_name] == capacity[candidate_name],
            f"boot resource {boot_name} differs from exact evidence",
        )
    fixed_terms = (
        "safety_bytes",
        "target_fixed_bytes",
        "target_matching_draft_extra_bytes",
        "quantized_draft_weights_bytes",
        "weight_refresh_peak_bytes",
        "graphs_and_workspaces_bytes",
    )
    accounted_non_kv = sum(int(resources[name]) for name in fixed_terms)
    measured_non_kv = (
        resources["usable_hbm_bytes"]
        - resources["shared_target_kv_blocks"] * capacity["kv_bytes_per_block"]
    )
    _require(
        accounted_non_kv == measured_non_kv,
        "strict boot HBM terms do not close against exact KV capacity",
    )

    actions = list(registry["actions"])
    action_ids = {action["action_id"] for action in actions}
    _require(
        action_ids == {"off", "target-matching-k4"},
        "minimal B0 registry must be exactly OFF plus target-matching K4",
    )
    off_actions = [action for action in actions if action["kind"] == "off"]
    spec_actions = [action for action in actions if action["kind"] == "speculative"]
    _require(len(off_actions) == 1, "minimal B0 requires exactly one OFF action")
    _require(
        len(spec_actions) == 1,
        "minimal B0 requires exactly one speculative action",
    )
    off = off_actions[0]
    spec = spec_actions[0]
    _require(
        off["target_graph_descriptor"]["query_width"] == 1,
        "OFF must use one-token target decode",
    )
    _require(
        off["target_graph_descriptor"]["graph_id"] == "target-k1",
        "OFF must use the exact target-k1 graph",
    )
    _require(
        spec["draft_weight_path_id"] == "target-matching",
        "minimal B0 action must use target-matching weights",
    )
    _require(spec["window"]["mode"] == "off", "P3 cannot enable windowing")
    _require(spec["skip_set"] == [], "P3 cannot enable layer skipping")
    _require(spec["k"] == 4, "exact minimal-B0 evidence covers only K4")
    _require(
        spec["target_graph_descriptor"]["query_width"] == spec["k"] + 1,
        "target verification query width must equal K+1",
    )
    _require(
        spec["target_graph_descriptor"]["graph_id"] == "target-k5",
        "K4 verification must use the exact target-k5 graph",
    )
    _require(
        spec["draft_graph_descriptor"]["query_width"] == 1,
        "draft graph query width must be one token",
    )
    _require(
        spec["draft_graph_descriptor"]["graph_id"] == "draft-target-matching-k1",
        "K4 drafting must use the exact target-matching graph",
    )
    for action in actions:
        _require(
            set(action["legal_switch_predecessors"]) == action_ids,
            "minimal B0 actions must permit both K4/OFF predecessors",
        )


def _action_ready(
    action: Mapping[str, Any],
    current_action_id: str,
    runtime_state: Mapping[str, Any],
) -> bool:
    if current_action_id not in action["legal_switch_predecessors"]:
        return False
    available_graphs = set(runtime_state["available_graph_ids"])
    target_graph = action["target_graph_descriptor"]["graph_id"]
    if target_graph not in available_graphs:
        return False
    if action["kind"] == "off":
        return True
    if action["draft_weight_path_id"] not in set(
        runtime_state["available_weight_path_ids"]
    ):
        return False
    draft_graph = action["draft_graph_descriptor"]["graph_id"]
    return draft_graph in available_graphs


def _choose_action(
    *,
    intent: str,
    requested_action_id: str,
    current_action_id: str,
    actions: Mapping[str, Mapping[str, Any]],
    ready: Mapping[str, bool],
    fresh: Mapping[str, bool],
    tie_set: Sequence[str],
    off_action_id: str,
) -> tuple[str, str | None]:
    if intent == "force_off":
        _require(ready[off_action_id], "OFF graph is unavailable")
        return off_action_id, None
    if intent == "probe":
        requested = actions[requested_action_id]
        if requested["kind"] == "speculative" and ready[requested_action_id]:
            _require(
                not fresh[requested_action_id],
                "probe intent is reserved for stale evidence",
            )
            return requested_action_id, None
        return off_action_id, "requested_ineligible"

    requested = actions[requested_action_id]
    if not ready[requested_action_id]:
        fallback = current_action_id if current_action_id in tie_set else off_action_id
        return fallback, "requested_ineligible"
    if requested["kind"] == "speculative" and not fresh[requested_action_id]:
        fallback = current_action_id if current_action_id in tie_set else off_action_id
        return fallback, "stale_evidence"
    if current_action_id in tie_set:
        if current_action_id == requested_action_id:
            return current_action_id, None
        return current_action_id, "incumbent_tie"
    if requested_action_id in tie_set:
        return requested_action_id, None
    if off_action_id in tie_set:
        return off_action_id, "outside_tie_set"
    return sorted(tie_set)[0], "outside_tie_set"


def _validate_runtime_state(
    *,
    state: Mapping[str, Any],
    boot: Mapping[str, Any],
    trace: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> None:
    _require(state["binding_id"] == trace["binding_id"], "KV binding changed")
    _require(state["pool_id"] == trace["target_pool_id"], "KV pool changed")
    _require(
        state["true_slot_mapping_id"] == trace["canonical_true_slot_mapping_id"],
        "true slot mapping changed",
    )
    _require(
        state["binding_id"] == boot["kv"]["binding_id"],
        "trace binding differs from boot",
    )
    _require(
        state["pool_id"] == runtime["target_pool_id"],
        "trace pool differs from runtime snapshot",
    )
    resident_graphs = set(boot["resident_graph_ids"])
    _require(
        set(state["available_graph_ids"]) <= resident_graphs,
        "runtime exposes an unregistered graph",
    )
    resident_paths = {path["path_id"] for path in boot["draft_weight_paths"]}
    _require(
        set(state["available_weight_path_ids"]) <= resident_paths,
        "runtime exposes an unregistered weight path",
    )
    _require(
        set(state["draft_weight_versions"]) == resident_paths,
        "runtime weight-version map must cover every resident path",
    )
    for path in boot["draft_weight_paths"]:
        if path["kind"] == "target_matching":
            _require(
                state["draft_weight_versions"][path["path_id"]]
                == state["target_weight_version"],
                "target-matching draft version diverged from target",
            )


def replay_trace(
    environment: Mapping[str, Any],
    workload: Mapping[str, Any],
    candidate: Mapping[str, Any],
    boot: Mapping[str, Any],
    registry: Mapping[str, Any],
    runtime: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the full package and replay one passive K/OFF trace.

    Returns:
        Machine-readable action accounting and policy events.

    Raises:
        KOffReplayError: If any invariant, selection, or counter check fails.
    """
    validate_p3_package(
        environment,
        workload,
        candidate,
        boot,
        registry,
        runtime,
    )
    _validate_trace_schema(trace)

    _require(trace["boot_class_id"] == boot["boot_class_id"], "trace boot mismatch")
    _require(trace["workload_id"] == workload["workload_id"], "trace workload mismatch")
    _require(trace["binding_id"] == runtime["binding_id"], "trace binding mismatch")
    _require(
        trace["target_pool_id"] == runtime["target_pool_id"],
        "trace target pool mismatch",
    )
    _require(
        trace["canonical_true_slot_mapping_id"]
        == runtime["canonical_true_slot_mapping_id"],
        "trace true-slot mapping mismatch",
    )

    actions = {action["action_id"]: action for action in registry["actions"]}
    off_ids = [
        action_id for action_id, action in actions.items() if action["kind"] == "off"
    ]
    off_action_id = off_ids[0]
    _require(
        trace["initial_action_id"] == off_action_id,
        "trace must start from OFF",
    )
    spec_ids = {
        action_id
        for action_id, action in actions.items()
        if action["kind"] == "speculative"
    }

    segment_ids = [segment["segment_id"] for segment in trace["segments"]]
    _require(
        len(segment_ids) == len(set(segment_ids)),
        "trace segment ids must be unique",
    )

    current_action_id = trace["initial_action_id"]
    next_target_step = 0
    next_engine_step = 0
    observed_steps = {action_id: {0} for action_id in spec_ids}
    aggregates: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {
            "segments": 0,
            "H_target_steps": 0,
            "D_armed": 0,
            "A_accepted": 0,
            "C_clipped": 0,
            "E_committed": 0,
            "decode_time_s": 0.0,
        }
    )
    events: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    probe_steps = 0

    for segment in trace["segments"]:
        segment_id = segment["segment_id"]
        _require(
            segment["start_target_step"] == next_target_step,
            f"segment {segment_id} is not target-step contiguous",
        )
        _require(
            segment["requested_action_id"] in actions,
            f"segment {segment_id} requests an unknown action",
        )
        if segment["intent"] == "force_off":
            _require(
                segment["requested_action_id"] == off_action_id,
                f"segment {segment_id} force_off must request OFF",
            )
        if segment["intent"] == "probe":
            _require(
                segment["requested_action_id"] in spec_ids,
                f"segment {segment_id} probe must request a speculative action",
            )
        state = segment["runtime_state"]
        _validate_runtime_state(
            state=state,
            boot=boot,
            trace=trace,
            runtime=runtime,
        )

        estimates = {
            estimate["action_id"]: estimate
            for estimate in segment["interval_estimates"]
        }
        _require(
            len(estimates) == len(segment["interval_estimates"]),
            f"segment {segment_id} has duplicate interval estimates",
        )
        _require(
            set(estimates) == spec_ids,
            f"segment {segment_id} must carry action-specific estimates",
        )

        ready = {
            action_id: _action_ready(action, current_action_id, state)
            for action_id, action in actions.items()
        }
        _require(ready[off_action_id], f"segment {segment_id} loses OFF fallback")
        fresh = {off_action_id: True}
        speed_intervals: dict[str, tuple[float, float]] = {off_action_id: (1.0, 1.0)}
        for action_id in sorted(spec_ids):
            estimate = estimates[action_id]
            _require(
                estimate["q_lo"] <= estimate["q_hi"],
                f"segment {segment_id} has inverted q interval",
            )
            _require(
                estimate["tau_lo"] <= estimate["tau_hi"],
                f"segment {segment_id} has inverted tau interval",
            )
            _require(
                estimate["tau_hi"] <= actions[action_id]["k"] + 1,
                f"segment {segment_id} tau exceeds K+1",
            )
            observed = estimate["observed_at_target_step"]
            _require(
                observed <= segment["start_target_step"],
                f"segment {segment_id} uses future evidence",
            )
            _require(
                observed in observed_steps[action_id],
                f"segment {segment_id} uses unproven freshness",
            )
            age = segment["start_target_step"] - observed
            fresh[action_id] = age <= trace["policy"]["max_evidence_age_target_steps"]
            if ready[action_id] and fresh[action_id]:
                speed_intervals[action_id] = (
                    estimate["tau_lo"] / estimate["q_hi"],
                    estimate["tau_hi"] / estimate["q_lo"],
                )

        tie_set = interval_tie_set(
            speed_intervals,
            trace["policy"]["epsilon_sel"],
        )
        selected, fallback_reason = _choose_action(
            intent=segment["intent"],
            requested_action_id=segment["requested_action_id"],
            current_action_id=current_action_id,
            actions=actions,
            ready=ready,
            fresh=fresh,
            tie_set=tie_set,
            off_action_id=off_action_id,
        )
        switched = selected != current_action_id
        if switched:
            _require(
                segment["target_step_boundary"],
                f"segment {segment_id} switches inside a target step",
            )

        expected = segment["expected"]
        _require(
            expected["selected_action_id"] == selected,
            f"segment {segment_id} selected-action expectation mismatch",
        )
        _require(
            sorted(expected["tie_set_action_ids"]) == tie_set,
            f"segment {segment_id} tie-set expectation mismatch",
        )
        _require(
            expected["fallback_reason"] == fallback_reason,
            f"segment {segment_id} fallback expectation mismatch",
        )
        _require(
            expected["switched"] == switched,
            f"segment {segment_id} switch expectation mismatch",
        )

        counters = segment["counters"]
        h_steps = counters["H_target_steps"]
        _require(
            counters["D_armed"] <= h_steps,
            f"segment {segment_id} has D_armed greater than H",
        )
        _require(
            counters["E_committed"] + counters["C_clipped"]
            == counters["A_accepted"] + h_steps,
            f"segment {segment_id} violates E+C=A+H",
        )
        selected_action = actions[selected]
        if selected_action["kind"] == "off":
            _require(
                counters["D_armed"] == 0 and counters["A_accepted"] == 0,
                f"segment {segment_id} attributes drafting to OFF",
            )
            expected_target_graph = selected_action["target_graph_descriptor"][
                "graph_id"
            ]
            expected_draft_graph = None
        else:
            _require(
                counters["A_accepted"] <= selected_action["k"] * counters["D_armed"],
                f"segment {segment_id} accepts more than K per armed step",
            )
            expected_target_graph = selected_action["target_graph_descriptor"][
                "graph_id"
            ]
            expected_draft_graph = selected_action["draft_graph_descriptor"]["graph_id"]

        engine_h = 0
        max_scheduled_kv = 0
        for engine_step in segment["engine_steps"]:
            _require(
                engine_step["engine_step_index"] == next_engine_step,
                f"segment {segment_id} engine-step trace is not contiguous",
            )
            next_engine_step += 1
            engine_h += engine_step["active_request_count"]
            max_scheduled_kv = max(
                max_scheduled_kv,
                engine_step["total_scheduled_kv_tokens"],
            )
            _require(
                engine_step["active_request_count"]
                <= workload["concurrency"]["hard_admission_requests"],
                f"segment {segment_id} exceeds admitted concurrency",
            )
            _require(
                engine_step["total_scheduled_kv_tokens"]
                <= workload["live_kv"]["hard_admission_tokens"],
                f"segment {segment_id} exceeds admitted live KV",
            )
            _require(
                engine_step["generated_suffix_min"]
                <= engine_step["generated_suffix_max"],
                f"segment {segment_id} has an inverted suffix range",
            )
            _require(
                engine_step["target_graph_id"] == expected_target_graph,
                f"segment {segment_id} target graph does not match action",
            )
            _require(
                engine_step["draft_graph_id"] == expected_draft_graph,
                f"segment {segment_id} draft graph does not match action",
            )
        _require(
            engine_h == h_steps,
            f"segment {segment_id} H does not match exact engine-step trace",
        )
        minimum_blocks = (
            max_scheduled_kv + boot["resources"]["kv_block_size_tokens"] - 1
        ) // boot["resources"]["kv_block_size_tokens"]
        _require(
            minimum_blocks
            <= counters["shared_target_kv_blocks_in_use"]
            <= boot["resources"]["shared_target_kv_blocks"],
            f"segment {segment_id} shared-KV block accounting is invalid",
        )
        _require(
            counters["preemptions"]
            <= workload["preemption_recompute"]["max_preemptions_per_1000_requests"],
            f"segment {segment_id} exceeds preemption tolerance",
        )
        _require(
            counters["recomputed_tokens"]
            <= workload["preemption_recompute"][
                "max_recomputed_tokens_per_1000_requests"
            ],
            f"segment {segment_id} exceeds recompute tolerance",
        )

        aggregate = aggregates[selected]
        aggregate["segments"] += 1
        for name in (
            "H_target_steps",
            "D_armed",
            "A_accepted",
            "C_clipped",
            "E_committed",
            "decode_time_s",
        ):
            aggregate[name] += counters[name]

        if switched:
            events.append(
                {
                    "kind": "switch",
                    "segment_id": segment_id,
                    "from_action_id": current_action_id,
                    "to_action_id": selected,
                }
            )
        if segment["intent"] == "probe" and selected in spec_ids:
            probe_steps += h_steps
            events.append(
                {
                    "kind": "probe",
                    "segment_id": segment_id,
                    "action_id": selected,
                }
            )
        if fallback_reason is not None:
            events.append(
                {
                    "kind": "fallback",
                    "segment_id": segment_id,
                    "requested_action_id": segment["requested_action_id"],
                    "selected_action_id": selected,
                    "reason": fallback_reason,
                }
            )

        decisions.append(
            {
                "segment_id": segment_id,
                "requested_action_id": segment["requested_action_id"],
                "selected_action_id": selected,
                "tie_set_action_ids": tie_set,
                "fallback_reason": fallback_reason,
                "switched": switched,
            }
        )
        next_target_step += h_steps
        if selected in spec_ids and counters["D_armed"] > 0:
            observed_steps[selected].add(next_target_step)
        current_action_id = selected

    probe_fraction = probe_steps / next_target_step
    _require(
        probe_fraction <= trace["policy"]["max_probe_target_step_fraction"],
        "synthetic trace exceeds its declared probe-duty budget",
    )

    rendered_aggregates = []
    for action_id in sorted(aggregates):
        row = {"action_id": action_id, **aggregates[action_id]}
        h_steps = int(row["H_target_steps"])
        row["U_unarmed"] = h_steps - int(row["D_armed"])
        row["tau_eff"] = round(int(row["E_committed"]) / h_steps, 6)
        row["decode_time_s"] = round(float(row["decode_time_s"]), 6)
        rendered_aggregates.append(row)

    return {
        "schema_version": 1,
        "status": "pass",
        "trace_id": trace["trace_id"],
        "boot_class_id": boot["boot_class_id"],
        "scored": False,
        "total_target_steps": next_target_step,
        "total_engine_steps": next_engine_step,
        "probe_target_steps": probe_steps,
        "probe_target_step_fraction": round(probe_fraction, 6),
        "final_action_id": current_action_id,
        "action_aggregates": rendered_aggregates,
        "decisions": decisions,
        "events": events,
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise KOffReplayError(f"JSON root must be an object: {path}")
    return value


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--boot", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    """Replay a trace and emit a machine-readable result."""
    args = parse_args()
    try:
        result = replay_trace(
            _load_json(args.environment),
            _load_json(args.workload),
            _load_json(args.candidate),
            _load_json(args.boot),
            _load_json(args.actions),
            _load_json(args.runtime),
            _load_json(args.trace),
        )
    except (OSError, json.JSONDecodeError, KOffReplayError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        return 2

    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
