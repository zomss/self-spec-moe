#!/usr/bin/env python3
"""Validate the Phase 97 conservative boot-static w512 B0 resource bound."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
SCHEMA_PATH = PHASE_DIR / "schemas" / "p4_b0_resource_bound.schema.json"

MIB = 1024**2
GIB = 1024**3
WINDOW = 512
SINKS = 16
K = 4
BUFFER_ELEMENT_BYTES_UPPER = 8
WINDOW_BUFFER_QUANTUM = MIB
GRAPH_REPORTED_GIB = Decimal("0.54")
GRAPH_REPORT_ALLOWANCE_GIB = Decimal("0.01")
GRAPH_MULTIPLIER = 3
GRAPH_ROUNDING_QUANTUM = GIB
METADATA_RESERVE_MIB = 256
SAFETY_BASIS_POINTS = 500
SAFETY_MINIMUM_BYTES = 4 * GIB

EXPECTED_SOURCE_PATHS = {
    "measured_b0": (
        "research/97_composition_runtime/data/preflight/candidate_b0_measured.json"
    ),
    "fixed_environment": (
        "research/97_composition_runtime/data/preflight/"
        "environment_qwen3_8b_h100_tp1.json"
    ),
    "workload": (
        "research/97_composition_runtime/data/preflight/workload_rl_capacity_v1.json"
    ),
    "baseline_boot_proxy": (
        "research/97_composition_runtime/data/boot_proxy/baseline.json"
    ),
    "baseline_boot_log": (
        "research/97_composition_runtime/logs/boot_proxy/baseline.log"
    ),
    "base_boot_manifest": (
        "research/97_composition_runtime/data/p3/boot_b0_minimal_k4.json"
    ),
    "rejected_w512_projection": (
        "research/97_composition_runtime/data/p4/"
        "candidate_b0_window512_masked_projection.json"
    ),
    "value_screen_preregistration": (
        "research/97_composition_runtime/data/p4/p4_b0_value_screen_prereg.json"
    ),
    "w512_equivalence": (
        "research/97_composition_runtime/data/p4/p4_w512_acceptance_equivalence.json"
    ),
}

EXPECTED_INVALIDATORS = {
    "source_or_geometry_drift",
    "window_buffers_exceed_reserved_bytes",
    "graph_or_workspace_exceeds_reserved_bytes",
    "recorder_or_runtime_metadata_exceeds_reserved_bytes",
    "shared_kv_pool_or_alias_invariant_fails",
    "post_capture_shared_kv_blocks_below_requirement",
    "unregistered_scratchpad_or_graph_path_enabled",
}


class B0ResourceBoundError(ValueError):
    """Raised when the B0 resource bound drifts or overclaims authority."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise B0ResourceBoundError(message)


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _round_up(value: int, quantum: int) -> int:
    return _ceil_div(value, quantum) * quantum


def _format_json_path(parts: Sequence[Any]) -> str:
    path = "$"
    for part in parts:
        path += f"[{part}]" if isinstance(part, int) else f".{part}"
    return path


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise B0ResourceBoundError(f"cannot load JSON artifact {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON artifact must be an object: {path}")
    return value


def _validate_schema(bound: Mapping[str, Any]) -> None:
    schema = _load_json(SCHEMA_PATH)
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise B0ResourceBoundError(f"invalid resource-bound schema: {exc}") from exc
    errors = sorted(
        Draft202012Validator(schema).iter_errors(bound),
        key=lambda error: [str(part) for part in error.absolute_path],
    )
    if errors:
        first = errors[0]
        path = _format_json_path(list(first.absolute_path))
        raise B0ResourceBoundError(
            f"resource-bound schema rejected {path}: {first.message}"
        )


def _repository_path(relative_path: str) -> Path:
    path = (REPO_ROOT / relative_path).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise B0ResourceBoundError(f"artifact escapes repository: {path}") from exc
    return path


def _validate_sources(bound: Mapping[str, Any]) -> dict[str, Any]:
    references = bound["source_artifacts"]
    _require(
        set(references) == set(EXPECTED_SOURCE_PATHS),
        "resource bound must bind exactly the registered source roles",
    )
    loaded: dict[str, Any] = {}
    for role, expected_path in EXPECTED_SOURCE_PATHS.items():
        reference = references[role]
        _require(
            reference["path"] == expected_path,
            f"source role {role} points to an unexpected artifact",
        )
        path = _repository_path(expected_path)
        _require(path.is_file(), f"missing referenced artifact: {path}")
        content = path.read_bytes()
        actual_hash = hashlib.sha256(content).hexdigest()
        _require(
            actual_hash == reference["sha256"],
            f"artifact hash mismatch for {expected_path}: "
            f"{actual_hash} != {reference['sha256']}",
        )
        loaded[role] = (
            content.decode("utf-8") if role == "baseline_boot_log" else _load_json(path)
        )
    return loaded


def _validate_upstream(sources: Mapping[str, Any]) -> dict[str, int]:
    measured = sources["measured_b0"]
    capacity = measured["evidence"]["capacity"]
    _require(
        measured["candidate_id"] == "qwen3-8b-b0-minimal-k4-measured"
        and measured["capability_class"] == "B0",
        "measured base is not the fixed minimal B0 candidate",
    )
    _require(
        measured["evidence"]["grade"] == "measured_exact"
        and measured["evidence"]["candidate_realization_match"]
        and capacity["measurement_relation"] == "exact_candidate",
        "minimal B0 base is not exact realization-matched evidence",
    )
    _require(
        measured["shared_kv"]
        == {"path": "shared_target", "owner": "target", "pool_count": 1},
        "minimal B0 base no longer has one target-owned shared-KV pool",
    )

    environment = sources["fixed_environment"]
    _require(
        environment["environment_id"] == measured["fixed_environment_id"],
        "measured B0 and fixed environment differ",
    )
    _require(
        environment["self_spec_contract"]["shared_target_kv_required"]
        and environment["self_spec_contract"]["private_draft_kv_pool_count"] == 0,
        "fixed environment no longer requires shared target KV only",
    )
    _require(
        capacity["kv_block_size_tokens"] == environment["kv"]["block_size_tokens"]
        and capacity["kv_bytes_per_block"]
        == environment["kv"]["bytes_per_token"]
        * environment["kv"]["block_size_tokens"],
        "measured B0 KV block geometry differs from the fixed environment",
    )

    workload = sources["workload"]
    _require(
        workload["evidence_grade"] == "engineering_assumption"
        and not workload["scored"],
        "resource bound may use only the non-scored engineering envelope",
    )
    live_kv = workload["live_kv"]
    required_tokens = live_kv["hard_admission_tokens"] + live_kv["safety_margin_tokens"]
    required_blocks = _ceil_div(required_tokens, environment["kv"]["block_size_tokens"])

    proxy = sources["baseline_boot_proxy"]
    engine_args = proxy["engine_args"]
    _require(
        proxy["status"] == "booted"
        and proxy["boot_class"] == "baseline"
        and proxy["cache"]["num_gpu_blocks"]
        == capacity["available_shared_target_kv_blocks"],
        "baseline boot proxy does not match measured B0 capacity",
    )
    _require(
        proxy["cache"]["block_size"] == capacity["kv_block_size_tokens"]
        and proxy["cache"]["kv_cache_size_tokens"]
        == capacity["available_shared_target_kv_blocks"]
        * capacity["kv_block_size_tokens"],
        "baseline boot proxy KV geometry drifted",
    )

    boot = sources["base_boot_manifest"]
    _require(
        boot["boot_class_id"] == "qwen3-8b-b0-minimal-k4"
        and boot["admitted_windows"] == [0]
        and boot["max_k"] == K,
        "base boot manifest is no longer minimal K4 without a window",
    )
    _require(
        boot["kv"]["path"] == "shared_target"
        and boot["kv"]["owner"] == "target"
        and boot["kv"]["pool_count"] == 1,
        "base boot manifest no longer has one target-owned KV pool",
    )

    projection = sources["rejected_w512_projection"]
    projection_evidence = projection["evidence"]
    _require(
        projection_evidence["grade"] == "static_proxy_projection"
        and not projection_evidence["candidate_realization_match"]
        and projection_evidence["capacity"]["measurement_relation"]
        == "optimistic_proxy_ceiling",
        "older w512 projection was relabelled or ceased to fail closed",
    )

    preregistration = sources["value_screen_preregistration"]
    resource_gate = preregistration["decision_rule"]["resource_gate"]
    _require(
        resource_gate["evidence_grade"] == "measured_base_plus_conservative_delta_bound"
        and resource_gate["measurement_relation"]
        == "conservative_candidate_upper_bound"
        and resource_gate["current_state"] == "reject",
        "frozen preregistration resource-gate contract drifted",
    )
    _require(
        "conservative_resource_bound_missing"
        in preregistration["readiness"]["blocking_reason_codes"],
        "frozen preregistration no longer records the resource blocker",
    )

    equivalence = sources["w512_equivalence"]
    _require(
        equivalence["status"] == "pass_acceptance_semantics_only"
        and equivalence["readiness_update"]["remaining"]
        == ["conservative_resource_bound_missing"],
        "w512 equivalence prerequisite is not the current readiness handoff",
    )
    _require(
        not any(equivalence["authorizations"].values()),
        "w512 equivalence source unexpectedly authorizes follow-up work",
    )
    required_environment = equivalence["surrogate_contract"]["required_environment"]
    _require(
        required_environment["VLLM_SELF_SPEC_DRAFT_KV_WINDOW"] == str(WINDOW)
        and required_environment["VLLM_SELF_SPEC_DRAFT_KV_SINKS"] == str(SINKS)
        and required_environment["VLLM_SELF_SPEC_DRAFT_FULLCG"] == "0",
        "w512 resource candidate enables an unregistered scratchpad path",
    )

    log = sources["baseline_boot_log"]
    _require(
        "Estimated CUDA graph memory: 0.47 GiB total" in log
        and "took 0.54 GiB" in log
        and "CUDA graph pool memory: 0.54 GiB (actual)" in log,
        "baseline log no longer contains the registered graph-pool evidence",
    )

    return {
        "available_blocks": capacity["available_shared_target_kv_blocks"],
        "kv_bytes_per_block": capacity["kv_bytes_per_block"],
        "block_size_tokens": capacity["kv_block_size_tokens"],
        "required_blocks": required_blocks,
        "max_model_len": environment["target"]["max_model_len"],
        "usable_hbm_bytes": environment["hardware"]["usable_hbm_bytes"],
        "max_num_batched_tokens": engine_args["max_num_batched_tokens"],
        "max_num_seqs": engine_args["max_num_seqs"],
    }


def _window_buffer_breakdown(rows: int, columns: int) -> dict[str, int]:
    return {
        "window_block_table": rows * columns * BUFFER_ELEMENT_BYTES_UPPER,
        "window_sequence_lengths": rows * BUFFER_ELEMENT_BYTES_UPPER,
        "column_arange": columns * BUFFER_ELEMENT_BYTES_UPPER,
        "source_column_indices": rows * columns * BUFFER_ELEMENT_BYTES_UPPER,
        "sink_column_mask": columns * BUFFER_ELEMENT_BYTES_UPPER,
    }


def _validate_candidate(bound: Mapping[str, Any], facts: Mapping[str, int]) -> None:
    candidate = bound["candidate"]
    geometry = candidate["geometry"]
    _require(
        geometry
        == {
            "max_model_len": facts["max_model_len"],
            "max_num_batched_tokens": facts["max_num_batched_tokens"],
            "max_num_seqs": facts["max_num_seqs"],
            "k": K,
            "window_tokens": WINDOW,
            "sink_tokens": SINKS,
            "block_size_tokens": facts["block_size_tokens"],
        },
        "resource-bound candidate geometry differs from measured minimal B0",
    )
    _require(
        not candidate["candidate_realization_match"]
        and not candidate["exact_post_capture_measurement"],
        "conservative resource bound cannot masquerade as exact measurement",
    )

    base = bound["base_capacity"]
    expected_base = {
        "base_candidate_id": "qwen3-8b-b0-minimal-k4-measured",
        "base_evidence_grade": "measured_exact",
        "base_measurement_relation": "exact_candidate",
        "available_shared_kv_blocks": facts["available_blocks"],
        "kv_bytes_per_block": facts["kv_bytes_per_block"],
        "kv_block_size_tokens": facts["block_size_tokens"],
        "required_live_kv_blocks": facts["required_blocks"],
        "apparent_headroom_blocks": (
            facts["available_blocks"] - facts["required_blocks"]
        ),
        "older_projection_relabelled_exact": False,
    }
    _require(base == expected_base, "base capacity or apparent headroom drifted")


def _validate_reserves(
    bound: Mapping[str, Any], facts: Mapping[str, int]
) -> dict[str, int]:
    reserves = bound["reserve_policy"]
    kv_bytes = facts["kv_bytes_per_block"]

    window = reserves["persistent_window_buffers"]
    rows = facts["max_num_seqs"]
    columns = _ceil_div(facts["max_model_len"], facts["block_size_tokens"])
    breakdown = _window_buffer_breakdown(rows, columns)
    observed_breakdown = {
        row["name"]: row["bytes_upper_bound"] for row in window["buffers"]
    }
    observed_elements = {row["name"]: row["elements"] for row in window["buffers"]}
    expected_elements = {
        "window_block_table": rows * columns,
        "window_sequence_lengths": rows,
        "column_arange": columns,
        "source_column_indices": rows * columns,
        "sink_column_mask": columns,
    }
    _require(
        window["rows"] == rows
        and window["columns"] == columns
        and window["element_bytes_upper_bound"] == BUFFER_ELEMENT_BYTES_UPPER,
        "persistent window-buffer geometry drifted",
    )
    _require(
        observed_breakdown == breakdown and observed_elements == expected_elements,
        "persistent window-buffer breakdown is incomplete or understated",
    )
    derived_window_bytes = sum(breakdown.values())
    reserved_window_bytes = _round_up(derived_window_bytes, WINDOW_BUFFER_QUANTUM)
    _require(
        window["derived_bytes_upper_bound"] == derived_window_bytes
        and window["rounding_quantum_bytes"] == WINDOW_BUFFER_QUANTUM
        and window["reserved_bytes"] == reserved_window_bytes,
        "persistent window-buffer reserve is not the rounded static bound",
    )

    graph = reserves["graph_capture_and_temporary_workspace"]
    reported_gib = Decimal(str(graph["baseline_actual_graph_pool_reported_gib"]))
    allowance_gib = Decimal(str(graph["report_rounding_allowance_gib"]))
    reported_upper_mib = int(
        ((reported_gib + allowance_gib) * 1024).to_integral_value(
            rounding=ROUND_CEILING
        )
    )
    graph_unrounded_bytes = reported_upper_mib * MIB * GRAPH_MULTIPLIER
    reserved_graph_bytes = _round_up(graph_unrounded_bytes, GRAPH_ROUNDING_QUANTUM)
    _require(
        reported_gib == GRAPH_REPORTED_GIB
        and allowance_gib == GRAPH_REPORT_ALLOWANCE_GIB
        and graph["reported_upper_bound_mib"] == reported_upper_mib
        and graph["conservative_multiplier"] == GRAPH_MULTIPLIER
        and graph["rounding_quantum_bytes"] == GRAPH_ROUNDING_QUANTUM
        and graph["reserved_bytes"] == reserved_graph_bytes,
        "graph/workspace reserve understates the registered conservative policy",
    )

    metadata = reserves["recorder_and_runtime_metadata"]
    reserved_metadata_bytes = METADATA_RESERVE_MIB * MIB
    _require(
        metadata["policy_floor_mib"] == METADATA_RESERVE_MIB
        and not metadata["hbm_credit_for_cpu_records"]
        and metadata["reserved_bytes"] == reserved_metadata_bytes,
        "recorder/runtime metadata reserve is below its fixed HBM contingency",
    )

    safety = reserves["hbm_fragmentation_safety"]
    fraction_bytes = _ceil_div(facts["usable_hbm_bytes"] * SAFETY_BASIS_POINTS, 10_000)
    reserved_safety_bytes = max(fraction_bytes, SAFETY_MINIMUM_BYTES)
    _require(
        safety["usable_hbm_bytes"] == facts["usable_hbm_bytes"]
        and safety["fraction_basis_points"] == SAFETY_BASIS_POINTS
        and safety["fraction_bytes"] == fraction_bytes
        and safety["minimum_bytes"] == SAFETY_MINIMUM_BYTES
        and safety["reserved_bytes"] == reserved_safety_bytes,
        "HBM/fragmentation reserve is below five percent or four GiB",
    )

    expected_bytes = {
        "persistent_window_buffers": reserved_window_bytes,
        "graph_capture_and_temporary_workspace": reserved_graph_bytes,
        "recorder_and_runtime_metadata": reserved_metadata_bytes,
        "hbm_fragmentation_safety": reserved_safety_bytes,
    }
    expected_blocks = {
        role: _ceil_div(value, kv_bytes) for role, value in expected_bytes.items()
    }
    for role, expected in expected_blocks.items():
        _require(
            reserves[role]["reserved_blocks"] == expected,
            f"{role} block debit is not independently rounded up",
        )
    return expected_blocks


def _validate_result(
    bound: Mapping[str, Any],
    facts: Mapping[str, int],
    reserve_blocks: Mapping[str, int],
) -> dict[str, int]:
    reserved_total = sum(reserve_blocks.values())
    lower_bound_blocks = facts["available_blocks"] - reserved_total
    headroom_blocks = lower_bound_blocks - facts["required_blocks"]
    headroom_tokens = headroom_blocks * facts["block_size_tokens"]
    expected = {
        "reserved_blocks_total": reserved_total,
        "lower_bound_shared_kv_blocks": lower_bound_blocks,
        "required_live_kv_blocks": facts["required_blocks"],
        "lower_bound_headroom_blocks": headroom_blocks,
        "lower_bound_headroom_tokens": headroom_tokens,
        "decision": "pass" if headroom_blocks >= 0 else "reject",
        "decision_scope": "pre_engineering_resource_readiness_only",
    }
    _require(bound["result"] == expected, "resource-bound result arithmetic drifted")
    _require(
        headroom_blocks >= 0,
        "conservative reserves leave insufficient shared-KV capacity",
    )
    return {
        "reserved_blocks": reserved_total,
        "lower_bound_blocks": lower_bound_blocks,
        "headroom_blocks": headroom_blocks,
        "headroom_tokens": headroom_tokens,
    }


def _validate_boundary(bound: Mapping[str, Any]) -> None:
    _require(
        set(bound["fail_closed"]["bound_invalidators"]) == EXPECTED_INVALIDATORS,
        "resource bound must retain every fail-closed invalidator",
    )
    claims = bound["claims"]
    _require(
        claims["resource_readiness_cleared"]
        and not any(
            claims[field]
            for field in (
                "exact_candidate_measured",
                "gpu_measurement_authorized",
                "p4a_engineering_authorized",
                "action_admitted",
                "performance_claim_allowed",
            )
        ),
        "resource bound overclaims measurement, authority, admission, or value",
    )
    _require(
        not any(bound["authorizations"].values()),
        "resource-bound artifact cannot authorize GPU or P4a work",
    )
    _require(
        bound["readiness_update"]["remaining"] == []
        and bound["next_artifact"]["separate_decision_required"]
        and not bound["next_artifact"]["gpu_command_included_here"],
        "resource readiness must hand off to a separate authorization decision",
    )


def validate_bound(bound: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one resource bound and return its additive readiness summary."""
    _validate_schema(bound)
    sources = _validate_sources(bound)
    facts = _validate_upstream(sources)
    _validate_candidate(bound, facts)
    reserve_blocks = _validate_reserves(bound, facts)
    result = _validate_result(bound, facts, reserve_blocks)
    _validate_boundary(bound)
    return {
        "status": "pass",
        "bound_id": bound["bound_id"],
        "resource_decision": "pass",
        "base_available_shared_kv_blocks": facts["available_blocks"],
        "required_live_kv_blocks": facts["required_blocks"],
        "reserved_blocks_total": result["reserved_blocks"],
        "lower_bound_shared_kv_blocks": result["lower_bound_blocks"],
        "lower_bound_headroom_blocks": result["headroom_blocks"],
        "lower_bound_headroom_tokens": result["headroom_tokens"],
        "cleared_blocker": "conservative_resource_bound_missing",
        "remaining_readiness_blockers": [],
        "gpu_measurement_authorized": False,
        "p4a_engineering_authorized": False,
        "action_admitted": False,
        "next_artifact": "run_ready_b0_value_screen_authorization_package",
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bound", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Validate the requested resource bound and print a JSON summary."""
    args = parse_args()
    result = validate_bound(_load_json(args.bound))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
