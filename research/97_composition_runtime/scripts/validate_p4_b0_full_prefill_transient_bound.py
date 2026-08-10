#!/usr/bin/env python3
"""Validate the Phase 97 full-prefill transient-memory rejection bound."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from decimal import ROUND_CEILING, Decimal
from math import ceil
from pathlib import Path
from typing import Any

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]

GIB = 1024**3
KV_BYTES_PER_BLOCK = 2_359_296
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 12288
ELEMENT_BYTES = 2
OBSERVED_KV_BLOCKS = 22090
LAUNCH_FLOOR_BLOCKS = 21682
REQUIRED_LIVE_BLOCKS = 21000

EXPECTED_SOURCE_PATHS = {
    "v9_authorization": (
        "research/97_composition_runtime/data/p4/p4_b0_run_authorization_v9.json"
    ),
    "v9_failure": (
        "research/97_composition_runtime/data/p4/run_b0_value_screen_v8/failure.json"
    ),
    "v9_preparation": (
        "research/97_composition_runtime/data/p4/"
        "run_b0_value_screen_v8/preparation.json"
    ),
    "first_boot_spec": (
        "research/97_composition_runtime/data/p4/run_b0_value_screen_v8/"
        "boot_specs/p4-b0-b1-p1-off.json"
    ),
    "first_plan": (
        "research/97_composition_runtime/data/p4/run_b0_value_screen_v8/"
        "plans/p4-b0-b1-p1-off.json"
    ),
    "prompt_manifest": (
        "research/97_composition_runtime/data/p4/p4_b0_prompt_manifest.json"
    ),
    "model_config": (
        "/data/smcho/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/"
        "b968826d9c46dd6066d109eabc6255188de91218/config.json"
    ),
    "qwen3_model_source": "vllm/model_executor/models/qwen3.py",
    "qwen_mlp_source": "vllm/model_executor/models/qwen2.py",
    "silu_and_mul_source": "vllm/model_executor/layers/activation.py",
}

EXPECTED_TOP_LEVEL_KEYS = {
    "schema_version",
    "bound_id",
    "date",
    "status",
    "evidence_grade",
    "source_artifacts",
    "source_limitations",
    "measurement_geometry",
    "model_geometry",
    "observed_attempt",
    "compiled_mlp_live_set",
    "graph_pool_correction",
    "failed_allocation_at_kv_floor",
    "decision",
    "required_repair",
    "claims",
    "authorizations",
    "next_artifact",
}


class FullPrefillTransientBoundError(ValueError):
    """Raised when the transient-memory bound drifts or overclaims authority."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FullPrefillTransientBoundError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FullPrefillTransientBoundError(
            f"cannot load JSON artifact {path}: {exc}"
        ) from exc
    _require(isinstance(value, dict), f"JSON artifact must be an object: {path}")
    return value


def _artifact_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    resolved = (REPO_ROOT / path).resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise FullPrefillTransientBoundError(
            f"artifact escapes repository: {resolved}"
        ) from exc
    return resolved


def _hash(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise FullPrefillTransientBoundError(
            f"cannot hash source artifact {path}: {exc}"
        ) from exc


def _validate_sources(bound: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    references = bound["source_artifacts"]
    _require(
        set(references) == set(EXPECTED_SOURCE_PATHS),
        "transient bound source closure drifted",
    )
    loaded: dict[str, dict[str, Any]] = {}
    for role, expected_path in EXPECTED_SOURCE_PATHS.items():
        reference = references[role]
        _require(
            reference.get("path") == expected_path,
            f"unexpected source path for {role}",
        )
        path = _artifact_path(expected_path)
        _require(path.is_file(), f"missing source artifact: {path}")
        actual_hash = _hash(path)
        _require(
            reference.get("sha256") == actual_hash,
            f"artifact hash mismatch for {expected_path}",
        )
        if path.suffix == ".json":
            loaded[role] = _load_json(path)
    return loaded


def _validate_failure_captures(failure: Mapping[str, Any]) -> None:
    evidence = failure["capture_evidence"]
    captures = evidence["captures"]
    _require(len(captures) == 8, "V9 failure must preserve eight captures")
    total_events = 0
    total_commits = 0
    for reference in captures:
        path = _artifact_path(reference["path"])
        _require(path.is_file(), f"missing V9 capture: {path}")
        _require(path.stat().st_size == reference["bytes"], "capture size drifted")
        _require(_hash(path) == reference["sha256"], "capture hash drifted")
        capture = _load_json(path)
        _require(
            capture.get("complete") is True and capture.get("scored") is False,
            "V9 preserved capture is not complete and unscored",
        )
        event_count = len(capture["events"])
        commits = sum(event["counters"]["E_committed"] for event in capture["events"])
        _require(event_count == reference["event_count"], "capture event count drifted")
        _require(commits == reference["committed_tokens"], "capture commits drifted")
        total_events += event_count
        total_commits += commits

    empty = evidence["empty_placeholder"]
    empty_path = _artifact_path(empty["path"])
    _require(empty_path.is_file(), "V9 empty R5 placeholder is missing")
    _require(empty_path.stat().st_size == 0, "V9 R5 placeholder is no longer empty")
    _require(_hash(empty_path) == empty["sha256"], "empty placeholder hash drifted")
    _require(
        total_events == evidence["complete_event_count"] == 16384,
        "V9 complete event total drifted",
    )
    _require(
        total_commits == evidence["complete_committed_tokens"] == 131072,
        "V9 complete commit total drifted",
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
        _require(len(rows) == 32, f"prompt group drifted for {regime_id}/{seed}")
        for start in range(0, len(rows), batch):
            chunk = rows[start : start + batch]
            maxima.append(sum(row["token_count"] for row in chunk))
    return max(maxima)


def _ceil_decimal_bytes(gib: Decimal) -> int:
    return int((gib * GIB).to_integral_value(rounding=ROUND_CEILING))


def _validate_upstream(sources: Mapping[str, Mapping[str, Any]]) -> None:
    authorization = sources["v9_authorization"]
    _require(
        authorization.get("package_id") == "p4-b0-value-screen-run-authorization-v9",
        "bound source is not V9 authorization",
    )
    _require(
        authorization.get("authorizations", {}).get("v9_execution") is True,
        "V9 source did not authorize its consumed execution",
    )

    failure = sources["v9_failure"]
    attempt = failure.get("attempt", {})
    _require(
        failure.get("record_type") == "p4_b0_value_screen_execution_failure"
        and attempt.get("complete_captures_emitted") == 8
        and attempt.get("empty_capture_placeholders") == 1
        and attempt.get("score_emitted") is False
        and failure.get("disposition", {}).get("v9_consumed") is True,
        "V9 failure disposition drifted",
    )
    _require(
        failure.get("diagnostic", {}).get("scope")
        == "full_prefill_transient_activation_hbm_underbound",
        "V9 failure cause drifted",
    )
    _validate_failure_captures(failure)

    preparation = sources["v9_preparation"]
    budget = preparation["full_prefill_budget"]
    _require(
        budget["max_num_batched_tokens"] == 114688
        and budget["effective_scheduler_token_budget"] == 114656
        and budget["regime_max_microbatch_prompt_tokens"]["R5"] == 112304
        and budget["regime_max_microbatch_prompt_tokens"]["R5cot"] == 112908,
        "V9 preparation full-prefill geometry drifted",
    )
    _require(
        preparation["minimum_shared_kv_blocks"] == LAUNCH_FLOOR_BLOCKS,
        "V9 launch floor drifted",
    )

    boot = sources["first_boot_spec"]
    _require(
        boot["boot_id"] == "p4-b0-b1-p1-off"
        and boot["engine"]["max_num_batched_tokens"] == 114688
        and boot["engine"]["gpu_memory_utilization"] == 0.96
        and boot["engine"]["enable_chunked_prefill"] is True,
        "failed boot engine geometry drifted",
    )
    _require(
        boot["minimum_shared_kv_blocks"] == LAUNCH_FLOOR_BLOCKS,
        "failed boot KV floor drifted",
    )

    plan = sources["first_plan"]
    _require(
        plan["boot_action_id"] == "off"
        and plan["minimum_shared_kv_blocks"] == LAUNCH_FLOOR_BLOCKS
        and len(plan["cells"]) == 48,
        "failed boot plan drifted",
    )

    manifest = sources["prompt_manifest"]
    _require(_max_microbatch_tokens(manifest, "R5") == 112304, "R5 tokens drifted")
    _require(
        _max_microbatch_tokens(manifest, "R5cot") == 112908,
        "R5cot tokens drifted",
    )

    config = sources["model_config"]
    _require(
        config["hidden_size"] == HIDDEN_SIZE
        and config["intermediate_size"] == INTERMEDIATE_SIZE
        and config["torch_dtype"] == "bfloat16",
        "Qwen3 model geometry drifted",
    )


def _validate_live_set(bound: Mapping[str, Any]) -> None:
    model = bound["model_geometry"]
    _require(
        model
        == {
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "tensor_parallel_size": 1,
            "activation_dtype": "bfloat16",
            "activation_element_bytes": ELEMENT_BYTES,
            "gate_up_width": 2 * INTERMEDIATE_SIZE,
            "silu_multiply_output_width": INTERMEDIATE_SIZE,
        },
        "bound model geometry drifted",
    )

    live_set = bound["compiled_mlp_live_set"]
    expected_tensors = {
        "attention_output_projection": HIDDEN_SIZE,
        "post_attention_normalized_hidden": HIDDEN_SIZE,
        "residual": HIDDEN_SIZE,
        "gate_up_projection": 2 * INTERMEDIATE_SIZE,
        "silu_multiply_output": INTERMEDIATE_SIZE,
    }
    tensors = live_set["tensors"]
    _require(len(tensors) == len(expected_tensors), "MLP live tensor set drifted")
    observed = {row["name"]: row["width"] for row in tensors}
    _require(observed == expected_tensors, "MLP live tensor widths drifted")
    _require(
        all(row["multiplicity"] == 1 for row in tensors),
        "MLP live tensor multiplicity drifted",
    )
    total_width = sum(row["width"] * row["multiplicity"] for row in tensors)
    bytes_per_token = total_width * ELEMENT_BYTES
    _require(
        live_set["total_width"] == total_width == 49152
        and live_set["bytes_per_prompt_token"] == bytes_per_token == 98304,
        "MLP live-set arithmetic drifted",
    )

    for regime_id, prompt_tokens in (("R5", 112304), ("R5cot", 112908)):
        row = live_set["regime_bounds"][regime_id]
        gate_up = prompt_tokens * 2 * INTERMEDIATE_SIZE * ELEMENT_BYTES
        silu = prompt_tokens * INTERMEDIATE_SIZE * ELEMENT_BYTES
        other = prompt_tokens * 3 * HIDDEN_SIZE * ELEMENT_BYTES
        total = gate_up + silu + other
        expected = {
            "prompt_tokens": prompt_tokens,
            "gate_up_projection_bytes": gate_up,
            "silu_multiply_output_bytes": silu,
            "other_live_rows_bytes": other,
            "live_set_bytes": total,
            "live_set_gib": total / GIB,
            "kv_block_equivalent_ceil": ceil(total / KV_BYTES_PER_BLOCK),
        }
        _require(row == expected, f"{regime_id} live-set bound arithmetic drifted")


def _validate_graph_and_floor(bound: Mapping[str, Any]) -> None:
    graph = bound["graph_pool_correction"]
    estimate = Decimal(str(graph["estimated_pool_gib_reported"]))
    actual = Decimal(str(graph["actual_pool_gib_reported"]))
    decimals = graph["reported_values_decimal_places"]
    _require(decimals == 2, "graph report precision drifted")
    half_unit = Decimal(5).scaleb(-(decimals + 1))
    minimum_underestimate = (actual - half_unit) - (estimate + half_unit)
    minimum_bytes = _ceil_decimal_bytes(minimum_underestimate)
    minimum_blocks = ceil(minimum_bytes / KV_BYTES_PER_BLOCK)
    corrected_capacity = OBSERVED_KV_BLOCKS - minimum_blocks
    expected_graph = {
        "estimated_pool_gib_reported": 0.5,
        "actual_pool_gib_reported": 4.9,
        "nominal_underestimate_gib": 4.4,
        "reported_values_decimal_places": 2,
        "minimum_underestimate_gib_under_round_to_nearest": 4.39,
        "minimum_underestimate_bytes": minimum_bytes,
        "kv_bytes_per_block": KV_BYTES_PER_BLOCK,
        "minimum_underestimate_blocks_ceil": minimum_blocks,
        "observed_shared_kv_blocks": OBSERVED_KV_BLOCKS,
        "optimistic_graph_corrected_capacity_blocks": corrected_capacity,
        "required_live_kv_blocks": REQUIRED_LIVE_BLOCKS,
        "launch_floor_blocks": LAUNCH_FLOOR_BLOCKS,
        "shortfall_to_required_live_blocks": REQUIRED_LIVE_BLOCKS - corrected_capacity,
        "shortfall_to_launch_floor_blocks": LAUNCH_FLOOR_BLOCKS - corrected_capacity,
        "decision": "fail_even_under_optimistic_report_rounding",
    }
    _require(graph == expected_graph, "graph-pool correction arithmetic drifted")
    _require(
        corrected_capacity < REQUIRED_LIVE_BLOCKS,
        "graph-corrected capacity no longer proves rejection",
    )

    floor = bound["failed_allocation_at_kv_floor"]
    releasable_blocks = OBSERVED_KV_BLOCKS - LAUNCH_FLOOR_BLOCKS
    releasable_bytes = releasable_blocks * KV_BYTES_PER_BLOCK
    reported_free = Decimal(str(floor["allocator_reported_free_gib"]))
    free_upper_gib = reported_free + half_unit
    free_upper_bytes = _ceil_decimal_bytes(free_upper_gib)
    available = free_upper_bytes + releasable_bytes
    request = 112304 * INTERMEDIATE_SIZE * ELEMENT_BYTES
    deficit = request - available
    additional_blocks = ceil(deficit / KV_BYTES_PER_BLOCK)
    expected_floor = {
        "observed_shared_kv_blocks": OBSERVED_KV_BLOCKS,
        "launch_floor_blocks": LAUNCH_FLOOR_BLOCKS,
        "releasable_blocks_before_floor": releasable_blocks,
        "kv_bytes_per_block": KV_BYTES_PER_BLOCK,
        "optimistic_releasable_bytes": releasable_bytes,
        "allocator_reported_free_gib": 1.45,
        "reported_values_decimal_places": 2,
        "optimistic_free_upper_bound_gib": 1.455,
        "optimistic_free_upper_bound_bytes": free_upper_bytes,
        "optimistic_bytes_available_at_floor": available,
        "failed_request_bytes": request,
        "minimum_deficit_bytes": deficit,
        "additional_blocks_needed_ceil": additional_blocks,
        "maximum_kv_blocks_compatible_with_failed_request": OBSERVED_KV_BLOCKS
        - releasable_blocks
        - additional_blocks,
        "decision": "launch_floor_cannot_cover_first_failed_allocation",
    }
    _require(floor == expected_floor, "KV-floor allocation arithmetic drifted")
    _require(deficit > 0, "KV-floor bound no longer proves rejection")


def validate_bound(bound: Mapping[str, Any]) -> dict[str, Any]:
    """Validate sources, arithmetic, decision, and authority boundaries.

    Args:
        bound: Parsed transient-memory bound.

    Returns:
        Compact validation result.

    Raises:
        FullPrefillTransientBoundError: If evidence or claims drift.
    """
    _require(set(bound) == EXPECTED_TOP_LEVEL_KEYS, "bound top-level fields drifted")
    _require(
        bound.get("schema_version") == 1
        and bound.get("bound_id") == "p4-b0-full-prefill-transient-bound-v1"
        and bound.get("status") == "reject_current_full_prefill_geometry"
        and bound.get("evidence_grade")
        == "observed_failure_plus_static_live_set_lower_bound",
        "bound identity or evidence grade drifted",
    )
    sources = _validate_sources(bound)
    _validate_upstream(sources)
    _validate_live_set(bound)
    _validate_graph_and_floor(bound)

    limitations = bound["source_limitations"]
    _require(
        limitations["allocator_stderr_log_persisted_separately"] is False
        and limitations["allocator_values_preserved_in_failure_record"] is True
        and limitations["torchinductor_cache_used_as_required_evidence"] is False,
        "source-evidence limitation drifted",
    )
    shared_kv = bound["measurement_geometry"]["shared_kv"]
    _require(
        shared_kv
        == {
            "owner": "target",
            "pool_count": 1,
            "private_draft_pool_count": 0,
            "dtype": "bfloat16",
            "layer_count": 36,
        },
        "shared-KV invariant drifted",
    )
    _require(
        bound["decision"]["state"] == "reject"
        and bound["decision"]["scope"]
        == "fresh_value_screen_authorization_with_114688_0.96_full_microbatch_prefill",
        "rejection decision drifted",
    )
    _require(
        not any(bound["authorizations"].values()),
        "transient bound must not grant authority",
    )
    _require(
        not any(bound["claims"].values()),
        "transient bound contains a positive downstream claim",
    )
    next_artifact = bound["next_artifact"]
    _require(
        next_artifact
        == {
            "kind": "p4_b0_chunked_prefill_cohort_barrier_design_and_cpu_proof",
            "separate_gpu_probe_authorization_required": True,
            "separate_v10_authorization_required": True,
        },
        "next-artifact boundary drifted",
    )
    return {
        "status": "pass",
        "bound_id": bound["bound_id"],
        "decision": "reject_current_full_prefill_geometry",
        "complete_captures_preserved": 8,
        "score_emitted": False,
        "r5_live_set_bytes": bound["compiled_mlp_live_set"]["regime_bounds"]["R5"][
            "live_set_bytes"
        ],
        "r5cot_live_set_bytes": bound["compiled_mlp_live_set"]["regime_bounds"][
            "R5cot"
        ]["live_set_bytes"],
        "optimistic_graph_corrected_capacity_blocks": bound["graph_pool_correction"][
            "optimistic_graph_corrected_capacity_blocks"
        ],
        "minimum_floor_deficit_bytes": bound["failed_allocation_at_kv_floor"][
            "minimum_deficit_bytes"
        ],
        "gpu_probe_authorized": False,
        "v10_authorized": False,
    }


def parse_args() -> argparse.Namespace:
    """Parse the validator command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bound", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Validate a bound and print a stable JSON summary."""
    args = parse_args()
    try:
        result = validate_bound(_load_json(args.bound))
    except FullPrefillTransientBoundError as exc:
        raise SystemExit(f"P4 transient bound rejected: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
