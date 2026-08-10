#!/usr/bin/env python3
"""Plan Phase 97 B0/B1 admission from resource evidence."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

PHASE_DIR = Path(__file__).resolve().parents[1]
SCHEMA_DIR = PHASE_DIR / "schemas"

_B1_HBM_TERMS = (
    (
        "co_resident_weight_hbm_bytes",
        "unknown_co_resident_weight_hbm",
        "co_resident_weight_not_reserved",
    ),
    (
        "co_resident_graphs_and_workspaces_hbm_bytes",
        "unknown_graphs_and_workspaces_hbm",
        "graphs_and_workspaces_not_reserved",
    ),
)
_RL_HBM_TERM = (
    "weight_refresh_peak_hbm_bytes",
    "unknown_weight_refresh_peak_hbm",
    "weight_refresh_peak_not_reserved",
)


class ResourcePreflightError(ValueError):
    """Raised when a resource-preflight input is invalid."""


def _format_json_path(parts: Sequence[Any]) -> str:
    path = "$"
    for part in parts:
        if isinstance(part, int):
            path += f"[{part}]"
        else:
            path += f".{part}"
    return path


def load_schema(name: str) -> dict[str, Any]:
    """Load and meta-validate a Phase 97 schema.

    Args:
        name: Schema filename under ``schemas/``.

    Returns:
        Parsed JSON schema.

    Raises:
        ResourcePreflightError: If the schema is missing or invalid.
    """
    path = SCHEMA_DIR / name
    if not path.is_file():
        raise ResourcePreflightError(f"missing schema: {path}")
    schema = json.loads(path.read_text())
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise ResourcePreflightError(f"invalid schema {name}: {exc}") from exc
    return schema


def _validate_schema(instance: Mapping[str, Any], name: str) -> None:
    validator = Draft202012Validator(load_schema(name))
    errors = sorted(
        validator.iter_errors(instance),
        key=lambda error: [str(part) for part in error.absolute_path],
    )
    if not errors:
        return
    first = errors[0]
    path = _format_json_path(list(first.absolute_path))
    raise ResourcePreflightError(f"{name} rejected {path}: {first.message}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ResourcePreflightError(message)


def validate_environment(environment: Mapping[str, Any]) -> None:
    """Validate a fixed environment and its resource arithmetic.

    Args:
        environment: Parsed environment manifest.

    Raises:
        ResourcePreflightError: If schema or cross-field validation fails.
    """
    _validate_schema(environment, "environment.schema.json")
    hardware = environment["hardware"]
    _require(
        hardware["usable_hbm_bytes"] <= hardware["total_hbm_bytes"],
        "usable HBM cannot exceed physical HBM",
    )
    parallelism = environment["parallelism"]
    required_devices = (
        parallelism["tensor_parallel_size"] * parallelism["pipeline_parallel_size"]
    )
    _require(
        required_devices == hardware["accelerators_per_replica"],
        "accelerators_per_replica must match TP multiplied by PP",
    )


def validate_workload(
    environment: Mapping[str, Any],
    workload: Mapping[str, Any],
) -> None:
    """Validate a workload envelope against its fixed environment.

    Args:
        environment: Parsed and validated environment manifest.
        workload: Parsed workload envelope.

    Raises:
        ResourcePreflightError: If schema or envelope checks fail.
    """
    _validate_schema(workload, "workload.schema.json")
    concurrency = workload["concurrency"]
    _require(
        concurrency["high_percentile_requests"]
        <= concurrency["hard_admission_requests"],
        "high-percentile concurrency exceeds the hard admission bound",
    )

    token_lengths = workload["token_lengths"]
    for label in ("prompt", "requested_output"):
        envelope = token_lengths[label]
        _require(
            envelope["high_percentile_tokens"] <= envelope["hard_max_tokens"],
            f"{label} high-percentile length exceeds its hard maximum",
        )
    total_max = (
        token_lengths["prompt"]["hard_max_tokens"]
        + token_lengths["requested_output"]["hard_max_tokens"]
    )
    _require(
        total_max <= environment["target"]["max_model_len"],
        "prompt plus requested-output maxima exceed max_model_len",
    )

    live_kv = workload["live_kv"]
    _require(
        live_kv["high_percentile_tokens"] <= live_kv["hard_admission_tokens"],
        "high-percentile live KV exceeds the hard admission bound",
    )
    _require(
        max(live_kv["frozen_trace_tokens"]) <= live_kv["hard_admission_tokens"],
        "frozen live-KV trace exceeds the hard admission bound",
    )
    _require(
        not (
            workload["evidence_grade"] == "engineering_assumption"
            and workload["scored"]
        ),
        "an assumed workload envelope cannot authorize scored work",
    )


def validate_candidate(candidate: Mapping[str, Any]) -> None:
    """Validate one resource-candidate evidence artifact.

    Args:
        candidate: Parsed B0 or B1 resource candidate.

    Raises:
        ResourcePreflightError: If schema or class semantics fail.
    """
    _validate_schema(candidate, "boot_candidate.schema.json")
    paths = candidate["resident_objects"]["draft_weight_paths"]
    path_ids = [str(path["path_id"]) for path in paths]
    _require(
        len(path_ids) == len(set(path_ids)),
        "resident draft weight path ids must be unique",
    )
    terms = candidate["resource_terms"]
    if candidate["capability_class"] == "B0":
        for name, term in terms.items():
            _require(
                term["grade"] == "not_applicable",
                f"B0 resource term {name} must be not_applicable",
            )


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _reason(code: str, detail: str) -> dict[str, str]:
    return {"code": code, "detail": detail}


def _append_b1_term_reason(
    reasons: list[dict[str, str]],
    name: str,
    unknown_code: str,
    unreserved_code: str,
    term: Mapping[str, Any],
) -> None:
    if term["grade"] == "unknown":
        reasons.append(
            _reason(
                unknown_code,
                f"{name} has no conservative measured value",
            )
        )
    elif not term["included_in_capacity_evidence"]:
        reasons.append(
            _reason(
                unreserved_code,
                f"{name} is not reserved by the capacity measurement",
            )
        )


def plan_candidate(
    environment: Mapping[str, Any],
    workload: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Return one fail-closed boot-class admission decision.

    Args:
        environment: Valid fixed environment manifest.
        workload: Valid workload envelope.
        candidate: Valid resource candidate.

    Returns:
        Machine-readable admission decision and capacity arithmetic.
    """
    reasons: list[dict[str, str]] = []
    capacity = candidate["evidence"]["capacity"]
    environment_kv = environment["kv"]
    hardware = environment["hardware"]

    if candidate["fixed_environment_id"] != environment["environment_id"]:
        reasons.append(
            _reason(
                "fixed_environment_mismatch",
                "candidate evidence belongs to another environment",
            )
        )
    if capacity["usable_hbm_bytes"] != hardware["usable_hbm_bytes"]:
        reasons.append(
            _reason(
                "usable_hbm_mismatch",
                "candidate and environment usable-HBM budgets differ",
            )
        )
    if capacity["sampled_peak_hbm_bytes"] > hardware["total_hbm_bytes"]:
        reasons.append(
            _reason(
                "sampled_peak_exceeds_physical_hbm",
                "sampled peak is larger than physical HBM",
            )
        )

    expected_block_size = environment_kv["block_size_tokens"]
    expected_block_bytes = expected_block_size * environment_kv["bytes_per_token"]
    block_contract_matches = True
    if capacity["kv_block_size_tokens"] != expected_block_size:
        block_contract_matches = False
        reasons.append(
            _reason(
                "kv_block_size_mismatch",
                "candidate block size differs from the target KV spec",
            )
        )
    if capacity["kv_bytes_per_block"] != expected_block_bytes:
        block_contract_matches = False
        reasons.append(
            _reason(
                "kv_block_bytes_mismatch",
                "candidate block bytes differ from the target KV spec",
            )
        )

    exact_capacity = (
        candidate["evidence"]["grade"] == "measured_exact"
        and candidate["evidence"]["candidate_realization_match"]
        and capacity["measurement_relation"] == "exact_candidate"
    )
    if not exact_capacity:
        reasons.append(
            _reason(
                "capacity_evidence_not_exact",
                "a static path proxy cannot authorize the co-resident class",
            )
        )

    capability = candidate["capability_class"]
    terms = candidate["resource_terms"]
    if capability == "B1":
        for name, unknown_code, unreserved_code in _B1_HBM_TERMS:
            _append_b1_term_reason(
                reasons,
                name,
                unknown_code,
                unreserved_code,
                terms[name],
            )
        if workload["mode"] == "rl_rollout":
            _append_b1_term_reason(
                reasons,
                _RL_HBM_TERM[0],
                _RL_HBM_TERM[1],
                _RL_HBM_TERM[2],
                terms[_RL_HBM_TERM[0]],
            )
            host_term = terms["weight_refresh_pinned_host_bytes"]
            if host_term["grade"] == "unknown":
                reasons.append(
                    _reason(
                        "unknown_weight_refresh_pinned_host",
                        "RL refresh pinned-host memory is unknown",
                    )
                )
            elif host_term["value_bytes"] > hardware["pinned_host_budget_bytes"]:
                reasons.append(
                    _reason(
                        "pinned_host_budget_exceeded",
                        "RL refresh exceeds the pinned-host budget",
                    )
                )

    live_kv = workload["live_kv"]
    required_tokens = live_kv["hard_admission_tokens"] + live_kv["safety_margin_tokens"]
    required_blocks = _ceil_div(required_tokens, expected_block_size)
    available_blocks = capacity["available_shared_target_kv_blocks"]
    headroom_blocks: int | None = None
    if block_contract_matches:
        headroom_blocks = available_blocks - required_blocks
        if headroom_blocks < 0:
            reasons.append(
                _reason(
                    "insufficient_shared_kv_capacity",
                    f"shared KV is short by {-headroom_blocks} blocks",
                )
            )

    return {
        "candidate_id": candidate["candidate_id"],
        "capability_class": capability,
        "decision": "reject" if reasons else "admit",
        "evidence_grade": candidate["evidence"]["grade"],
        "capacity_measurement_relation": capacity["measurement_relation"],
        "required_live_kv_tokens": required_tokens,
        "required_live_kv_blocks": required_blocks,
        "available_shared_target_kv_blocks": available_blocks,
        "shared_target_kv_headroom_blocks": headroom_blocks,
        "frozen_trace_peak_tokens": max(live_kv["frozen_trace_tokens"]),
        "reasons": reasons,
    }


def plan_boot_classes(
    environment: Mapping[str, Any],
    workload: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate inputs and plan all requested boot classes.

    Args:
        environment: Fixed deployment environment.
        workload: Workload resource envelope.
        candidates: B0/B1 resource-evidence candidates.

    Returns:
        Aggregate preflight result.

    Raises:
        ResourcePreflightError: If an input is invalid or duplicated.
    """
    validate_environment(environment)
    validate_workload(environment, workload)
    _require(bool(candidates), "at least one boot candidate is required")
    for candidate in candidates:
        validate_candidate(candidate)

    candidate_ids = [str(candidate["candidate_id"]) for candidate in candidates]
    _require(
        len(candidate_ids) == len(set(candidate_ids)),
        "boot candidate ids must be unique",
    )
    decisions = [
        plan_candidate(environment, workload, candidate) for candidate in candidates
    ]
    admitted = [
        decision["candidate_id"]
        for decision in decisions
        if decision["decision"] == "admit"
    ]
    return {
        "schema_version": 1,
        "status": "pass" if admitted else "no_feasible_candidate",
        "environment_id": environment["environment_id"],
        "workload_id": workload["workload_id"],
        "workload_evidence_grade": workload["evidence_grade"],
        "scored": False,
        "admitted_candidate_ids": admitted,
        "decisions": decisions,
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ResourcePreflightError(f"JSON root must be an object: {path}")
    return value


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        type=Path,
        action="append",
        required=True,
        help="B0/B1 resource candidate; repeat to compare classes",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    """Run the resource preflight and emit a JSON plan."""
    args = parse_args()
    try:
        result = plan_boot_classes(
            _load_json(args.environment),
            _load_json(args.workload),
            [_load_json(path) for path in args.candidate],
        )
    except (OSError, json.JSONDecodeError, ResourcePreflightError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        return 2

    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
