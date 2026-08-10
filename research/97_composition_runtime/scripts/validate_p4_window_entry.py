#!/usr/bin/env python3
"""Validate the fail-closed Phase 97 P4 runtime-window entry package."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
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
from validate_shared_kv import (
    validate_action_registry,
    validate_runtime_snapshot,
)

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
SCHEMA_PATH = PHASE_DIR / "schemas" / "p4_window_entry.schema.json"

OFF_ACTION_ID = "off"
K4_ACTION_ID = "target-matching-k4"
WINDOW_ACTION_ID = "target-matching-w512-masked-k4"
EXPECTED_PHASE96_SUPPORT = {
    "window_value_selection",
    "acceptance_only_classification",
    "separate_boot_cost_transfer",
    "interval_fix_diagnostic",
}
EXPECTED_PENDING_GATES = {
    "runtime_action_transport",
    "live_shared_kv_and_true_slot",
    "target_local_controls",
    "exact_post_capture_resource",
    "transition_overhead",
}
EXPECTED_NOT_APPROVED_GATES = {"phase96_f_resident_value"}


class P4WindowEntryError(ValueError):
    """Raised when the P4 entry package overclaims or breaks its base."""


def _format_json_path(parts: Sequence[Any]) -> str:
    path = "$"
    for part in parts:
        path += f"[{part}]" if isinstance(part, int) else f".{part}"
    return path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise P4WindowEntryError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise P4WindowEntryError(f"cannot load JSON artifact {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON artifact must be an object: {path}")
    return value


def _validate_schema(entry: Mapping[str, Any]) -> None:
    schema = _load_json(SCHEMA_PATH)
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise P4WindowEntryError(f"invalid P4 entry schema: {exc}") from exc
    errors = sorted(
        Draft202012Validator(schema).iter_errors(entry),
        key=lambda error: [str(part) for part in error.absolute_path],
    )
    if errors:
        first = errors[0]
        path = _format_json_path(list(first.absolute_path))
        raise P4WindowEntryError(f"P4 entry schema rejected {path}: {first.message}")


def _artifact_path(reference: Mapping[str, Any]) -> Path:
    path = (REPO_ROOT / str(reference["path"])).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise P4WindowEntryError(f"artifact escapes repository: {path}") from exc
    _require(path.is_file(), f"missing referenced artifact: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = str(reference["sha256"])
    _require(
        actual == expected,
        f"artifact hash mismatch for {reference['path']}: {actual} != {expected}",
    )
    return path


def _load_artifact(reference: Mapping[str, Any]) -> dict[str, Any]:
    return _load_json(_artifact_path(reference))


def _action_by_id(
    registry: Mapping[str, Any],
    action_id: str,
) -> dict[str, Any]:
    matches = [
        action for action in registry["actions"] if action["action_id"] == action_id
    ]
    _require(len(matches) == 1, f"expected exactly one action {action_id!r}")
    return matches[0]


def _validate_phase96_evidence(entry: Mapping[str, Any]) -> None:
    evidence = entry["phase96_evidence"]
    supports = [row["supports"] for row in evidence]
    _require(
        set(supports) == EXPECTED_PHASE96_SUPPORT,
        "P4 entry must bind exactly the four frozen Phase 96 evidence roles",
    )
    _require(
        len(supports) == len(set(supports)),
        "Phase 96 evidence roles must be unique",
    )
    for row in evidence:
        _artifact_path(row["artifact"])


def _derive_registry(
    base_registry: Mapping[str, Any],
    proposed_boot_id: str,
    selected: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a schema-only three-action registry from the preregistration.

    The temporary zero transition cost exists only to exercise the existing
    action schema. It is not persisted evidence, and the entry manifest keeps
    the action ineligible until a measured transition value is supplied.
    """
    registry = copy.deepcopy(base_registry)
    registry["boot_class_id"] = proposed_boot_id
    predecessors = list(selected["legal_switch_predecessors"])
    for action in registry["actions"]:
        action["legal_switch_predecessors"] = predecessors

    candidate = copy.deepcopy(selected)
    candidate.pop("base_action_id")
    candidate.pop("transition_cost")
    candidate["measured_transition_cost_us"] = 0.0
    registry["actions"].append(candidate)
    return registry


def _validate_window_contract(
    entry: Mapping[str, Any],
    base_boot: Mapping[str, Any],
    base_registry: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    selection = entry["selection"]
    _require(selection["window_value"] == 512, "P4 selected window must be w512")
    _require(selection["sink_tokens"] == 16, "P4 selected sink count must be 16")
    _require(
        not selection["phase96_exact_action_match"],
        "Phase 96 did not measure the target-matching masked B0 action",
    )

    base_ids = {action["action_id"] for action in base_registry["actions"]}
    _require(
        base_ids == {OFF_ACTION_ID, K4_ACTION_ID},
        "P4 must extend the frozen two-action P3 registry",
    )
    base_k4 = _action_by_id(base_registry, K4_ACTION_ID)
    selected = entry["selected_action"]
    _require(
        selected["action_id"] == WINDOW_ACTION_ID,
        "unexpected P4 window action id",
    )
    _require(
        selected["base_action_id"] == K4_ACTION_ID,
        "masked window must name K4 as its proper-subset base",
    )
    _require(
        selected["window"]["value"] == selection["window_value"],
        "selected action and selection window disagree",
    )

    same_as_base = (
        "draft_weight_path_id",
        "realization",
        "skip_set",
        "k",
        "target_graph_descriptor",
        "draft_graph_descriptor",
        "requires_current_draft_weight",
    )
    for field in same_as_base:
        _require(
            selected[field] == base_k4[field],
            f"masked window must reuse K4 field {field}",
        )
    expected_ids = {OFF_ACTION_ID, K4_ACTION_ID, WINDOW_ACTION_ID}
    _require(
        set(selected["legal_switch_predecessors"]) == expected_ids,
        "window transition graph must be closed over OFF, K4, and w512",
    )

    delta = entry["proposed_boot_delta"]
    _require(
        delta["base_boot_class_id"] == base_boot["boot_class_id"],
        "proposed boot delta names the wrong base",
    )
    _require(
        delta["shared_kv_binding_id"] == base_boot["kv"]["binding_id"],
        "proposed window changes the shared-KV binding",
    )
    _require(
        delta["kv_owner"] == base_boot["kv"]["owner"],
        "proposed window changes KV ownership",
    )
    _require(
        delta["kv_pool_count"] == base_boot["kv"]["pool_count"],
        "proposed window changes KV pool count",
    )
    _require(
        delta["max_k"] == base_boot["max_k"] == selected["k"],
        "P4 may not change the frozen K4 bound",
    )
    _require(
        set(delta["admitted_windows"]) == {0, 512},
        "P4 boot delta may admit only unmasked and w512",
    )
    _require(
        not delta["new_resident_graph_ids"],
        "acceptance-only masking must not preregister a new graph",
    )

    derived_boot = copy.deepcopy(base_boot)
    derived_boot["boot_class_id"] = delta["proposed_boot_class_id"]
    derived_boot["admitted_windows"] = list(delta["admitted_windows"])
    derived_registry = _derive_registry(
        base_registry,
        derived_boot["boot_class_id"],
        selected,
    )
    validate_action_registry(derived_boot, derived_registry)
    return derived_boot, derived_registry


def _validate_alias_overlay(
    entry: Mapping[str, Any],
    base_snapshot: Mapping[str, Any],
    derived_boot: Mapping[str, Any],
    derived_registry: Mapping[str, Any],
) -> int:
    snapshot = copy.deepcopy(base_snapshot)
    snapshot["boot_class_id"] = derived_boot["boot_class_id"]
    view = entry["action_view"]
    _require(view["action_id"] == WINDOW_ACTION_ID, "window action view id mismatch")
    _require(
        view["binding_id"] == snapshot["binding_id"],
        "window action overrides the shared-KV binding",
    )
    _require(
        view["pool_id"] == snapshot["target_pool_id"],
        "window action overrides the target KV pool",
    )
    _require(
        view["true_slot_mapping_id"] == snapshot["canonical_true_slot_mapping_id"],
        "window action changes the canonical true-slot mapping",
    )
    snapshot["action_views"].append(copy.deepcopy(view))
    validate_runtime_snapshot(derived_boot, derived_registry, snapshot)

    alias_count = len(snapshot["layer_bindings"])
    _require(alias_count == 36, f"P4 requires 36 KV aliases, got {alias_count}")
    _require(
        all(row["same_storage"] for row in snapshot["layer_bindings"]),
        "every P4 layer binding must retain exact target storage",
    )
    return alias_count


def _validate_controls(entry: Mapping[str, Any]) -> None:
    controls = entry["controls"]
    expected = {OFF_ACTION_ID, K4_ACTION_ID, WINDOW_ACTION_ID}
    _require(
        set(controls["comparison_action_ids"]) == expected,
        "P4 controls must be exactly OFF, K4, and masked w512",
    )
    _require(
        controls["proper_subset_action_ids"] == [K4_ACTION_ID],
        "the w512 singleton proper subset must be target-matching K4",
    )
    required_match = {
        "target_weight_version",
        "draft_weight_version",
        "boot_class",
        "shared_kv_binding",
        "hardware",
        "parallel_layout",
        "workload_trace",
        "batch",
        "context",
        "generated_suffix",
        "k",
        "kernel_backend",
        "graph_grade",
        "warmup_policy",
        "measurement_currency",
    }
    _require(
        set(controls["matched_fields"]) == required_match,
        "P4 matched-control fields are incomplete",
    )


def _validate_resource_gate(
    entry: Mapping[str, Any],
    environment: Mapping[str, Any],
    workload: Mapping[str, Any],
    derived_boot: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = _load_artifact(entry["resource_gate"]["candidate"])
    validate_candidate(candidate)
    _require(candidate["capability_class"] == "B0", "P4 resource class must be B0")
    _require(
        candidate["fixed_environment_id"] == derived_boot["fixed_environment_id"],
        "P4 resource candidate belongs to another environment",
    )
    _require(
        candidate["resident_objects"]["graph_ids"]
        == derived_boot["resident_graph_ids"],
        "masked P4 candidate must reuse the P3 resident graph set",
    )
    _require(
        candidate["evidence"]["grade"] == "static_proxy_projection",
        "entry package cannot relabel the pre-window capacity as exact",
    )
    _require(
        not candidate["evidence"]["candidate_realization_match"],
        "pre-window capacity does not match the runtime-window realization",
    )
    relation = candidate["evidence"]["capacity"]["measurement_relation"]
    _require(
        relation == "optimistic_proxy_ceiling",
        "P4 entry resource evidence must remain an optimistic proxy",
    )

    result = plan_boot_classes(environment, workload, [candidate])
    decision = result["decisions"][0]
    codes = {reason["code"] for reason in decision["reasons"]}
    _require(decision["decision"] == "reject", "P4 entry candidate must fail closed")
    _require(
        "capacity_evidence_not_exact" in codes,
        "P4 proxy must be rejected for missing exact capacity evidence",
    )
    return decision


def validate_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one P4 entry manifest and return its gate summary.

    Args:
        entry: Parsed P4 entry manifest.

    Returns:
        Machine-readable validation and blocking-gate summary.

    Raises:
        P4WindowEntryError: If the package is inconsistent or overclaims.
    """
    _validate_schema(entry)
    base = entry["base_artifacts"]
    base_boot = _load_artifact(base["boot"])
    base_registry = _load_artifact(base["actions"])
    base_snapshot = _load_artifact(base["runtime_snapshot"])
    _artifact_path(base["p3b_result"])
    environment = _load_artifact(base["environment"])
    workload = _load_artifact(base["workload"])

    validate_action_registry(base_boot, base_registry)
    validate_runtime_snapshot(base_boot, base_registry, base_snapshot)
    validate_environment(environment)
    validate_workload(environment, workload)
    _validate_phase96_evidence(entry)

    derived_boot, derived_registry = _validate_window_contract(
        entry,
        base_boot,
        base_registry,
    )
    alias_count = _validate_alias_overlay(
        entry,
        base_snapshot,
        derived_boot,
        derived_registry,
    )
    _validate_controls(entry)
    resource = _validate_resource_gate(
        entry,
        environment,
        workload,
        derived_boot,
    )

    gates = entry["promotion_gates"]
    pending = {name for name, status in gates.items() if status == "pending"}
    not_approved = {name for name, status in gates.items() if status == "not_approved"}
    _require(
        pending == EXPECTED_PENDING_GATES,
        "P4 entry must retain every unproven promotion gate",
    )
    _require(
        not_approved == EXPECTED_NOT_APPROVED_GATES,
        "P4 entry must retain the failed Phase 96/F approval decision",
    )
    return {
        "status": "pass",
        "package_id": entry["package_id"],
        "entry_decision": entry["status"],
        "window_action_id": entry["selected_action"]["action_id"],
        "window_value": entry["selection"]["window_value"],
        "window_grade": entry["selection"]["cost_grade"],
        "cost_credit_allowed": entry["selection"]["cost_credit_allowed"],
        "shared_kv_alias_count": alias_count,
        "true_slot_mapping": "canonical",
        "resource_decision": resource["decision"],
        "resource_reason_codes": sorted(
            reason["code"] for reason in resource["reasons"]
        ),
        "pending_promotion_gates": sorted(pending),
        "not_approved_promotion_gates": sorted(not_approved),
        "performance_claims_allowed": entry["resource_gate"][
            "performance_claims_allowed"
        ],
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Validate a P4 entry file and print a machine-readable result."""
    args = parse_args()
    result = validate_entry(_load_json(args.entry))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
