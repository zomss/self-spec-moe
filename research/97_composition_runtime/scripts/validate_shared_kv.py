#!/usr/bin/env python3
"""Validate Phase 97 shared-target-KV manifests and runtime evidence."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

PHASE_DIR = Path(__file__).resolve().parents[1]
SCHEMA_DIR = PHASE_DIR / "schemas"

_FORBIDDEN_KEY_FRAGMENTS = (
    "draftkv",
    "privatedraftkv",
    "privatekv",
    "kvmirror",
    "kvreadiness",
    "kvbackfill",
    "kvconversion",
    "kvstaging",
)
_FORBIDDEN_EXACT_KEYS = frozenset({"kvpath"})
_DRAFT_KV_DTYPE_ENV = "VLLM_SELF_SPEC_DRAFT_KV_DTYPE"


class SharedKVInvariantError(ValueError):
    """Raised when a Phase 97 manifest violates the shared-KV contract."""


def load_schema(name: str) -> dict[str, Any]:
    """Load and meta-validate one phase-local JSON schema.

    Args:
        name: Schema filename under ``schemas/``.

    Returns:
        Parsed schema.

    Raises:
        SharedKVInvariantError: If the file is missing or is not a valid
            Draft 2020-12 schema.
    """
    path = SCHEMA_DIR / name
    if not path.is_file():
        raise SharedKVInvariantError(f"missing schema: {path}")
    schema = json.loads(path.read_text())
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise SharedKVInvariantError(f"invalid schema {name}: {exc}") from exc
    return schema


def _format_json_path(parts: list[Any]) -> str:
    path = "$"
    for part in parts:
        if isinstance(part, int):
            path += f"[{part}]"
        else:
            path += f".{part}"
    return path


def _validate_schema(instance: Mapping[str, Any], name: str) -> None:
    schema = load_schema(name)
    validator = Draft202012Validator(schema)
    errors = sorted(
        validator.iter_errors(instance),
        key=lambda error: [str(part) for part in error.absolute_path],
    )
    if not errors:
        return
    first = errors[0]
    path = _format_json_path(list(first.absolute_path))
    raise SharedKVInvariantError(f"{name} rejected {path}: {first.message}")


def _normalized_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


def _reject_forbidden_fields(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            normalized = _normalized_key(key_text)
            child_path = f"{path}.{key_text}"
            if normalized == _normalized_key(_DRAFT_KV_DTYPE_ENV):
                allowed = path == "$.environment" and child == ""
                if not allowed:
                    raise SharedKVInvariantError(
                        f"draft-only KV dtype is forbidden at {child_path}"
                    )
            elif normalized in _FORBIDDEN_EXACT_KEYS or any(
                fragment in normalized for fragment in _FORBIDDEN_KEY_FRAGMENTS
            ):
                raise SharedKVInvariantError(
                    f"private or selectable KV field is forbidden at {child_path}"
                )
            _reject_forbidden_fields(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_fields(child, f"{path}[{index}]")


def _require_unique(values: list[str], label: str) -> None:
    if len(values) != len(set(values)):
        raise SharedKVInvariantError(f"{label} must be unique")


def validate_boot_manifest(boot: Mapping[str, Any]) -> None:
    """Validate the shared-KV boot manifest and B0/B1 semantics.

    Args:
        boot: Parsed boot-class manifest.

    Raises:
        SharedKVInvariantError: If the manifest is invalid or admits a
            private/selectable KV path.
    """
    _reject_forbidden_fields(boot)
    _validate_schema(boot, "boot_class.schema.json")

    paths = list(boot["draft_weight_paths"])
    path_ids = [str(path["path_id"]) for path in paths]
    _require_unique(path_ids, "draft weight path ids")

    bindings = list(boot["kv"]["expected_layer_bindings"])
    draft_layers = [str(binding["draft_layer"]) for binding in bindings]
    target_layers = [str(binding["target_layer"]) for binding in bindings]
    _require_unique(draft_layers, "draft layer bindings")
    _require_unique(target_layers, "target layer bindings")

    for skip_set in boot["admitted_skip_sets"]:
        if list(skip_set) != sorted(skip_set):
            raise SharedKVInvariantError("admitted skip sets must be sorted")

    target_paths = [path for path in paths if path["kind"] == "target_matching"]
    quant_paths = [path for path in paths if path["kind"] == "quantized"]
    capability = boot["capability_class"]
    if capability == "B0" and quant_paths:
        raise SharedKVInvariantError("B0 cannot contain quantized weights")
    if capability == "B1" and len(quant_paths) != 1:
        raise SharedKVInvariantError("B1 requires exactly one quantized path")
    if len(target_paths) != 1:
        raise SharedKVInvariantError(
            "every boot requires exactly one target-matching path"
        )

    resources = boot["resources"]
    target_resident = int(target_paths[0]["resident_bytes"])
    quant_resident = sum(int(path["resident_bytes"]) for path in quant_paths)
    refresh_peak = sum(int(path["refresh_peak_bytes"]) for path in quant_paths)
    if resources["target_matching_draft_extra_bytes"] != target_resident:
        raise SharedKVInvariantError(
            "target-matching resident bytes disagree with resource ledger"
        )
    if resources["quantized_draft_weights_bytes"] != quant_resident:
        raise SharedKVInvariantError(
            "quantized resident bytes disagree with resource ledger"
        )
    if resources["weight_refresh_peak_bytes"] != refresh_peak:
        raise SharedKVInvariantError(
            "weight refresh peak disagrees with resource ledger"
        )


def validate_action_registry(
    boot: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> None:
    """Validate an action registry against its shared-KV boot manifest.

    Args:
        boot: Parsed boot-class manifest.
        registry: Parsed action registry.

    Raises:
        SharedKVInvariantError: If an action escapes the boot manifest or
            attempts to select a KV path.
    """
    validate_boot_manifest(boot)
    _reject_forbidden_fields(registry)
    _validate_schema(registry, "action.schema.json")

    if registry["boot_class_id"] != boot["boot_class_id"]:
        raise SharedKVInvariantError("action registry boot_class_id mismatch")

    actions = list(registry["actions"])
    action_ids = [str(action["action_id"]) for action in actions]
    _require_unique(action_ids, "action ids")
    action_id_set = set(action_ids)

    paths = {str(path["path_id"]): path for path in boot["draft_weight_paths"]}
    graph_ids = set(boot["resident_graph_ids"])
    windows = set(boot["admitted_windows"])
    skip_sets = {tuple(skip_set) for skip_set in boot["admitted_skip_sets"]}
    target_matching_seen = False

    for action in actions:
        predecessors = set(action["legal_switch_predecessors"])
        unknown_predecessors = predecessors - action_id_set
        if unknown_predecessors:
            raise SharedKVInvariantError(
                f"action {action['action_id']} has unknown predecessors: "
                f"{sorted(unknown_predecessors)}"
            )
        if action["kind"] == "off":
            graph_id = action["target_graph_descriptor"]["graph_id"]
            if graph_id not in graph_ids:
                raise SharedKVInvariantError(
                    f"OFF action uses unregistered graph {graph_id}"
                )
            continue

        path_id = str(action["draft_weight_path_id"])
        if path_id not in paths:
            raise SharedKVInvariantError(
                f"action {action['action_id']} uses unknown weight path {path_id}"
            )
        path = paths[path_id]
        needs_current = bool(action["requires_current_draft_weight"])
        expected_current = path["kind"] == "quantized"
        if needs_current != expected_current:
            raise SharedKVInvariantError(
                f"action {action['action_id']} has wrong weight-version policy"
            )
        target_matching_seen |= path["kind"] == "target_matching"

        if action["k"] > boot["max_k"]:
            raise SharedKVInvariantError(
                f"action {action['action_id']} exceeds boot max_k"
            )
        if action["window"]["value"] not in windows:
            raise SharedKVInvariantError(
                f"action {action['action_id']} uses an unadmitted window"
            )
        skip_set = list(action["skip_set"])
        if skip_set != sorted(skip_set):
            raise SharedKVInvariantError(
                f"action {action['action_id']} skip set must be sorted"
            )
        if tuple(skip_set) not in skip_sets:
            raise SharedKVInvariantError(
                f"action {action['action_id']} uses an unadmitted skip set"
            )

        for descriptor_name in (
            "target_graph_descriptor",
            "draft_graph_descriptor",
        ):
            graph_id = action[descriptor_name]["graph_id"]
            if graph_id not in graph_ids:
                raise SharedKVInvariantError(
                    f"action {action['action_id']} uses unregistered graph {graph_id}"
                )

    if not target_matching_seen:
        raise SharedKVInvariantError(
            "registry requires target-matching recourse in addition to OFF"
        )


def validate_runtime_snapshot(
    boot: Mapping[str, Any],
    registry: Mapping[str, Any],
    snapshot: Mapping[str, Any],
) -> None:
    """Validate serialized evidence for one shared-KV runtime.

    Args:
        boot: Parsed boot-class manifest.
        registry: Parsed action registry.
        snapshot: Runtime binding and allocation evidence.

    Raises:
        SharedKVInvariantError: If any layer, action, or allocation escapes
            the single target-owned cache.
    """
    validate_action_registry(boot, registry)
    _reject_forbidden_fields(snapshot)
    _validate_schema(snapshot, "shared_kv_runtime.schema.json")

    if snapshot["boot_class_id"] != boot["boot_class_id"]:
        raise SharedKVInvariantError("runtime boot_class_id mismatch")
    if snapshot["binding_id"] != boot["kv"]["binding_id"]:
        raise SharedKVInvariantError("runtime shared-KV binding mismatch")

    target_pool = snapshot["target_pool_id"]
    if snapshot["allocated_pool_ids"] != [target_pool]:
        raise SharedKVInvariantError("runtime must allocate exactly the target KV pool")

    expected_pairs = {
        (binding["draft_layer"], binding["target_layer"])
        for binding in boot["kv"]["expected_layer_bindings"]
    }
    rows = list(snapshot["layer_bindings"])
    actual_pairs = {
        (binding["draft_layer"], binding["target_layer"]) for binding in rows
    }
    if len(actual_pairs) != len(rows):
        raise SharedKVInvariantError("runtime layer bindings must be unique")
    if actual_pairs != expected_pairs:
        raise SharedKVInvariantError(
            "runtime layer bindings do not match the boot alias union"
        )

    expected_spec = boot["kv"]["cache_spec"]
    for binding in rows:
        if binding["target_cache_spec"] != expected_spec:
            raise SharedKVInvariantError(
                f"target cache spec mismatch for {binding['target_layer']}"
            )
        if binding["draft_cache_spec"] != binding["target_cache_spec"]:
            raise SharedKVInvariantError(
                f"draft cache spec mismatch for {binding['draft_layer']}"
            )
        if binding["pool_id"] != target_pool:
            raise SharedKVInvariantError(
                f"layer {binding['draft_layer']} uses a non-target KV pool"
            )

    speculative_ids = {
        action["action_id"]
        for action in registry["actions"]
        if action["kind"] == "speculative"
    }
    views = list(snapshot["action_views"])
    view_ids = [str(view["action_id"]) for view in views]
    _require_unique(view_ids, "runtime action views")
    if set(view_ids) != speculative_ids:
        raise SharedKVInvariantError(
            "runtime action views must cover every speculative action"
        )

    binding_id = snapshot["binding_id"]
    slot_mapping_id = snapshot["canonical_true_slot_mapping_id"]
    for view in views:
        if view["binding_id"] != binding_id:
            raise SharedKVInvariantError(
                f"action {view['action_id']} overrides the KV binding"
            )
        if view["pool_id"] != target_pool:
            raise SharedKVInvariantError(
                f"action {view['action_id']} overrides the target KV pool"
            )
        if view["true_slot_mapping_id"] != slot_mapping_id:
            raise SharedKVInvariantError(
                f"action {view['action_id']} changes true slot mapping"
            )


def _shares_exact_storage(left: Any, right: Any) -> bool:
    if left is right:
        return True
    try:
        left_storage = left.untyped_storage()
        right_storage = right.untyped_storage()
        return (
            left_storage.data_ptr() == right_storage.data_ptr()
            and left_storage.nbytes() == right_storage.nbytes()
            and left.storage_offset() == right.storage_offset()
            and tuple(left.size()) == tuple(right.size())
            and tuple(left.stride()) == tuple(right.stride())
        )
    except (AttributeError, RuntimeError, TypeError):
        return False


def validate_runtime_objects(
    boot: Mapping[str, Any],
    registry: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    *,
    target_tensors: Mapping[str, Any],
    draft_tensors: Mapping[str, Any],
    canonical_slot_mapping: Any,
    action_slot_mappings: Mapping[str, Any],
) -> None:
    """Validate actual cache and slot-mapping object identity.

    Args:
        boot: Parsed boot-class manifest.
        registry: Parsed action registry.
        snapshot: Serialized runtime evidence.
        target_tensors: Target layer name to KV tensor.
        draft_tensors: Draft layer name to KV tensor.
        canonical_slot_mapping: Target-owned true slot mapping object.
        action_slot_mappings: Speculative action id to true slot mapping.

    Raises:
        SharedKVInvariantError: If tensors do not share exact storage or an
            action changes the true slot mapping object.
    """
    validate_runtime_snapshot(boot, registry, snapshot)
    expected_draft_layers = {
        binding["draft_layer"] for binding in boot["kv"]["expected_layer_bindings"]
    }
    if set(draft_tensors) != expected_draft_layers:
        raise SharedKVInvariantError(
            "draft tensor set does not match the boot alias union"
        )

    for binding in boot["kv"]["expected_layer_bindings"]:
        draft_layer = binding["draft_layer"]
        target_layer = binding["target_layer"]
        if target_layer not in target_tensors:
            raise SharedKVInvariantError(f"missing target tensor for {target_layer}")
        if not _shares_exact_storage(
            draft_tensors[draft_layer],
            target_tensors[target_layer],
        ):
            raise SharedKVInvariantError(
                f"{draft_layer} does not alias exact target KV storage"
            )

    speculative_ids = {
        action["action_id"]
        for action in registry["actions"]
        if action["kind"] == "speculative"
    }
    if set(action_slot_mappings) != speculative_ids:
        raise SharedKVInvariantError(
            "action slot mappings must cover every speculative action"
        )
    for action_id, slot_mapping in action_slot_mappings.items():
        if slot_mapping is not canonical_slot_mapping:
            raise SharedKVInvariantError(
                f"action {action_id} changes the true slot mapping object"
            )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boot", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--runtime", type=Path)
    return parser.parse_args()


def main() -> int:
    """Validate JSON artifacts and print a machine-readable result."""
    args = parse_args()
    boot = _load_json(args.boot)
    registry = _load_json(args.actions)
    validate_action_registry(boot, registry)
    result = {
        "status": "pass",
        "boot_class_id": boot["boot_class_id"],
        "action_count": len(registry["actions"]),
        "kv_path": boot["kv"]["path"],
    }
    if args.runtime is not None:
        validate_runtime_snapshot(boot, registry, _load_json(args.runtime))
        result["runtime_snapshot"] = "pass"
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
