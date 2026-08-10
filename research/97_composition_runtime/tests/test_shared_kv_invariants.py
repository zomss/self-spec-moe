"""CPU-only tests for the Phase 97 shared-target-KV contract."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR / "scripts"))

from validate_shared_kv import (  # noqa: E402
    SharedKVInvariantError,
    load_schema,
    validate_action_registry,
    validate_boot_manifest,
    validate_runtime_objects,
    validate_runtime_snapshot,
)

SPEC = {
    "dtype": "bfloat16",
    "block_size": 16,
    "layout": "NHD",
    "spec_fingerprint": "qwen3-bf16-kv-v1",
}


def valid_boot() -> dict:
    """Return a valid B1 manifest with one target and one quantized path."""
    gib = 1024**3
    return {
        "schema_version": 1,
        "boot_class_id": "qwen3-8b-b1",
        "capability_class": "B1",
        "fixed_environment_id": "qwen3-8b-tp1-h100",
        "environment": {
            "VLLM_SELF_SPEC_SHARED_KV": "1",
            "VLLM_SELF_SPEC_DRAFT_KV_DTYPE": "",
            "VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE": "1",
        },
        "kv": {
            "path": "shared_target",
            "owner": "target",
            "binding_id": "qwen3-layer-twins-v1",
            "pool_count": 1,
            "cache_spec": copy.deepcopy(SPEC),
            "expected_layer_bindings": [
                {
                    "draft_layer": "draft_model.model.layers.0.self_attn",
                    "target_layer": "model.layers.0.self_attn",
                },
                {
                    "draft_layer": "draft_model.model.layers.1.self_attn",
                    "target_layer": "model.layers.1.self_attn",
                },
            ],
        },
        "draft_weight_paths": [
            {
                "path_id": "target-matching",
                "kind": "target_matching",
                "quantization": None,
                "version_policy": "target_alias",
                "resident_bytes": 0,
                "refresh_peak_bytes": 0,
            },
            {
                "path_id": "w4a8-humming",
                "kind": "quantized",
                "quantization": {
                    "format": "w4a8",
                    "kernel": "humming",
                    "source": "/data/checkpoints/qwen3-8b-w4a8",
                },
                "version_policy": "atomically_published",
                "resident_bytes": 4 * gib,
                "refresh_peak_bytes": gib,
            },
        ],
        "admitted_windows": [0, 512, 2048],
        "admitted_skip_sets": [[], [2, 8]],
        "max_k": 4,
        "resident_graph_ids": [
            "target-k1",
            "target-k5",
            "draft-target-matching",
            "draft-w4a8-win512-skip2-8",
        ],
        "resources": {
            "usable_hbm_bytes": 80 * gib,
            "safety_bytes": 2 * gib,
            "target_fixed_bytes": 16 * gib,
            "target_matching_draft_extra_bytes": 0,
            "quantized_draft_weights_bytes": 4 * gib,
            "weight_refresh_peak_bytes": gib,
            "graphs_and_workspaces_bytes": 2 * gib,
            "shared_target_kv_blocks": 20000,
            "kv_block_size_tokens": 16,
        },
    }


def valid_registry() -> dict:
    """Return a valid registry containing OFF, baseline, and a triple."""
    action_ids = ["off", "baseline-k4", "w4a8-win512-skip2-8-k4"]
    return {
        "schema_version": 1,
        "boot_class_id": "qwen3-8b-b1",
        "actions": [
            {
                "action_id": "off",
                "kind": "off",
                "target_graph_descriptor": {
                    "graph_id": "target-k1",
                    "grade": "piecewise",
                    "realization": "target-decode",
                    "query_width": 1,
                },
                "legal_switch_predecessors": action_ids,
                "measured_transition_cost_us": 0,
            },
            {
                "action_id": "baseline-k4",
                "kind": "speculative",
                "draft_weight_path_id": "target-matching",
                "realization": "piecewise-shared",
                "window": {
                    "mode": "off",
                    "value": 0,
                    "cost_grade": "none",
                },
                "skip_set": [],
                "k": 4,
                "target_graph_descriptor": {
                    "graph_id": "target-k5",
                    "grade": "piecewise",
                    "realization": "target-verify",
                    "query_width": 5,
                },
                "draft_graph_descriptor": {
                    "graph_id": "draft-target-matching",
                    "grade": "piecewise",
                    "realization": "target-matching",
                    "query_width": 1,
                },
                "requires_current_draft_weight": False,
                "legal_switch_predecessors": action_ids,
                "measured_transition_cost_us": 0,
            },
            {
                "action_id": "w4a8-win512-skip2-8-k4",
                "kind": "speculative",
                "draft_weight_path_id": "w4a8-humming",
                "realization": "fullcg-shared",
                "window": {
                    "mode": "cost_true",
                    "value": 512,
                    "cost_grade": "cost_true",
                },
                "skip_set": [2, 8],
                "k": 4,
                "target_graph_descriptor": {
                    "graph_id": "target-k5",
                    "grade": "full",
                    "realization": "target-verify",
                    "query_width": 5,
                },
                "draft_graph_descriptor": {
                    "graph_id": "draft-w4a8-win512-skip2-8",
                    "grade": "full",
                    "realization": "w4a8-win512-skip2-8",
                    "query_width": 1,
                },
                "requires_current_draft_weight": True,
                "legal_switch_predecessors": action_ids,
                "measured_transition_cost_us": 0,
            },
        ],
    }


def valid_b0_boot() -> dict:
    """Return a valid target-matching-only B0 manifest."""
    boot = valid_boot()
    boot["boot_class_id"] = "qwen3-8b-b0"
    boot["capability_class"] = "B0"
    boot["draft_weight_paths"] = [boot["draft_weight_paths"][0]]
    boot["resident_graph_ids"] = [
        "target-k1",
        "target-k5",
        "draft-target-matching",
    ]
    boot["resources"]["quantized_draft_weights_bytes"] = 0
    boot["resources"]["weight_refresh_peak_bytes"] = 0
    return boot


def valid_snapshot() -> dict:
    """Return valid serialized cache-binding and action-view evidence."""
    return {
        "schema_version": 1,
        "boot_class_id": "qwen3-8b-b1",
        "binding_id": "qwen3-layer-twins-v1",
        "target_pool_id": "target-kv-pool",
        "allocated_pool_ids": ["target-kv-pool"],
        "canonical_true_slot_mapping_id": "slot-mapping-v1",
        "layer_bindings": [
            {
                "draft_layer": "draft_model.model.layers.0.self_attn",
                "target_layer": "model.layers.0.self_attn",
                "draft_cache_spec": copy.deepcopy(SPEC),
                "target_cache_spec": copy.deepcopy(SPEC),
                "same_storage": True,
                "pool_id": "target-kv-pool",
            },
            {
                "draft_layer": "draft_model.model.layers.1.self_attn",
                "target_layer": "model.layers.1.self_attn",
                "draft_cache_spec": copy.deepcopy(SPEC),
                "target_cache_spec": copy.deepcopy(SPEC),
                "same_storage": True,
                "pool_id": "target-kv-pool",
            },
        ],
        "action_views": [
            {
                "action_id": "baseline-k4",
                "binding_id": "qwen3-layer-twins-v1",
                "pool_id": "target-kv-pool",
                "true_slot_mapping_id": "slot-mapping-v1",
            },
            {
                "action_id": "w4a8-win512-skip2-8-k4",
                "binding_id": "qwen3-layer-twins-v1",
                "pool_id": "target-kv-pool",
                "true_slot_mapping_id": "slot-mapping-v1",
            },
        ],
    }


class SharedKVSchemaTests(unittest.TestCase):
    """Validate schema documents and positive manifests."""

    def test_schema_documents_are_valid(self) -> None:
        for name in (
            "boot_class.schema.json",
            "action.schema.json",
            "shared_kv_runtime.schema.json",
        ):
            with self.subTest(schema=name):
                self.assertEqual(load_schema(name)["$schema"].split("/")[-1], "schema")

    def test_valid_package_passes(self) -> None:
        boot = valid_boot()
        registry = valid_registry()
        snapshot = valid_snapshot()
        validate_boot_manifest(boot)
        validate_action_registry(boot, registry)
        validate_runtime_snapshot(boot, registry, snapshot)

    def test_valid_b0_boot_passes(self) -> None:
        validate_boot_manifest(valid_b0_boot())


class SharedKVFailClosedTests(unittest.TestCase):
    """Every private, mismatched, or action-selectable KV state must fail."""

    def test_boot_rejects_disabled_sharing(self) -> None:
        boot = valid_boot()
        boot["environment"]["VLLM_SELF_SPEC_SHARED_KV"] = "0"
        with self.assertRaises(SharedKVInvariantError):
            validate_boot_manifest(boot)

    def test_boot_rejects_draft_kv_dtype(self) -> None:
        boot = valid_boot()
        boot["environment"]["VLLM_SELF_SPEC_DRAFT_KV_DTYPE"] = "fp8"
        with self.assertRaises(SharedKVInvariantError):
            validate_boot_manifest(boot)

    def test_boot_rejects_private_path(self) -> None:
        boot = valid_boot()
        boot["kv"]["path"] = "private_fp8"
        with self.assertRaises(SharedKVInvariantError):
            validate_boot_manifest(boot)

    def test_boot_rejects_second_pool(self) -> None:
        boot = valid_boot()
        boot["kv"]["pool_count"] = 2
        with self.assertRaises(SharedKVInvariantError):
            validate_boot_manifest(boot)

    def test_boot_rejects_missing_target_matching_recourse(self) -> None:
        boot = valid_boot()
        boot["draft_weight_paths"] = [boot["draft_weight_paths"][1]]
        with self.assertRaises(SharedKVInvariantError):
            validate_boot_manifest(boot)

    def test_action_rejects_kv_override(self) -> None:
        boot = valid_boot()
        registry = valid_registry()
        registry["actions"][1]["draft_kv_path_id"] = "private-fp8"
        with self.assertRaises(SharedKVInvariantError):
            validate_action_registry(boot, registry)

    def test_action_rejects_unknown_weight_path(self) -> None:
        boot = valid_boot()
        registry = valid_registry()
        registry["actions"][1]["draft_weight_path_id"] = "unknown"
        with self.assertRaises(SharedKVInvariantError):
            validate_action_registry(boot, registry)

    def test_action_rejects_wrong_weight_version_policy(self) -> None:
        boot = valid_boot()
        registry = valid_registry()
        registry["actions"][2]["requires_current_draft_weight"] = False
        with self.assertRaises(SharedKVInvariantError):
            validate_action_registry(boot, registry)

    def test_runtime_rejects_cache_spec_mismatch(self) -> None:
        snapshot = valid_snapshot()
        snapshot["layer_bindings"][0]["draft_cache_spec"]["dtype"] = "fp8"
        with self.assertRaises(SharedKVInvariantError):
            validate_runtime_snapshot(valid_boot(), valid_registry(), snapshot)

    def test_runtime_rejects_missing_union_layer(self) -> None:
        snapshot = valid_snapshot()
        snapshot["layer_bindings"].pop()
        with self.assertRaises(SharedKVInvariantError):
            validate_runtime_snapshot(valid_boot(), valid_registry(), snapshot)

    def test_runtime_rejects_private_pool(self) -> None:
        snapshot = valid_snapshot()
        snapshot["allocated_pool_ids"].append("draft-kv-pool")
        with self.assertRaises(SharedKVInvariantError):
            validate_runtime_snapshot(valid_boot(), valid_registry(), snapshot)

    def test_runtime_rejects_action_binding_override(self) -> None:
        snapshot = valid_snapshot()
        snapshot["action_views"][1]["binding_id"] = "other-binding"
        with self.assertRaises(SharedKVInvariantError):
            validate_runtime_snapshot(valid_boot(), valid_registry(), snapshot)

    def test_runtime_rejects_action_slot_mapping_override(self) -> None:
        snapshot = valid_snapshot()
        snapshot["action_views"][1]["true_slot_mapping_id"] = "other-slots"
        with self.assertRaises(SharedKVInvariantError):
            validate_runtime_snapshot(valid_boot(), valid_registry(), snapshot)

    def test_runtime_rejects_nonaliased_tensor(self) -> None:
        target_0 = object()
        target_1 = object()
        target_tensors = {
            "model.layers.0.self_attn": target_0,
            "model.layers.1.self_attn": target_1,
        }
        draft_tensors = {
            "draft_model.model.layers.0.self_attn": object(),
            "draft_model.model.layers.1.self_attn": target_1,
        }
        canonical_slots = object()
        action_slots = {
            "baseline-k4": canonical_slots,
            "w4a8-win512-skip2-8-k4": canonical_slots,
        }
        with self.assertRaises(SharedKVInvariantError):
            validate_runtime_objects(
                valid_boot(),
                valid_registry(),
                valid_snapshot(),
                target_tensors=target_tensors,
                draft_tensors=draft_tensors,
                canonical_slot_mapping=canonical_slots,
                action_slot_mappings=action_slots,
            )

    def test_runtime_rejects_noncanonical_slot_object(self) -> None:
        target_0 = object()
        target_1 = object()
        target_tensors = {
            "model.layers.0.self_attn": target_0,
            "model.layers.1.self_attn": target_1,
        }
        draft_tensors = {
            "draft_model.model.layers.0.self_attn": target_0,
            "draft_model.model.layers.1.self_attn": target_1,
        }
        canonical_slots = object()
        action_slots = {
            "baseline-k4": canonical_slots,
            "w4a8-win512-skip2-8-k4": object(),
        }
        with self.assertRaises(SharedKVInvariantError):
            validate_runtime_objects(
                valid_boot(),
                valid_registry(),
                valid_snapshot(),
                target_tensors=target_tensors,
                draft_tensors=draft_tensors,
                canonical_slot_mapping=canonical_slots,
                action_slot_mappings=action_slots,
            )

    def test_runtime_objects_accept_exact_aliases(self) -> None:
        target_0 = object()
        target_1 = object()
        target_tensors = {
            "model.layers.0.self_attn": target_0,
            "model.layers.1.self_attn": target_1,
        }
        draft_tensors = {
            "draft_model.model.layers.0.self_attn": target_0,
            "draft_model.model.layers.1.self_attn": target_1,
        }
        canonical_slots = object()
        action_slots = {
            "baseline-k4": canonical_slots,
            "w4a8-win512-skip2-8-k4": canonical_slots,
        }
        validate_runtime_objects(
            valid_boot(),
            valid_registry(),
            valid_snapshot(),
            target_tensors=target_tensors,
            draft_tensors=draft_tensors,
            canonical_slot_mapping=canonical_slots,
            action_slot_mappings=action_slots,
        )


if __name__ == "__main__":
    unittest.main()
