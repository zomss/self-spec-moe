"""CPU-only tests for the Phase 97 conservative B0 resource bound."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR / "scripts"))

from validate_p4_b0_resource_bound import (  # noqa: E402
    B0ResourceBoundError,
    _window_buffer_breakdown,
    validate_bound,
)

BOUND_PATH = PHASE_DIR / "data" / "p4" / "p4_b0_conservative_resource_bound.json"
SCHEMA_PATH = PHASE_DIR / "schemas" / "p4_b0_resource_bound.schema.json"


def valid_bound() -> dict:
    """Return an independent copy of the checked-in resource bound."""
    return json.loads(BOUND_PATH.read_text(encoding="utf-8"))


class P4B0ResourceBoundPositiveTests(unittest.TestCase):
    """Verify the conservative arithmetic and narrow claim boundary."""

    def test_schema_document_is_valid(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)

    def test_checked_in_bound_passes_and_clears_only_resource_readiness(self) -> None:
        result = validate_bound(valid_bound())

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["resource_decision"], "pass")
        self.assertEqual(result["base_available_shared_kv_blocks"], 24529)
        self.assertEqual(result["required_live_kv_blocks"], 21000)
        self.assertEqual(result["reserved_blocks_total"], 2847)
        self.assertEqual(result["lower_bound_shared_kv_blocks"], 21682)
        self.assertEqual(result["lower_bound_headroom_blocks"], 682)
        self.assertEqual(result["lower_bound_headroom_tokens"], 10912)
        self.assertEqual(result["remaining_readiness_blockers"], [])
        self.assertFalse(result["gpu_measurement_authorized"])
        self.assertFalse(result["p4a_engineering_authorized"])
        self.assertFalse(result["action_admitted"])

    def test_window_buffer_formula_uses_eight_byte_upper_bounds(self) -> None:
        breakdown = _window_buffer_breakdown(rows=32, columns=1280)

        self.assertEqual(
            breakdown,
            {
                "window_block_table": 327680,
                "window_sequence_lengths": 256,
                "column_arange": 10240,
                "source_column_indices": 327680,
                "sink_column_mask": 10240,
            },
        )
        self.assertEqual(sum(breakdown.values()), 676096)

    def test_each_reserve_is_rounded_to_whole_kv_blocks(self) -> None:
        bound = valid_bound()
        reserves = bound["reserve_policy"]

        self.assertEqual(reserves["persistent_window_buffers"]["reserved_blocks"], 1)
        self.assertEqual(
            reserves["graph_capture_and_temporary_workspace"]["reserved_blocks"],
            911,
        )
        self.assertEqual(
            reserves["recorder_and_runtime_metadata"]["reserved_blocks"], 114
        )
        self.assertEqual(reserves["hbm_fragmentation_safety"]["reserved_blocks"], 1821)

    def test_bound_does_not_relabel_projection_or_grant_authority(self) -> None:
        bound = valid_bound()

        self.assertFalse(bound["candidate"]["candidate_realization_match"])
        self.assertFalse(bound["candidate"]["exact_post_capture_measurement"])
        self.assertFalse(bound["base_capacity"]["older_projection_relabelled_exact"])
        self.assertFalse(any(bound["authorizations"].values()))
        self.assertEqual(
            bound["fail_closed"]["admission_still_requires"],
            [
                "measured_exact_post_capture_candidate",
                "exact_candidate_measurement_relation",
            ],
        )


class P4B0ResourceBoundFailClosedTests(unittest.TestCase):
    """Reject source drift, understated reserves, and authority inflation."""

    def test_rejects_upstream_artifact_hash_drift(self) -> None:
        bound = valid_bound()
        bound["source_artifacts"]["measured_b0"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(B0ResourceBoundError, "artifact hash mismatch"):
            validate_bound(bound)

    def test_rejects_geometry_drift(self) -> None:
        bound = valid_bound()
        bound["candidate"]["geometry"]["max_num_seqs"] = 16
        with self.assertRaisesRegex(B0ResourceBoundError, "geometry"):
            validate_bound(bound)

    def test_rejects_understated_window_buffer(self) -> None:
        bound = valid_bound()
        reserve = bound["reserve_policy"]["persistent_window_buffers"]
        reserve["derived_bytes_upper_bound"] -= 1
        with self.assertRaisesRegex(B0ResourceBoundError, "window-buffer reserve"):
            validate_bound(bound)

    def test_rejects_missing_window_buffer(self) -> None:
        bound = valid_bound()
        bound["reserve_policy"]["persistent_window_buffers"]["buffers"].pop()
        with self.assertRaises(B0ResourceBoundError):
            validate_bound(bound)

    def test_rejects_understated_graph_workspace_reserve(self) -> None:
        bound = valid_bound()
        reserve = bound["reserve_policy"]["graph_capture_and_temporary_workspace"]
        reserve["reserved_bytes"] = 1024**3
        with self.assertRaisesRegex(B0ResourceBoundError, "graph/workspace"):
            validate_bound(bound)

    def test_rejects_understated_metadata_reserve(self) -> None:
        bound = valid_bound()
        reserve = bound["reserve_policy"]["recorder_and_runtime_metadata"]
        reserve["reserved_bytes"] -= 1
        with self.assertRaisesRegex(B0ResourceBoundError, "metadata reserve"):
            validate_bound(bound)

    def test_rejects_understated_hbm_safety_reserve(self) -> None:
        bound = valid_bound()
        reserve = bound["reserve_policy"]["hbm_fragmentation_safety"]
        reserve["reserved_bytes"] -= 1
        with self.assertRaisesRegex(B0ResourceBoundError, "HBM/fragmentation"):
            validate_bound(bound)

    def test_rejects_block_rounding_credit(self) -> None:
        bound = valid_bound()
        reserve = bound["reserve_policy"]["graph_capture_and_temporary_workspace"]
        reserve["reserved_blocks"] -= 1
        with self.assertRaisesRegex(B0ResourceBoundError, "independently rounded"):
            validate_bound(bound)

    def test_rejects_result_arithmetic_drift(self) -> None:
        bound = valid_bound()
        bound["result"]["lower_bound_shared_kv_blocks"] += 1
        with self.assertRaisesRegex(B0ResourceBoundError, "arithmetic"):
            validate_bound(bound)

    def test_rejects_a_missing_fail_closed_invalidator(self) -> None:
        bound = valid_bound()
        bound["fail_closed"]["bound_invalidators"].pop()
        with self.assertRaisesRegex(B0ResourceBoundError, "fail-closed invalidator"):
            validate_bound(bound)

    def test_rejects_exact_measurement_claim(self) -> None:
        bound = copy.deepcopy(valid_bound())
        bound["claims"]["exact_candidate_measured"] = True
        with self.assertRaises(B0ResourceBoundError):
            validate_bound(bound)

    def test_rejects_gpu_authorization(self) -> None:
        bound = copy.deepcopy(valid_bound())
        bound["authorizations"]["gpu_measurement"] = True
        with self.assertRaises(B0ResourceBoundError):
            validate_bound(bound)

    def test_rejects_self_authorizing_handoff(self) -> None:
        bound = valid_bound()
        bound["next_artifact"]["separate_decision_required"] = False
        with self.assertRaises(B0ResourceBoundError):
            validate_bound(bound)


if __name__ == "__main__":
    unittest.main()
