"""CPU-only tests for the Phase 97 full-prefill transient-memory bound."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR / "scripts"))

from validate_p4_b0_full_prefill_transient_bound import (  # noqa: E402
    FullPrefillTransientBoundError,
    validate_bound,
)

BOUND_PATH = PHASE_DIR / "data" / "p4" / "p4_b0_full_prefill_transient_bound.json"


def valid_bound() -> dict:
    """Return an independent copy of the checked-in transient bound."""
    return json.loads(BOUND_PATH.read_text(encoding="utf-8"))


class P4B0FullPrefillTransientBoundPositiveTests(unittest.TestCase):
    """Verify the checked failure evidence and conservative arithmetic."""

    def test_checked_in_bound_rejects_only_the_current_geometry(self) -> None:
        result = validate_bound(valid_bound())

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["decision"], "reject_current_full_prefill_geometry")
        self.assertEqual(result["complete_captures_preserved"], 8)
        self.assertFalse(result["score_emitted"])
        self.assertEqual(result["r5_live_set_bytes"], 11_039_932_416)
        self.assertEqual(result["r5cot_live_set_bytes"], 11_099_308_032)
        self.assertEqual(result["optimistic_graph_corrected_capacity_blocks"], 20092)
        self.assertEqual(result["minimum_floor_deficit_bytes"], 235_095_982)
        self.assertFalse(result["gpu_probe_authorized"])
        self.assertFalse(result["v10_authorized"])

    def test_bound_preserves_shared_target_kv_only(self) -> None:
        shared_kv = valid_bound()["measurement_geometry"]["shared_kv"]

        self.assertEqual(shared_kv["owner"], "target")
        self.assertEqual(shared_kv["pool_count"], 1)
        self.assertEqual(shared_kv["private_draft_pool_count"], 0)
        self.assertEqual(shared_kv["layer_count"], 36)

    def test_r5cot_bound_exceeds_failed_r5_bound(self) -> None:
        regimes = valid_bound()["compiled_mlp_live_set"]["regime_bounds"]

        self.assertGreater(
            regimes["R5cot"]["live_set_bytes"], regimes["R5"]["live_set_bytes"]
        )


class P4B0FullPrefillTransientBoundFailClosedTests(unittest.TestCase):
    """Reject source, arithmetic, invariant, and authority drift."""

    def test_rejects_failure_hash_drift(self) -> None:
        bound = valid_bound()
        bound["source_artifacts"]["v9_failure"]["sha256"] = "0" * 64

        with self.assertRaisesRegex(
            FullPrefillTransientBoundError, "artifact hash mismatch"
        ):
            validate_bound(bound)

    def test_rejects_model_geometry_drift(self) -> None:
        bound = valid_bound()
        bound["model_geometry"]["intermediate_size"] = 11008

        with self.assertRaisesRegex(FullPrefillTransientBoundError, "model geometry"):
            validate_bound(bound)

    def test_rejects_missing_live_tensor(self) -> None:
        bound = valid_bound()
        bound["compiled_mlp_live_set"]["tensors"].pop()

        with self.assertRaisesRegex(FullPrefillTransientBoundError, "live tensor set"):
            validate_bound(bound)

    def test_rejects_r5cot_arithmetic_drift(self) -> None:
        bound = valid_bound()
        bound["compiled_mlp_live_set"]["regime_bounds"]["R5cot"]["live_set_bytes"] -= 1

        with self.assertRaisesRegex(
            FullPrefillTransientBoundError, "R5cot live-set bound"
        ):
            validate_bound(bound)

    def test_rejects_graph_pool_correction_drift(self) -> None:
        bound = valid_bound()
        bound["graph_pool_correction"][
            "optimistic_graph_corrected_capacity_blocks"
        ] += 1

        with self.assertRaisesRegex(
            FullPrefillTransientBoundError, "graph-pool correction"
        ):
            validate_bound(bound)

    def test_rejects_floor_deficit_drift(self) -> None:
        bound = valid_bound()
        bound["failed_allocation_at_kv_floor"]["minimum_deficit_bytes"] -= 1

        with self.assertRaisesRegex(
            FullPrefillTransientBoundError, "KV-floor allocation"
        ):
            validate_bound(bound)

    def test_rejects_private_draft_kv(self) -> None:
        bound = valid_bound()
        bound["measurement_geometry"]["shared_kv"]["private_draft_pool_count"] = 1

        with self.assertRaisesRegex(
            FullPrefillTransientBoundError, "shared-KV invariant"
        ):
            validate_bound(bound)

    def test_rejects_gpu_authority(self) -> None:
        bound = copy.deepcopy(valid_bound())
        bound["authorizations"]["gpu_probe"] = True

        with self.assertRaisesRegex(
            FullPrefillTransientBoundError, "must not grant authority"
        ):
            validate_bound(bound)

    def test_rejects_positive_downstream_claim(self) -> None:
        bound = copy.deepcopy(valid_bound())
        bound["claims"]["v10_run_ready"] = True

        with self.assertRaisesRegex(
            FullPrefillTransientBoundError, "positive downstream claim"
        ):
            validate_bound(bound)

    def test_rejects_self_authorizing_v10_handoff(self) -> None:
        bound = valid_bound()
        bound["next_artifact"]["separate_v10_authorization_required"] = False

        with self.assertRaisesRegex(
            FullPrefillTransientBoundError, "next-artifact boundary"
        ):
            validate_bound(bound)


if __name__ == "__main__":
    unittest.main()
