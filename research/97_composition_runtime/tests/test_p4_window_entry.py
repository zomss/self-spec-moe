"""CPU-only tests for the Phase 97 P4 runtime-window entry package."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR / "scripts"))

from validate_p4_window_entry import (  # noqa: E402
    P4WindowEntryError,
    validate_entry,
)

ENTRY_PATH = PHASE_DIR / "data" / "p4" / "p4_window_entry_w512_masked.json"


def valid_entry() -> dict:
    """Return an independent copy of the checked-in P4 entry package."""
    return json.loads(ENTRY_PATH.read_text())


class P4WindowEntryPositiveTests(unittest.TestCase):
    """Prove the selected action and its intentional blocking state."""

    def test_checked_in_entry_passes_as_not_admitted(self) -> None:
        result = validate_entry(valid_entry())

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["entry_decision"], "registered_not_admitted")
        self.assertEqual(result["window_value"], 512)
        self.assertEqual(result["window_grade"], "acceptance_only")
        self.assertFalse(result["cost_credit_allowed"])
        self.assertFalse(result["performance_claims_allowed"])

    def test_shared_kv_and_resource_summary_is_exact(self) -> None:
        result = validate_entry(valid_entry())

        self.assertEqual(result["shared_kv_alias_count"], 36)
        self.assertEqual(result["true_slot_mapping"], "canonical")
        self.assertEqual(result["resource_decision"], "reject")
        self.assertEqual(
            result["resource_reason_codes"],
            ["capacity_evidence_not_exact"],
        )
        self.assertEqual(len(result["pending_promotion_gates"]), 5)
        self.assertEqual(
            result["not_approved_promotion_gates"],
            ["phase96_f_resident_value"],
        )


class P4WindowEntryFailClosedTests(unittest.TestCase):
    """Reject evidence drift, mechanism inflation, and premature promotion."""

    def test_rejects_artifact_hash_drift(self) -> None:
        entry = valid_entry()
        entry["base_artifacts"]["boot"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(P4WindowEntryError, "hash mismatch"):
            validate_entry(entry)

    def test_rejects_an_unregistered_window_value(self) -> None:
        entry = valid_entry()
        entry["selection"]["window_value"] = 2048
        entry["selected_action"]["window"]["value"] = 2048
        entry["proposed_boot_delta"]["admitted_windows"] = [0, 2048]
        with self.assertRaisesRegex(P4WindowEntryError, "must be w512"):
            validate_entry(entry)

    def test_rejects_cost_credit_for_masked_window(self) -> None:
        entry = valid_entry()
        entry["selection"]["cost_credit_allowed"] = True
        with self.assertRaises(P4WindowEntryError):
            validate_entry(entry)

    def test_rejects_cost_true_relabeling(self) -> None:
        entry = valid_entry()
        entry["selection"]["mode"] = "cost_true"
        entry["selection"]["cost_grade"] = "cost_true"
        with self.assertRaises(P4WindowEntryError):
            validate_entry(entry)

    def test_rejects_false_exact_action_transfer(self) -> None:
        entry = valid_entry()
        entry["selection"]["phase96_exact_action_match"] = True
        with self.assertRaisesRegex(P4WindowEntryError, "did not measure"):
            validate_entry(entry)

    def test_rejects_a_separate_masked_graph(self) -> None:
        entry = valid_entry()
        descriptor = entry["selected_action"]["draft_graph_descriptor"]
        descriptor["graph_id"] = "draft-target-matching-w512-k1"
        with self.assertRaisesRegex(P4WindowEntryError, "reuse K4 field"):
            validate_entry(entry)

    def test_rejects_new_resident_graphs(self) -> None:
        entry = valid_entry()
        entry["proposed_boot_delta"]["new_resident_graph_ids"] = [
            "draft-target-matching-w512-k1"
        ]
        with self.assertRaises(P4WindowEntryError):
            validate_entry(entry)

    def test_rejects_true_slot_mapping_override(self) -> None:
        entry = valid_entry()
        entry["action_view"]["true_slot_mapping_id"] = "window-slots"
        with self.assertRaisesRegex(P4WindowEntryError, "true-slot"):
            validate_entry(entry)

    def test_rejects_incomplete_matched_controls(self) -> None:
        entry = valid_entry()
        entry["controls"]["matched_fields"].remove("measurement_currency")
        with self.assertRaisesRegex(P4WindowEntryError, "incomplete"):
            validate_entry(entry)

    def test_rejects_wrong_proper_subset(self) -> None:
        entry = valid_entry()
        entry["controls"]["proper_subset_action_ids"] = ["off"]
        with self.assertRaisesRegex(P4WindowEntryError, "proper subset"):
            validate_entry(entry)

    def test_rejects_p3b_policy_regression(self) -> None:
        entry = valid_entry()
        entry["boundary_policy"]["mixed_prefill_decode"] = "verify_pending_draft"
        with self.assertRaises(P4WindowEntryError):
            validate_entry(entry)

    def test_rejects_unmeasured_transition_as_measured(self) -> None:
        entry = valid_entry()
        entry["selected_action"]["transition_cost"] = {
            "status": "measured",
            "value_us": 0.0,
        }
        with self.assertRaises(P4WindowEntryError):
            validate_entry(entry)

    def test_rejects_pre_window_capacity_as_exact(self) -> None:
        entry = valid_entry()
        entry["resource_gate"]["candidate"] = {
            "path": (
                "research/97_composition_runtime/data/preflight/"
                "candidate_b0_measured.json"
            ),
            "sha256": (
                "16b842c567b70a2f23b0a0db230d2c2f98c6bab7af2f63d2cf2c0a3569121ac7"
            ),
        }
        with self.assertRaisesRegex(P4WindowEntryError, "cannot relabel"):
            validate_entry(entry)

    def test_rejects_removed_pending_gate(self) -> None:
        entry = copy.deepcopy(valid_entry())
        entry["promotion_gates"]["transition_overhead"] = "pass"
        with self.assertRaises(P4WindowEntryError):
            validate_entry(entry)

    def test_rejects_reopening_failed_f_gate_as_pending(self) -> None:
        entry = copy.deepcopy(valid_entry())
        entry["promotion_gates"]["phase96_f_resident_value"] = "pending"
        with self.assertRaises(P4WindowEntryError):
            validate_entry(entry)


if __name__ == "__main__":
    unittest.main()
