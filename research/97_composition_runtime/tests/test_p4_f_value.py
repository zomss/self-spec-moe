"""CPU-only tests for the Phase 96/F value approval decision."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR / "scripts"))

from validate_p4_f_value import P4FValueError, validate_decision  # noqa: E402

DECISION_PATH = PHASE_DIR / "data" / "p4" / "p4_f_value_decision.json"
VALIDATION_PATH = PHASE_DIR / "data" / "p4" / "p4_f_value_validation.json"


def valid_decision() -> dict:
    """Return an independent copy of the checked-in F decision."""
    return json.loads(DECISION_PATH.read_text())


class P4FValuePositiveTests(unittest.TestCase):
    """Prove that validation passes without confusing it with approval."""

    def test_checked_in_decision_is_not_approved(self) -> None:
        result = validate_decision(valid_decision())

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["approval_state"], "not_approved")
        self.assertEqual(
            result["value_interpretation"],
            "unresolved_not_disproven",
        )
        self.assertFalse(result["p4a_engineering_authorized"])
        self.assertFalse(result["gpu_measurement_authorized"])
        self.assertFalse(result["action_admission_authorized"])

    def test_w14d_audit_and_next_artifact_are_exact(self) -> None:
        result = validate_decision(valid_decision())

        self.assertEqual(result["w14d_audit"]["observed_file_count"], 21)
        self.assertEqual(result["w14d_audit"]["round_count"], 1680)
        self.assertEqual(
            result["w14d_audit"]["negative_unarmed_round_count"],
            313,
        )
        self.assertEqual(result["w14d_audit"]["minimum_unarmed_steps"], -8)
        self.assertEqual(len(result["blocking_reason_codes"]), 10)
        self.assertEqual(
            result["next_artifact"],
            "matched_target_matching_b0_value_screen_preregistration",
        )

    def test_checked_validation_output_is_current(self) -> None:
        expected = json.loads(VALIDATION_PATH.read_text())

        self.assertEqual(validate_decision(valid_decision()), expected)


class P4FValueFailClosedTests(unittest.TestCase):
    """Reject evidence drift, invented values, and downstream authority."""

    def test_rejects_artifact_hash_drift(self) -> None:
        decision = valid_decision()
        decision["source_artifacts"]["p4_entry"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(P4FValueError, "hash mismatch"):
            validate_decision(decision)

    def test_rejects_source_role_substitution(self) -> None:
        decision = valid_decision()
        decision["source_artifacts"]["p4_entry"] = copy.deepcopy(
            decision["source_artifacts"]["workload"]
        )
        with self.assertRaisesRegex(P4FValueError, "unexpected artifact"):
            validate_decision(decision)

    def test_rejects_approval_without_evidence(self) -> None:
        decision = valid_decision()
        decision["decision"]["state"] = "approved"
        with self.assertRaises(P4FValueError):
            validate_decision(decision)

    def test_rejects_treating_unresolved_value_as_disproven(self) -> None:
        decision = valid_decision()
        decision["decision"]["value_interpretation"] = "disproven"
        with self.assertRaises(P4FValueError):
            validate_decision(decision)

    def test_rejects_a_missing_blocker(self) -> None:
        decision = valid_decision()
        decision["decision"]["blocking_reason_codes"].remove("phase96_e_result_missing")
        with self.assertRaisesRegex(P4FValueError, "every observed blocking"):
            validate_decision(decision)

    def test_rejects_p4a_engineering_authority(self) -> None:
        decision = valid_decision()
        decision["decision"]["authorizations"]["p4a_engineering"] = True
        with self.assertRaises(P4FValueError):
            validate_decision(decision)

    def test_rejects_gpu_authority(self) -> None:
        decision = valid_decision()
        decision["decision"]["authorizations"]["gpu_measurement"] = True
        with self.assertRaises(P4FValueError):
            validate_decision(decision)

    def test_rejects_action_admission_authority(self) -> None:
        decision = valid_decision()
        decision["decision"]["authorizations"]["action_admission"] = True
        with self.assertRaises(P4FValueError):
            validate_decision(decision)

    def test_rejects_zero_as_missing_acceptance(self) -> None:
        decision = valid_decision()
        decision["prerequisite_audit"]["exact_action_value"]["acceptance_result"] = 0
        with self.assertRaises(P4FValueError):
            validate_decision(decision)

    def test_rejects_zero_as_missing_robust_gain(self) -> None:
        decision = valid_decision()
        decision["prerequisite_audit"]["phase96_f_portfolio"]["robust_gain_lcb"] = 0.0
        with self.assertRaises(P4FValueError):
            validate_decision(decision)

    def test_rejects_invented_workload_weights(self) -> None:
        decision = valid_decision()
        decision["prerequisite_audit"]["workload_value_contract"][
            "workload_weights"
        ] = {"long_context": 1.0}
        with self.assertRaises(P4FValueError):
            validate_decision(decision)

    def test_rejects_ignoring_negative_unarmed_rounds(self) -> None:
        decision = valid_decision()
        decision["prerequisite_audit"]["phase96_d"]["negative_unarmed_round_count"] = 0
        with self.assertRaises(P4FValueError):
            validate_decision(decision)

    def test_rejects_an_invented_phase96_e_result(self) -> None:
        decision = valid_decision()
        decision["prerequisite_audit"]["phase96_e"]["result_present"] = True
        with self.assertRaises(P4FValueError):
            validate_decision(decision)

    def test_rejects_resource_approval_from_the_proxy(self) -> None:
        decision = valid_decision()
        decision["prerequisite_audit"]["resource_feasibility"]["decision"] = "admit"
        with self.assertRaises(P4FValueError):
            validate_decision(decision)

    def test_rejects_zero_as_measured_transition_latency(self) -> None:
        decision = valid_decision()
        decision["prerequisite_audit"]["transition_overhead"][
            "p95_switch_latency_us"
        ] = 0.0
        with self.assertRaises(P4FValueError):
            validate_decision(decision)

    def test_rejects_an_incomplete_next_preregistration(self) -> None:
        decision = valid_decision()
        decision["decision"]["next_artifact"]["requires"].remove(
            "matched_off_k4_w512_controls"
        )
        with self.assertRaisesRegex(P4FValueError, "omits a prerequisite"):
            validate_decision(decision)


if __name__ == "__main__":
    unittest.main()
