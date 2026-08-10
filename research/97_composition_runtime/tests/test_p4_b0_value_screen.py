"""CPU-only tests for the Phase 97 matched B0 value-screen preregistration."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR / "scripts"))

from validate_p4_b0_value_screen import (  # noqa: E402
    B0ValueScreenError,
    evaluate_value_gate,
    validate_preregistration,
)

PREREG_PATH = PHASE_DIR / "data" / "p4" / "p4_b0_value_screen_prereg.json"
VALIDATION_PATH = PHASE_DIR / "data" / "p4" / "p4_b0_value_screen_validation.json"


def valid_preregistration() -> dict:
    """Return an independent copy of the checked-in preregistration."""
    return json.loads(PREREG_PATH.read_text())


def synthetic_results(gain: float = 0.0) -> dict[str, dict[str, float]]:
    """Return complete non-dominated intervals for all registered regimes."""
    return {
        regime["regime_id"]: {
            "tau_k4_lcb": 4.8,
            "tau_w512_ucb": 4.9,
            "portfolio_gain_lcb": gain,
        }
        for regime in valid_preregistration()["objective"]["regimes"]
    }


class B0ValueScreenPositiveTests(unittest.TestCase):
    """Prove that the research screen is frozen but still blocked."""

    def test_checked_in_preregistration_is_valid_and_blocked(self) -> None:
        result = validate_preregistration(valid_preregistration())

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["readiness"], "blocked")
        self.assertTrue(result["scored_research_objective_frozen"])
        self.assertTrue(result["exact_prompt_manifest_frozen"])
        self.assertTrue(result["same_event_measurement_adapter_frozen"])
        self.assertTrue(result["runner_scorer_frozen"])
        self.assertEqual(result["prompt_record_count"], 384)
        self.assertFalse(result["production_workload_evidence"])
        self.assertFalse(result["gpu_measurement_authorized"])
        self.assertFalse(result["p4a_engineering_authorized"])

    def test_actions_and_equal_w3_weights_are_exact(self) -> None:
        result = validate_preregistration(valid_preregistration())

        self.assertEqual(
            set(result["action_ids"]),
            {
                "off",
                "target-matching-k4",
                "target-matching-w512-masked-k4",
            },
        )
        self.assertEqual(
            set(result["regime_weights"]), {"R4", "R5", "R5cot", "R8", "R1", "R6"}
        )
        self.assertAlmostEqual(sum(result["regime_weights"].values()), 1.0)

    def test_legacy_bundle_cannot_feed_the_screen(self) -> None:
        result = validate_preregistration(valid_preregistration())

        self.assertEqual(result["legacy_w14d_status"], "invalid_for_scored_value")
        self.assertEqual(
            result["next_artifact"],
            "run_ready_b0_value_screen_package",
        )

    def test_checked_validation_output_is_current(self) -> None:
        expected = json.loads(VALIDATION_PATH.read_text())

        self.assertEqual(
            validate_preregistration(valid_preregistration()),
            expected,
        )


class B0ValueScreenFailClosedTests(unittest.TestCase):
    """Reject source drift, invented authority, and unmatched controls."""

    def test_rejects_source_hash_drift(self) -> None:
        prereg = valid_preregistration()
        prereg["source_artifacts"]["p4_entry"]["sha256"] = "0" * 64

        with self.assertRaisesRegex(B0ValueScreenError, "hash mismatch"):
            validate_preregistration(prereg)

    def test_rejects_prompt_manifest_source_hash_drift(self) -> None:
        prereg = valid_preregistration()
        prereg["source_artifacts"]["prompt_manifest"]["sha256"] = "0" * 64

        with self.assertRaisesRegex(B0ValueScreenError, "hash mismatch"):
            validate_preregistration(prereg)

    def test_rejects_prompt_bundle_binding_drift(self) -> None:
        prereg = valid_preregistration()
        prereg["measurement_design"]["prompt_manifest"]["bundle_sha256"] = "0" * 64

        with self.assertRaisesRegex(B0ValueScreenError, "binding drifted"):
            validate_preregistration(prereg)

    def test_rejects_production_workload_claim(self) -> None:
        prereg = valid_preregistration()
        prereg["scope"]["production_workload_evidence"] = True

        with self.assertRaises(B0ValueScreenError):
            validate_preregistration(prereg)

    def test_rejects_legacy_w14d_reuse(self) -> None:
        prereg = valid_preregistration()
        prereg["scope"]["legacy_w14d_reuse_allowed"] = True

        with self.assertRaises(B0ValueScreenError):
            validate_preregistration(prereg)

    def test_rejects_unequal_research_weights(self) -> None:
        prereg = valid_preregistration()
        prereg["objective"]["regimes"][0]["weight"] = 0.2
        prereg["objective"]["regimes"][1]["weight"] -= 1 / 30

        with self.assertRaisesRegex(B0ValueScreenError, "remain equal"):
            validate_preregistration(prereg)

    def test_rejects_regime_substitution(self) -> None:
        prereg = valid_preregistration()
        prereg["objective"]["regimes"][0]["regime_id"] = "R2"

        with self.assertRaisesRegex(B0ValueScreenError, "six-regime"):
            validate_preregistration(prereg)

    def test_rejects_missing_off_control(self) -> None:
        prereg = valid_preregistration()
        prereg["actions"] = prereg["actions"][1:]

        with self.assertRaises(B0ValueScreenError):
            validate_preregistration(prereg)

    def test_rejects_private_kv(self) -> None:
        prereg = valid_preregistration()
        prereg["actions"][2]["kv_path"] = "private_draft"

        with self.assertRaises(B0ValueScreenError):
            validate_preregistration(prereg)

    def test_rejects_window_cost_credit(self) -> None:
        prereg = valid_preregistration()
        prereg["actions"][2]["window"]["cost_credit_allowed"] = True

        with self.assertRaises(B0ValueScreenError):
            validate_preregistration(prereg)

    def test_rejects_separate_boot_window_cost(self) -> None:
        prereg = valid_preregistration()
        prereg["actions"][2]["measurement"]["cost_source"] = "separate-boot-w512"

        with self.assertRaisesRegex(B0ValueScreenError, "window-cost credit"):
            validate_preregistration(prereg)

    def test_rejects_action_order_without_counterbalance(self) -> None:
        prereg = valid_preregistration()
        prereg["measurement_design"]["paired_boot_blocks"][1]["action_order"] = (
            copy.deepcopy(
                prereg["measurement_design"]["paired_boot_blocks"][0]["action_order"]
            )
        )

        with self.assertRaisesRegex(B0ValueScreenError, "counterbalanced"):
            validate_preregistration(prereg)

    def test_rejects_async_screen_protocol_drift(self) -> None:
        prereg = valid_preregistration()
        prereg["measurement_design"]["engine"]["async_scheduling"] = True

        with self.assertRaises(B0ValueScreenError):
            validate_preregistration(prereg)

    def test_rejects_prometheus_scoring(self) -> None:
        prereg = valid_preregistration()
        prereg["measurement_design"]["accounting"]["prometheus_use"] = "scoring"

        with self.assertRaises(B0ValueScreenError):
            validate_preregistration(prereg)

    def test_rejects_exact_resource_evidence_as_a_pre_p4a_requirement(self) -> None:
        prereg = valid_preregistration()
        resource = prereg["decision_rule"]["resource_gate"]
        resource["evidence_grade"] = "measured_exact"
        resource["measurement_relation"] = "exact_candidate"

        with self.assertRaises(B0ValueScreenError):
            validate_preregistration(prereg)

    def test_rejects_a_resource_projection_after_engineering(self) -> None:
        prereg = valid_preregistration()
        post = prereg["decision_rule"]["post_engineering_gate"]
        post["exact_resource_evidence_grade"] = "static_proxy_projection"

        with self.assertRaises(B0ValueScreenError):
            validate_preregistration(prereg)

    def test_rejects_missing_readiness_blocker(self) -> None:
        prereg = valid_preregistration()
        prereg["readiness"]["blocking_reason_codes"].remove(
            "same_event_recorder_unwired"
        )

        with self.assertRaisesRegex(B0ValueScreenError, "every run-readiness"):
            validate_preregistration(prereg)

    def test_rejects_missing_prompt_freeze_satisfaction(self) -> None:
        prereg = valid_preregistration()
        prereg["readiness"]["satisfied"].remove("exact_prompt_manifest_frozen")

        with self.assertRaisesRegex(B0ValueScreenError, "satisfied set"):
            validate_preregistration(prereg)

    def test_rejects_gpu_authority(self) -> None:
        prereg = valid_preregistration()
        prereg["authorizations"]["gpu_measurement"] = True

        with self.assertRaises(B0ValueScreenError):
            validate_preregistration(prereg)

    def test_rejects_p4a_authority(self) -> None:
        prereg = valid_preregistration()
        prereg["authorizations"]["p4a_engineering"] = True

        with self.assertRaises(B0ValueScreenError):
            validate_preregistration(prereg)


class B0ValueGateTests(unittest.TestCase):
    """Exercise the frozen dominance and two W3 value branches."""

    def test_dominance_short_circuit_stops_p4a(self) -> None:
        prereg = valid_preregistration()
        results = synthetic_results(gain=0.03)
        for row in results.values():
            row["tau_w512_ucb"] = row["tau_k4_lcb"]

        decision = evaluate_value_gate(prereg, results)

        self.assertTrue(decision["dominance_short_circuit"])
        self.assertFalse(decision["value_gate_pass"])
        self.assertFalse(decision["authority_granted"])

    def test_mean_gain_branch_can_pass_value_only(self) -> None:
        decision = evaluate_value_gate(
            valid_preregistration(), synthetic_results(gain=0.021)
        )

        self.assertTrue(decision["mean_branch"])
        self.assertTrue(decision["value_gate_pass"])
        self.assertFalse(decision["authority_granted"])

    def test_single_regime_branch_requires_nonnegative_mean(self) -> None:
        prereg = valid_preregistration()
        results = synthetic_results(gain=-0.02)
        results["R4"]["portfolio_gain_lcb"] = 0.06

        decision = evaluate_value_gate(prereg, results)

        self.assertFalse(decision["mean_branch"])
        self.assertFalse(decision["single_branch"])
        self.assertFalse(decision["value_gate_pass"])

    def test_single_regime_branch_passes_at_nonnegative_mean(self) -> None:
        results = synthetic_results(gain=0.0)
        results["R4"]["portfolio_gain_lcb"] = 0.05

        decision = evaluate_value_gate(valid_preregistration(), results)

        self.assertTrue(decision["single_branch"])
        self.assertTrue(decision["value_gate_pass"])

    def test_subthreshold_result_fails(self) -> None:
        decision = evaluate_value_gate(
            valid_preregistration(), synthetic_results(gain=0.019)
        )

        self.assertFalse(decision["value_gate_pass"])

    def test_rejects_missing_regime_result(self) -> None:
        results = synthetic_results()
        del results["R6"]

        with self.assertRaisesRegex(B0ValueScreenError, "exactly"):
            evaluate_value_gate(valid_preregistration(), results)

    def test_rejects_non_finite_result(self) -> None:
        results = synthetic_results()
        results["R6"]["portfolio_gain_lcb"] = float("nan")

        with self.assertRaisesRegex(B0ValueScreenError, "non-finite"):
            evaluate_value_gate(valid_preregistration(), results)


if __name__ == "__main__":
    unittest.main()
