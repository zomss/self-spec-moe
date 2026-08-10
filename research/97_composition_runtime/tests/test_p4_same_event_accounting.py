"""CPU-only tests for Phase 97 same-event target-step accounting."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR / "scripts"))

from validate_p4_same_event_accounting import (  # noqa: E402
    SameEventAccountingError,
    validate_contract,
    validate_event,
)

CONTRACT_PATH = PHASE_DIR / "data" / "p4" / "p4_same_event_accounting_contract.json"
VALIDATION_PATH = PHASE_DIR / "data" / "p4" / "p4_same_event_accounting_validation.json"


def valid_contract() -> dict:
    """Return a copy of the registered accounting contract."""
    return json.loads(CONTRACT_PATH.read_text())


def valid_event(action_id: str = "target-matching-k4") -> dict:
    """Return a complete same-event record with one unarmed target row."""
    k = 0 if action_id == "off" else 4
    rows = [
        {
            "request_id": "request-a",
            "target_processed": True,
            "draft_armed": k > 0,
            "accepted_draft_tokens": k,
            "raw_generated_tokens": k + 1,
            "committed_tokens": k + 1,
            "clipped_tokens": 0,
        },
        {
            "request_id": "request-b",
            "target_processed": True,
            "draft_armed": False,
            "accepted_draft_tokens": 0,
            "raw_generated_tokens": 1,
            "committed_tokens": 1,
            "clipped_tokens": 0,
        },
    ]
    event = {
        "schema_version": 1,
        "contract_id": "p4-same-event-target-step-v1",
        "event_id": "scheduler-event-17",
        "action_id": action_id,
        "k": k,
        "engine_step_index": 17,
        "complete": True,
        "pure_decode": True,
        "source": {
            "scheduler_event_id": "scheduler-event-17",
            "verified_action_id": action_id,
            "next_action_id": action_id,
            "scheduler_output_observed": True,
            "model_runner_output_observed": True,
            "post_stop_commit_observed": True,
            "prometheus_interval_delta_used": False,
        },
        "quality": {
            "preemptions": 0,
            "recomputed_tokens": 0,
            "invalid_spec_tokens": 0,
        },
        "timing": {
            "engine_event_elapsed_s": 0.01,
            "request_decode_time_s": 0.02,
            "source": "same_scheduler_event_monotonic",
            "queue_time_included": False,
            "prefill_time_included": False,
        },
        "request_steps": rows,
        "counters": {},
        "score_eligible": True,
        "exclusion_reasons": [],
    }
    _refresh_counters(event)
    return event


def _refresh_counters(event: dict) -> None:
    rows = event["request_steps"]
    h_steps = len(rows)
    d_armed = sum(row["draft_armed"] for row in rows)
    accepted = sum(row["accepted_draft_tokens"] for row in rows)
    clipped = sum(row["clipped_tokens"] for row in rows)
    committed = sum(row["committed_tokens"] for row in rows)
    event["counters"] = {
        "H_target_steps": h_steps,
        "D_armed": d_armed,
        "A_accepted": accepted,
        "C_clipped": clipped,
        "E_committed": committed,
        "U_unarmed": h_steps - d_armed,
        "closure_holds": True,
        "draft_subset_holds": True,
    }


class SameEventContractTests(unittest.TestCase):
    """Validate the checked-in contract and legacy-data disposition."""

    def test_registered_contract_passes_without_authority(self) -> None:
        result = validate_contract(valid_contract())

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["contract_status"], "registered_unwired")
        self.assertFalse(result["gpu_recollection_authorized"])
        self.assertFalse(result["p4a_engineering_authorized"])

    def test_legacy_w14d_disposition_is_exact(self) -> None:
        result = validate_contract(valid_contract())

        audit = result["legacy_w14d_audit"]
        self.assertEqual(audit["round_count"], 1680)
        self.assertEqual(audit["negative_unarmed_round_count"], 313)
        self.assertEqual(audit["minimum_unarmed_steps"], -8)
        self.assertEqual(result["legacy_w14d_status"], "invalid_for_scored_value")

    def test_checked_validation_output_is_current(self) -> None:
        expected = json.loads(VALIDATION_PATH.read_text())

        self.assertEqual(validate_contract(valid_contract()), expected)

    def test_rejects_event_schema_hash_drift(self) -> None:
        contract = valid_contract()
        contract["event_schema"]["sha256"] = "0" * 64

        with self.assertRaisesRegex(SameEventAccountingError, "hash mismatch"):
            validate_contract(contract)

    def test_rejects_legacy_reuse(self) -> None:
        contract = valid_contract()
        contract["legacy_w14d_disposition"]["reuse_for_value"] = True

        with self.assertRaises(SameEventAccountingError):
            validate_contract(contract)

    def test_rejects_downstream_authority(self) -> None:
        contract = valid_contract()
        contract["authorizations"]["gpu_recollection"] = True

        with self.assertRaises(SameEventAccountingError):
            validate_contract(contract)


class SameEventPositiveTests(unittest.TestCase):
    """Accept exact OFF, K4, w512, clipping, and exclusion records."""

    def test_valid_off_event(self) -> None:
        result = validate_event(valid_event("off"))

        self.assertTrue(result["score_eligible"])
        self.assertEqual(result["counters"]["D_armed"], 0)
        self.assertEqual(result["counters"]["U_unarmed"], 2)

    def test_valid_k4_event_allows_unarmed_target_row(self) -> None:
        result = validate_event(valid_event())

        self.assertEqual(result["counters"]["H_target_steps"], 2)
        self.assertEqual(result["counters"]["D_armed"], 1)
        self.assertEqual(result["counters"]["A_accepted"], 4)
        self.assertEqual(result["counters"]["U_unarmed"], 1)

    def test_valid_w512_event_and_clipping(self) -> None:
        event = valid_event("target-matching-w512-masked-k4")
        row = event["request_steps"][0]
        row["committed_tokens"] = 3
        row["clipped_tokens"] = 2
        _refresh_counters(event)

        result = validate_event(event)

        self.assertEqual(result["counters"]["C_clipped"], 2)
        self.assertEqual(result["counters"]["E_committed"], 4)

    def test_incomplete_event_is_recorded_but_never_scored(self) -> None:
        event = valid_event()
        event["complete"] = False
        event["score_eligible"] = False
        event["exclusion_reasons"] = ["incomplete_event"]

        result = validate_event(event)

        self.assertFalse(result["score_eligible"])

    def test_preempted_event_is_recorded_but_never_scored(self) -> None:
        event = valid_event()
        event["quality"]["preemptions"] = 1
        event["score_eligible"] = False
        event["exclusion_reasons"] = ["preemption"]

        self.assertFalse(validate_event(event)["score_eligible"])


class SameEventFailClosedTests(unittest.TestCase):
    """Reject cross-event, impossible, or optimistically scored records."""

    def test_rejects_cross_event_source(self) -> None:
        event = valid_event()
        event["source"]["scheduler_event_id"] = "scheduler-event-18"

        with self.assertRaisesRegex(SameEventAccountingError, "same scheduler"):
            validate_event(event)

    def test_rejects_transition_event_as_action_value(self) -> None:
        event = valid_event()
        event["source"]["next_action_id"] = "off"

        with self.assertRaisesRegex(SameEventAccountingError, "steady-state"):
            validate_event(event)

    def test_rejects_prometheus_as_scoring_source(self) -> None:
        event = valid_event()
        event["source"]["prometheus_interval_delta_used"] = True

        with self.assertRaises(SameEventAccountingError):
            validate_event(event)

    def test_rejects_duplicate_request_rows(self) -> None:
        event = valid_event()
        event["request_steps"][1]["request_id"] = "request-a"

        with self.assertRaisesRegex(SameEventAccountingError, "unique"):
            validate_event(event)

    def test_rejects_wrong_action_depth(self) -> None:
        event = valid_event()
        event["k"] = 0

        with self.assertRaisesRegex(SameEventAccountingError, "declare K=4"):
            validate_event(event)

    def test_rejects_armed_counter_not_derived_from_rows(self) -> None:
        event = valid_event()
        event["counters"]["D_armed"] = 2
        event["counters"]["U_unarmed"] = 0

        with self.assertRaisesRegex(SameEventAccountingError, "D_armed"):
            validate_event(event)

    def test_rejects_negative_unarmed_disguised_as_zero(self) -> None:
        event = valid_event()
        event["counters"]["D_armed"] = 3
        event["counters"]["U_unarmed"] = 0

        with self.assertRaises(SameEventAccountingError):
            validate_event(event)

    def test_rejects_unarmed_acceptance(self) -> None:
        event = valid_event()
        row = event["request_steps"][1]
        row["accepted_draft_tokens"] = 1
        row["raw_generated_tokens"] = 2
        row["committed_tokens"] = 2
        _refresh_counters(event)

        with self.assertRaisesRegex(SameEventAccountingError, "unarmed request"):
            validate_event(event)

    def test_rejects_acceptance_beyond_k(self) -> None:
        event = valid_event()
        row = event["request_steps"][0]
        row["accepted_draft_tokens"] = 5
        row["raw_generated_tokens"] = 6
        row["committed_tokens"] = 6
        _refresh_counters(event)

        with self.assertRaisesRegex(SameEventAccountingError, "acceptance bound"):
            validate_event(event)

    def test_rejects_raw_output_without_target_token(self) -> None:
        event = valid_event()
        event["request_steps"][0]["raw_generated_tokens"] = 4

        with self.assertRaisesRegex(SameEventAccountingError, "accepted.target"):
            validate_event(event)

    def test_rejects_commit_clipping_mismatch(self) -> None:
        event = valid_event()
        event["request_steps"][0]["committed_tokens"] = 4

        with self.assertRaisesRegex(SameEventAccountingError, "does not close"):
            validate_event(event)

    def test_rejects_decode_timing_from_another_boundary(self) -> None:
        event = valid_event()
        event["timing"]["request_decode_time_s"] = 0.01

        with self.assertRaisesRegex(SameEventAccountingError, "same engine event"):
            validate_event(event)

    def test_rejects_incomplete_event_marked_eligible(self) -> None:
        event = valid_event()
        event["complete"] = False

        with self.assertRaisesRegex(SameEventAccountingError, "exclusion reasons"):
            validate_event(event)

    def test_rejects_exclusion_reason_drift(self) -> None:
        event = valid_event()
        event["exclusion_reasons"] = ["preemption"]
        event["score_eligible"] = False

        with self.assertRaisesRegex(SameEventAccountingError, "exclusion reasons"):
            validate_event(event)

    def test_rejects_off_with_armed_draft(self) -> None:
        event = valid_event("off")
        event["request_steps"][0]["draft_armed"] = True
        _refresh_counters(event)

        with self.assertRaisesRegex(SameEventAccountingError, "OFF event"):
            validate_event(event)


if __name__ == "__main__":
    unittest.main()
