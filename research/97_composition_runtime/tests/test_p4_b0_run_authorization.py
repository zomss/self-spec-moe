"""CPU-only tests for the Phase 97 B0 run-authorization review."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR / "scripts"))

from validate_p4_b0_run_authorization import (  # noqa: E402
    EXPECTED_BLOCKERS,
    B0RunAuthorizationError,
    validate_authorization,
)

AUTHORIZATION_PATH = PHASE_DIR / "data" / "p4" / "p4_b0_run_authorization.json"
SCHEMA_PATH = PHASE_DIR / "schemas" / "p4_b0_run_authorization.schema.json"


def valid_authorization() -> dict:
    """Return an independent copy of the checked-in authorization review."""
    return json.loads(AUTHORIZATION_PATH.read_text(encoding="utf-8"))


class P4B0RunAuthorizationPositiveTests(unittest.TestCase):
    """Verify the evidence-ready but executable-not-ready decision."""

    def test_schema_document_is_valid(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)

    def test_checked_in_package_remains_a_valid_historical_hold(self) -> None:
        result = validate_authorization(
            valid_authorization(),
            enforce_current_sources=False,
        )

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["authorization_decision"], "hold")
        self.assertFalse(result["reviewed_sources_current"])
        self.assertEqual(result["evidence_readiness"], "complete")
        self.assertFalse(result["executable_run_ready"])
        self.assertEqual(
            result["failed_implementation_checks"], list(EXPECTED_BLOCKERS)
        )
        self.assertEqual(result["gpu_physical_index"], 4)
        self.assertEqual(result["physical_boot_count"], 9)
        self.assertEqual(result["capture_count"], 432)
        self.assertEqual(result["minimum_launch_capacity_blocks"], 21682)
        self.assertTrue(result["capture_runner_conformance_engineering_authorized"])
        self.assertFalse(result["gpu_measurement_authorized"])
        self.assertFalse(result["p4a_engineering_authorized"])
        self.assertFalse(result["action_admitted"])

    def test_checked_in_hold_cannot_validate_against_changed_sources(self) -> None:
        with self.assertRaisesRegex(B0RunAuthorizationError, "artifact hash mismatch"):
            validate_authorization(valid_authorization())

    def test_held_invocation_is_exact_but_not_launchable(self) -> None:
        invocation = valid_authorization()["run_contract"]["invocation"]

        self.assertEqual(invocation["argv"][0], ".venv/bin/python")
        self.assertEqual(
            invocation["runner_path"],
            "research/97_composition_runtime/scripts/run_p4_b0_value_screen.py",
        )
        self.assertFalse(invocation["runner_exists"])
        self.assertFalse(invocation["launchable_now"])
        self.assertFalse(invocation["overwrite_allowed"])

    def test_matrix_and_resource_contract_are_closed(self) -> None:
        authorization = valid_authorization()
        matrix = authorization["run_contract"]["matrix"]
        resources = authorization["resource_gates"]

        self.assertEqual(matrix["physical_boot_count"], 9)
        self.assertEqual(matrix["cells_per_boot"], 48)
        self.assertEqual(matrix["raw_capture_count"], 9 * 48)
        self.assertEqual(matrix["adapted_round_count"], 432)
        self.assertEqual(resources["base_available_shared_kv_blocks"], 24529)
        self.assertEqual(resources["conservative_floor_blocks"], 21682)
        self.assertEqual(resources["required_live_kv_blocks"], 21000)
        self.assertFalse(resources["preemption_allowed"])
        self.assertFalse(resources["recomputation_allowed"])


class P4B0RunAuthorizationFailClosedTests(unittest.TestCase):
    """Reject source drift, missing blockers, and authority inflation."""

    def test_rejects_source_reference_hash_drift(self) -> None:
        authorization = valid_authorization()
        authorization["source_artifacts"]["live_runtime"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(B0RunAuthorizationError, "reviewed artifact"):
            validate_authorization(authorization, enforce_current_sources=False)

    def test_rejects_evidence_readiness_invention(self) -> None:
        authorization = valid_authorization()
        authorization["evidence_readiness"]["satisfied"][-1] = "invented_evidence"
        with self.assertRaisesRegex(B0RunAuthorizationError, "readiness closure"):
            validate_authorization(authorization, enforce_current_sources=False)

    def test_rejects_executable_ready_claim(self) -> None:
        authorization = valid_authorization()
        authorization["claims"]["executable_run_ready"] = True
        with self.assertRaises(B0RunAuthorizationError):
            validate_authorization(authorization, enforce_current_sources=False)

    def test_rejects_gpu_measurement_authority(self) -> None:
        authorization = valid_authorization()
        authorization["authorizations"]["gpu_measurement"] = True
        with self.assertRaises(B0RunAuthorizationError):
            validate_authorization(authorization, enforce_current_sources=False)

    def test_rejects_p4a_engineering_authority(self) -> None:
        authorization = valid_authorization()
        authorization["authorizations"]["p4a_engineering"] = True
        with self.assertRaises(B0RunAuthorizationError):
            validate_authorization(authorization, enforce_current_sources=False)

    def test_rejects_self_authorizing_handoff(self) -> None:
        authorization = valid_authorization()
        authorization["next_artifact"]["may_self_authorize_gpu"] = True
        with self.assertRaises(B0RunAuthorizationError):
            validate_authorization(authorization, enforce_current_sources=False)

    def test_rejects_launchable_runner_claim(self) -> None:
        authorization = valid_authorization()
        authorization["run_contract"]["invocation"]["launchable_now"] = True
        with self.assertRaises(B0RunAuthorizationError):
            validate_authorization(authorization, enforce_current_sources=False)

    def test_rejects_gpu_fallback_authority(self) -> None:
        authorization = valid_authorization()
        fallback = authorization["run_contract"]["gpu_assignment"]["fallback_gpu"]
        fallback["authorized"] = True
        with self.assertRaises(B0RunAuthorizationError):
            validate_authorization(authorization, enforce_current_sources=False)

    def test_rejects_action_schedule_drift(self) -> None:
        authorization = valid_authorization()
        boots = authorization["run_contract"]["action_boots"]
        boots[0]["dynamic_k_schedule"][1][2] = 0
        with self.assertRaisesRegex(B0RunAuthorizationError, "K schedule"):
            validate_authorization(authorization, enforce_current_sources=False)

    def test_rejects_resource_floor_reduction(self) -> None:
        authorization = valid_authorization()
        authorization["resource_gates"]["minimum_launch_capacity_blocks"] = 21000
        with self.assertRaisesRegex(B0RunAuthorizationError, "resource gates"):
            validate_authorization(authorization, enforce_current_sources=False)

    def test_rejects_missing_hold_reason(self) -> None:
        authorization = valid_authorization()
        authorization["decision"]["reason_codes"][-1] = "invented_blocker"
        with self.assertRaisesRegex(B0RunAuthorizationError, "HOLD reason"):
            validate_authorization(authorization, enforce_current_sources=False)

    def test_rejects_weakened_blocker_evidence(self) -> None:
        authorization = copy.deepcopy(valid_authorization())
        authorization["implementation_audit"]["checks"][0]["evidence"] = "Not checked."
        with self.assertRaisesRegex(B0RunAuthorizationError, "evidence was weakened"):
            validate_authorization(authorization, enforce_current_sources=False)

    def test_rejects_runner_path_drift(self) -> None:
        authorization = valid_authorization()
        invocation = authorization["run_contract"]["invocation"]
        invocation["runner_path"] = "research/97_composition_runtime/scripts/other.py"
        with self.assertRaisesRegex(B0RunAuthorizationError, "invocation drifted"):
            validate_authorization(authorization, enforce_current_sources=False)


if __name__ == "__main__":
    unittest.main()
