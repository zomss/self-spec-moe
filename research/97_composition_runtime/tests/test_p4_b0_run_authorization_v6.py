"""CPU tests for the Phase 97 B0 atomic-ingress V6 authorization."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR / "scripts"))

from run_p4_b0_value_screen import (  # noqa: E402
    GPU4_UUID,
    INPROCESS_ENGINE_CORE_CLASS,
    V6_OUTPUT_PATH,
    P4RunnerError,
    validate_execution_authority,
)
from validate_p4_b0_run_authorization_v6 import (  # noqa: E402
    B0RunAuthorizationV6Error,
    validate_authorization_v6,
)

AUTHORIZATION_PATH = PHASE_DIR / "data" / "p4" / "p4_b0_run_authorization_v6.json"
SCHEMA_PATH = PHASE_DIR / "schemas" / "p4_b0_run_authorization_v6.schema.json"
V5_AUTHORIZATION_PATH = PHASE_DIR / "data" / "p4" / "p4_b0_run_authorization_v5.json"
V5_FAILURE_PATH = PHASE_DIR / "data" / "p4" / "run_b0_value_screen_v4" / "failure.json"
ATOMIC_PROOF_PATH = (
    PHASE_DIR / "data" / "p4" / "run_b0_atomic_ingress_proof_v2" / "proof.json"
)
V6_FAILURE_PATH = PHASE_DIR / "data" / "p4" / "run_b0_value_screen_v5" / "failure.json"


def _authorization() -> dict:
    return json.loads(AUTHORIZATION_PATH.read_text(encoding="utf-8"))


class P4B0RunAuthorizationV6Tests(unittest.TestCase):
    """Preserve the consumed V6 approval and request-id failure."""

    def test_schema_is_valid(self) -> None:
        Draft202012Validator.check_schema(
            json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        )

    def test_checked_in_v6_package_is_consumed_and_historical(self) -> None:
        self.assertTrue((PHASE_DIR.parents[1] / V6_OUTPUT_PATH).is_dir())
        with self.assertRaisesRegex(
            B0RunAuthorizationV6Error, "source hash drifted.*matrix_runner"
        ):
            validate_authorization_v6(_authorization())

    def test_materialized_v6_binds_inprocess_engine_core(self) -> None:
        authorization = _authorization()

        self.assertFalse(authorization["execution_policy"]["v1_multiprocessing"])
        self.assertEqual(
            authorization["execution_policy"]["engine_core_class"],
            INPROCESS_ENGINE_CORE_CLASS,
        )
        self.assertEqual(
            authorization["execution_policy"]["physical_gpu_uuid"], GPU4_UUID
        )

    def test_v6_failure_preserves_zero_complete_captures(self) -> None:
        failure = json.loads(V6_FAILURE_PATH.read_text(encoding="utf-8"))

        self.assertEqual(
            failure["authorization"]["package_id"],
            "p4-b0-value-screen-run-authorization-v6",
        )
        self.assertEqual(failure["attempt"]["captures_emitted"], 0)
        self.assertEqual(failure["attempt"]["empty_capture_placeholders"], 1)
        self.assertFalse(failure["attempt"]["score_emitted"])
        self.assertEqual(
            failure["diagnostic"]["scope"],
            "request_identity_normalization_mismatch",
        )
        self.assertTrue(failure["disposition"]["v6_consumed"])
        self.assertFalse(failure["disposition"]["scoring_allowed"])

    def test_v5_failure_remains_zero_capture_and_unscored(self) -> None:
        failure = json.loads(V5_FAILURE_PATH.read_text(encoding="utf-8"))

        self.assertEqual(
            failure["authorization"]["package_id"],
            "p4-b0-value-screen-run-authorization-v5",
        )
        self.assertEqual(failure["attempt"]["captures_emitted"], 0)
        self.assertFalse(failure["attempt"]["score_emitted"])
        self.assertTrue(failure["disposition"]["requires_fresh_authorization"])

    def test_atomic_proof_is_passed_but_unscored(self) -> None:
        proof = json.loads(ATOMIC_PROOF_PATH.read_text(encoding="utf-8"))

        self.assertEqual(proof["decision"], "atomic_ingress_gate_passed")
        self.assertFalse(proof["scored"])
        self.assertEqual(proof["trace"]["first_decode_query_widths"], [1] * 8)
        self.assertEqual(proof["trace"]["first_decode_exclusion_reasons"], [])
        self.assertFalse(proof["claims"]["v6_authorized"])

    def test_v5_authorization_cannot_execute_as_v6(self) -> None:
        v5 = json.loads(V5_AUTHORIZATION_PATH.read_text(encoding="utf-8"))
        with self.assertRaises(P4RunnerError):
            validate_execution_authority(v5)

    def test_rejects_source_hash_drift(self) -> None:
        authorization = _authorization()
        authorization["source_artifacts"]["matrix_runner"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(B0RunAuthorizationV6Error, "matrix_runner"):
            validate_authorization_v6(authorization)

    def test_rejects_atomic_proof_inflation(self) -> None:
        authorization = _authorization()
        authorization["atomic_ingress_proof"]["scored"] = True
        with self.assertRaises(B0RunAuthorizationV6Error):
            validate_authorization_v6(authorization)

    def test_rejects_multiprocess_engine_core(self) -> None:
        authorization = _authorization()
        authorization["execution_policy"]["v1_multiprocessing"] = True
        with self.assertRaises(B0RunAuthorizationV6Error):
            validate_authorization_v6(authorization)

    def test_rejects_changed_output_path(self) -> None:
        authorization = _authorization()
        authorization["run_contract"]["invocation"]["output_dir"] += "-other"
        with self.assertRaises(B0RunAuthorizationV6Error):
            validate_authorization_v6(authorization)

    def test_rejects_fallback_gpu(self) -> None:
        authorization = copy.deepcopy(_authorization())
        authorization["execution_policy"]["fallback_gpu_authorized"] = True
        with self.assertRaises(B0RunAuthorizationV6Error):
            validate_authorization_v6(authorization)

    def test_rejects_retry_authority(self) -> None:
        authorization = copy.deepcopy(_authorization())
        authorization["execution_policy"]["retry_allowed"] = True
        with self.assertRaises(B0RunAuthorizationV6Error):
            validate_authorization_v6(authorization)

    def test_rejects_p4a_authority_inflation(self) -> None:
        authorization = _authorization()
        authorization["authorizations"]["p4a_engineering"] = True
        with self.assertRaises(B0RunAuthorizationV6Error):
            validate_authorization_v6(authorization)

    def test_rejects_serving_budget_promotion(self) -> None:
        authorization = _authorization()
        boundary = authorization["serving_chunked_prefill_boundary"]
        boundary["max_num_batched_tokens"] = 114688
        boundary["measurement_override_is_serving_default"] = True
        with self.assertRaises(B0RunAuthorizationV6Error):
            validate_authorization_v6(authorization)

    def test_rejects_frozen_prompt_drift(self) -> None:
        authorization = _authorization()
        authorization["frozen_inputs"]["prompt_bundle"]["sha256"] = "0" * 64
        with self.assertRaises(B0RunAuthorizationV6Error):
            validate_authorization_v6(authorization)


if __name__ == "__main__":
    unittest.main()
