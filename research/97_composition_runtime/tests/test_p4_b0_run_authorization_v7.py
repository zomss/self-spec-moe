"""CPU tests for the Phase 97 B0 request-ID V7 authorization."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
sys.path.insert(0, str(PHASE_DIR / "scripts"))

from run_p4_b0_value_screen import (  # noqa: E402
    GPU4_UUID,
    INPROCESS_ENGINE_CORE_CLASS,
    V7_OUTPUT_PATH,
)
from validate_p4_b0_run_authorization_v7 import (  # noqa: E402
    B0RunAuthorizationV7Error,
    validate_authorization_v7,
)

AUTHORIZATION_PATH = PHASE_DIR / "data" / "p4" / "p4_b0_run_authorization_v7.json"
SCHEMA_PATH = PHASE_DIR / "schemas" / "p4_b0_run_authorization_v7.schema.json"
V6_FAILURE_PATH = PHASE_DIR / "data" / "p4" / "run_b0_value_screen_v5" / "failure.json"
V7_FAILURE_PATH = PHASE_DIR / "data" / "p4" / "run_b0_value_screen_v6" / "failure.json"


def _authorization() -> dict:
    return json.loads(AUTHORIZATION_PATH.read_text(encoding="utf-8"))


class P4B0RunAuthorizationV7Tests(unittest.TestCase):
    """Preserve the consumed V7 approval and decode-work failure."""

    def test_schema_is_valid(self) -> None:
        Draft202012Validator.check_schema(
            json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        )

    def test_checked_in_v7_package_is_consumed_and_historical(self) -> None:
        self.assertTrue((REPO_ROOT / V7_OUTPUT_PATH).is_dir())
        with self.assertRaisesRegex(
            B0RunAuthorizationV7Error, "source hash drifted.*matrix_runner"
        ):
            validate_authorization_v7(_authorization())

    def test_materialized_v7_binds_inprocess_engine_core(self) -> None:
        authorization = _authorization()

        self.assertFalse(authorization["execution_policy"]["v1_multiprocessing"])
        self.assertEqual(
            authorization["execution_policy"]["engine_core_class"],
            INPROCESS_ENGINE_CORE_CLASS,
        )
        self.assertEqual(
            authorization["execution_policy"]["physical_gpu_uuid"], GPU4_UUID
        )

    def test_v7_failure_preserves_one_incomplete_capture(self) -> None:
        failure = json.loads(V7_FAILURE_PATH.read_text(encoding="utf-8"))

        self.assertEqual(
            failure["authorization"]["package_id"],
            "p4-b0-value-screen-run-authorization-v7",
        )
        self.assertEqual(failure["attempt"]["complete_captures_emitted"], 0)
        self.assertEqual(failure["attempt"]["incomplete_captures_emitted"], 1)
        self.assertFalse(failure["attempt"]["score_emitted"])
        self.assertEqual(
            failure["diagnostic"]["scope"],
            "prefill_sample_vs_decode_only_fixed_work_off_by_one",
        )
        self.assertTrue(failure["disposition"]["v7_consumed"])
        self.assertFalse(failure["disposition"]["scoring_allowed"])

    def test_v6_failure_is_preserved_without_resume(self) -> None:
        failure = json.loads(V6_FAILURE_PATH.read_text(encoding="utf-8"))

        self.assertEqual(failure["attempt"]["captures_emitted"], 0)
        self.assertFalse(failure["attempt"]["score_emitted"])
        self.assertEqual(
            failure["diagnostic"]["scope"],
            "request_identity_normalization_mismatch",
        )
        self.assertTrue(failure["disposition"]["v6_consumed"])
        self.assertFalse(failure["disposition"]["retry_attempted"])

    def test_repair_preserves_randomization_and_canonicalizes_output(self) -> None:
        repair = _authorization()["request_id_repair"]

        self.assertEqual(repair["shared_helper"], "canonicalize_p4_request_ids")
        self.assertFalse(repair["disable_randomization_allowed"])
        self.assertEqual(repair["canonical_output"], "exact_frozen_request_id")
        self.assertIn("canonical_collision", repair["fail_closed_cases"])

    def test_rejects_source_hash_drift(self) -> None:
        authorization = _authorization()
        authorization["source_artifacts"]["matrix_runner"]["sha256"] = "0" * 64

        with self.assertRaisesRegex(B0RunAuthorizationV7Error, "matrix_runner"):
            validate_authorization_v7(authorization)

    def test_rejects_randomization_disable(self) -> None:
        authorization = _authorization()
        authorization["request_id_repair"]["disable_randomization_allowed"] = True

        with self.assertRaises(B0RunAuthorizationV7Error):
            validate_authorization_v7(authorization)

    def test_rejects_changed_output_path(self) -> None:
        authorization = _authorization()
        authorization["run_contract"]["invocation"]["output_dir"] += "-other"

        with self.assertRaises(B0RunAuthorizationV7Error):
            validate_authorization_v7(authorization)

    def test_rejects_fallback_retry_and_p4a_authority(self) -> None:
        mutations = (
            ("fallback_gpu_authorized", True),
            ("retry_allowed", True),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                authorization = copy.deepcopy(_authorization())
                authorization["execution_policy"][field] = value
                with self.assertRaises(B0RunAuthorizationV7Error):
                    validate_authorization_v7(authorization)

        authorization = copy.deepcopy(_authorization())
        authorization["authorizations"]["p4a_engineering"] = True
        with self.assertRaises(B0RunAuthorizationV7Error):
            validate_authorization_v7(authorization)


if __name__ == "__main__":
    unittest.main()
