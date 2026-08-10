"""CPU tests for the Phase 97 B0 native-sampler retry authorization."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR / "scripts"))

from run_p4_b0_value_screen import (  # noqa: E402
    NATIVE_SAMPLER_ENV,
    P4RunnerError,
    resolve_authorization_package,
    validate_execution_authority,
)
from validate_p4_b0_run_authorization_v4 import (  # noqa: E402
    B0RunAuthorizationV4Error,
    validate_authorization_v4,
)

AUTHORIZATION_PATH = PHASE_DIR / "data" / "p4" / "p4_b0_run_authorization_v4.json"
FAILURE_PATH = PHASE_DIR / "data" / "p4" / "run_b0_value_screen_v3" / "failure.json"
CONSUMED_OUTPUT_PATH = FAILURE_PATH.parent
SCHEMA_PATH = PHASE_DIR / "schemas" / "p4_b0_run_authorization_v4.schema.json"
REPO_ROOT = PHASE_DIR.parents[1]


def _authorization() -> dict:
    return json.loads(AUTHORIZATION_PATH.read_text(encoding="utf-8"))


def _authorization_with_current_sources() -> dict:
    authorization = _authorization()
    for reference in authorization["source_artifacts"].values():
        path = REPO_ROOT / reference["path"]
        reference["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return authorization


class P4B0RunAuthorizationV4Tests(unittest.TestCase):
    """Preserve the consumed native-sampler approval and narrow authority."""

    def test_schema_is_valid(self) -> None:
        Draft202012Validator.check_schema(
            json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        )

    def test_checked_in_retry_is_consumed_and_historical(self) -> None:
        self.assertTrue(CONSUMED_OUTPUT_PATH.is_dir())
        with self.assertRaisesRegex(
            B0RunAuthorizationV4Error, "V4 retry source hash drifted for matrix_runner"
        ):
            validate_authorization_v4(_authorization())

    def test_live_runner_rejects_consumed_retry(self) -> None:
        with self.assertRaisesRegex(
            P4RunnerError, "V4 retry source hash drifted for matrix_runner"
        ):
            resolve_authorization_package(_authorization())

    def test_failure_record_preserves_zero_complete_captures(self) -> None:
        failure = json.loads(FAILURE_PATH.read_text(encoding="utf-8"))

        self.assertEqual(
            failure["authorization"]["package_id"],
            "p4-b0-value-screen-run-authorization-v4",
        )
        self.assertEqual(failure["attempt"]["captures_emitted"], 0)
        self.assertEqual(failure["attempt"]["empty_capture_placeholders"], 1)
        self.assertFalse(failure["attempt"]["score_emitted"])
        self.assertTrue(failure["disposition"]["requires_fresh_authorization"])
        self.assertFalse(failure["disposition"]["scoring_allowed"])

    def test_repair_requires_native_sampler_environment(self) -> None:
        authorization = _authorization()
        repair = authorization["repair_audit"]
        repair["required_environment"][NATIVE_SAMPLER_ENV] = "1"
        with self.assertRaises(B0RunAuthorizationV4Error):
            validate_authorization_v4(authorization)

    def test_rejects_prior_authorization_hash_drift(self) -> None:
        authorization = _authorization()
        authorization["prior_authorization"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(
            B0RunAuthorizationV4Error, "consumed V3 authorization"
        ):
            validate_authorization_v4(authorization)

    def test_rejects_failed_attempt_capture_inflation(self) -> None:
        authorization = _authorization()
        authorization["failed_attempt"]["capture_count"] = 1
        with self.assertRaises(B0RunAuthorizationV4Error):
            validate_authorization_v4(authorization)

    def test_rejects_retry_source_hash_drift(self) -> None:
        authorization = _authorization()
        authorization["source_artifacts"]["matrix_runner"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(B0RunAuthorizationV4Error, "matrix_runner"):
            validate_authorization_v4(authorization)

    def test_rejects_sampler_source_hash_drift(self) -> None:
        authorization = _authorization_with_current_sources()
        authorization["source_artifacts"]["sampler_backend"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(B0RunAuthorizationV4Error, "sampler_backend"):
            validate_authorization_v4(authorization)

    def test_rejects_changed_output_path(self) -> None:
        authorization = _authorization()
        authorization["run_contract"]["invocation"]["output_dir"] += "-other"
        with self.assertRaises(B0RunAuthorizationV4Error):
            validate_authorization_v4(authorization)

    def test_rejects_prior_output_reuse(self) -> None:
        authorization = copy.deepcopy(_authorization())
        authorization["execution_policy"]["prior_output_reuse_allowed"] = True
        with self.assertRaises(B0RunAuthorizationV4Error):
            validate_authorization_v4(authorization)

    def test_rejects_partial_resume(self) -> None:
        authorization = copy.deepcopy(_authorization())
        authorization["execution_policy"]["partial_resume_allowed"] = True
        with self.assertRaises(B0RunAuthorizationV4Error):
            validate_authorization_v4(authorization)

    def test_rejects_p4a_authority_inflation(self) -> None:
        authorization = _authorization()
        authorization["authorizations"]["p4a_engineering"] = True
        with self.assertRaises(B0RunAuthorizationV4Error):
            validate_authorization_v4(authorization)

    def test_consumed_v3_cannot_execute_directly(self) -> None:
        path = PHASE_DIR / "data" / "p4" / "p4_b0_run_authorization_v3.json"
        consumed = json.loads(path.read_text(encoding="utf-8"))
        with self.assertRaisesRegex(P4RunnerError, "reviewed V5"):
            validate_execution_authority(consumed)


if __name__ == "__main__":
    unittest.main()
