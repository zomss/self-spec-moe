"""CPU tests for the source-bound Phase 97 B0 retry authorization."""

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
    P4RunnerError,
    resolve_authorization_package,
    validate_execution_authority,
)
from validate_p4_b0_run_authorization_v3 import (  # noqa: E402
    B0RunAuthorizationV3Error,
    validate_authorization_v3,
)

AUTHORIZATION_PATH = PHASE_DIR / "data" / "p4" / "p4_b0_run_authorization_v3.json"
SCHEMA_PATH = PHASE_DIR / "schemas" / "p4_b0_run_authorization_v3.schema.json"


def _authorization() -> dict:
    return json.loads(AUTHORIZATION_PATH.read_text(encoding="utf-8"))


class P4B0RunAuthorizationV3Tests(unittest.TestCase):
    """Prove the create-new retry approval and fail-closed boundary."""

    def test_schema_is_valid(self) -> None:
        Draft202012Validator.check_schema(
            json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        )

    def test_checked_in_retry_is_consumed_and_historical(self) -> None:
        with self.assertRaisesRegex(B0RunAuthorizationV3Error, "reviewed V5"):
            validate_authorization_v3(_authorization())

    def test_live_runner_rejects_consumed_retry(self) -> None:
        effective = resolve_authorization_package(_authorization())
        with self.assertRaisesRegex(P4RunnerError, "reviewed V5"):
            validate_execution_authority(effective)

    def test_rejects_prior_authorization_hash_drift(self) -> None:
        authorization = _authorization()
        authorization["prior_authorization"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(B0RunAuthorizationV3Error, "reviewed V5"):
            validate_authorization_v3(authorization)

    def test_rejects_failed_attempt_capture_inflation(self) -> None:
        authorization = _authorization()
        authorization["failed_attempt"]["capture_count"] = 1
        with self.assertRaises(B0RunAuthorizationV3Error):
            validate_authorization_v3(authorization)

    def test_rejects_retry_source_hash_drift(self) -> None:
        authorization = _authorization()
        authorization["source_artifacts"]["matrix_runner"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(B0RunAuthorizationV3Error, "reviewed V5"):
            validate_authorization_v3(authorization)

    def test_rejects_changed_output_path(self) -> None:
        authorization = _authorization()
        authorization["run_contract"]["invocation"]["output_dir"] += "-other"
        with self.assertRaises(B0RunAuthorizationV3Error):
            validate_authorization_v3(authorization)

    def test_rejects_prior_output_reuse(self) -> None:
        authorization = copy.deepcopy(_authorization())
        authorization["execution_policy"]["prior_output_reuse_allowed"] = True
        with self.assertRaises(B0RunAuthorizationV3Error):
            validate_authorization_v3(authorization)

    def test_rejects_partial_resume(self) -> None:
        authorization = copy.deepcopy(_authorization())
        authorization["execution_policy"]["partial_resume_allowed"] = True
        with self.assertRaises(B0RunAuthorizationV3Error):
            validate_authorization_v3(authorization)

    def test_rejects_p4a_authority_inflation(self) -> None:
        authorization = _authorization()
        authorization["authorizations"]["p4a_engineering"] = True
        with self.assertRaises(B0RunAuthorizationV3Error):
            validate_authorization_v3(authorization)

    def test_consumed_v2_cannot_execute_directly(self) -> None:
        path = PHASE_DIR / "data" / "p4" / "p4_b0_run_authorization_v2.json"
        consumed = json.loads(path.read_text(encoding="utf-8"))
        with self.assertRaisesRegex(P4RunnerError, "reviewed V5"):
            validate_execution_authority(consumed)


if __name__ == "__main__":
    unittest.main()
