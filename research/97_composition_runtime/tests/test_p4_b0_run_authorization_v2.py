"""CPU tests for the source-bound Phase 97 B0 GPU-4 authorization."""

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
    execute_run,
    validate_execution_authority,
)
from validate_p4_b0_run_authorization_v2 import (  # noqa: E402
    B0RunAuthorizationV2Error,
    validate_authorization_v2,
)

AUTHORIZATION_PATH = PHASE_DIR / "data" / "p4" / "p4_b0_run_authorization_v2.json"
SCHEMA_PATH = PHASE_DIR / "schemas" / "p4_b0_run_authorization_v2.schema.json"


def _authorization() -> dict:
    return json.loads(AUTHORIZATION_PATH.read_text(encoding="utf-8"))


class P4B0RunAuthorizationV2Tests(unittest.TestCase):
    """Prove the one-screen approval and its fail-closed boundary."""

    def test_schema_is_valid(self) -> None:
        Draft202012Validator.check_schema(
            json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        )

    def test_checked_in_approval_is_consumed_and_source_historical(self) -> None:
        with self.assertRaisesRegex(
            B0RunAuthorizationV2Error, "source hash drifted for live_runtime"
        ):
            validate_authorization_v2(_authorization())

    def test_live_runner_rejects_consumed_package(self) -> None:
        with self.assertRaisesRegex(P4RunnerError, "reviewed V5"):
            validate_execution_authority(_authorization())

    def test_rejects_conformance_hash_drift(self) -> None:
        authorization = _authorization()
        reference = authorization["source_artifacts"]["capture_runner_conformance"]
        reference["sha256"] = "0" * 64
        with self.assertRaisesRegex(
            B0RunAuthorizationV2Error,
            "source bindings",
        ):
            validate_authorization_v2(authorization)

    def test_live_runner_rejects_incomplete_source_closure(self) -> None:
        authorization = _authorization()
        del authorization["source_artifacts"]["model_runner"]
        with self.assertRaisesRegex(P4RunnerError, "reviewed V5"):
            validate_execution_authority(authorization)

    def test_live_runner_rejects_direct_source_hash_drift(self) -> None:
        authorization = _authorization()
        authorization["source_artifacts"]["model_runner"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(P4RunnerError, "reviewed V5"):
            validate_execution_authority(authorization)

    def test_parent_rejects_another_authorization_path(self) -> None:
        output = PHASE_DIR / "data" / "p4" / "run_b0_value_screen_v1"
        with self.assertRaisesRegex(P4RunnerError, "authorization path"):
            execute_run(
                Path("/tmp/copied-authorization.json"),
                _authorization(),
                output,
            )

    def test_parent_rejects_another_output_path(self) -> None:
        with self.assertRaisesRegex(P4RunnerError, "authorization path"):
            execute_run(
                AUTHORIZATION_PATH,
                _authorization(),
                Path("/tmp/other-p4-output"),
            )

    def test_rejects_p4a_authority_inflation(self) -> None:
        authorization = _authorization()
        authorization["authorizations"]["p4a_engineering"] = True
        with self.assertRaises(B0RunAuthorizationV2Error):
            validate_authorization_v2(authorization)

    def test_rejects_gpu_fallback(self) -> None:
        authorization = _authorization()
        fallback = authorization["run_contract"]["gpu_assignment"]["fallback_gpu"]
        fallback["authorized"] = True
        with self.assertRaises(B0RunAuthorizationV2Error):
            validate_authorization_v2(authorization)

    def test_rejects_partial_resume(self) -> None:
        authorization = copy.deepcopy(_authorization())
        authorization["execution_policy"]["partial_resume_allowed"] = True
        with self.assertRaises(B0RunAuthorizationV2Error):
            validate_authorization_v2(authorization)

    def test_rejects_changed_invocation(self) -> None:
        authorization = _authorization()
        authorization["run_contract"]["invocation"]["output_dir"] += "-other"
        with self.assertRaises(B0RunAuthorizationV2Error):
            validate_authorization_v2(authorization)


if __name__ == "__main__":
    unittest.main()
