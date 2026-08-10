"""CPU tests for the Phase 97 B0 decode-work V8 authorization."""

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

from run_p4_b0_value_screen import V8_OUTPUT_PATH  # noqa: E402
from validate_p4_b0_run_authorization_v8 import (  # noqa: E402
    B0RunAuthorizationV8Error,
    validate_authorization_v8,
)

AUTHORIZATION_PATH = PHASE_DIR / "data" / "p4" / "p4_b0_run_authorization_v8.json"
SCHEMA_PATH = PHASE_DIR / "schemas" / "p4_b0_run_authorization_v8.schema.json"
V7_FAILURE_PATH = PHASE_DIR / "data" / "p4" / "run_b0_value_screen_v6" / "failure.json"
V8_FAILURE_PATH = PHASE_DIR / "data" / "p4" / "run_b0_value_screen_v7" / "failure.json"


def _authorization() -> dict:
    return json.loads(AUTHORIZATION_PATH.read_text(encoding="utf-8"))


class P4B0RunAuthorizationV8Tests(unittest.TestCase):
    """Preserve the consumed V8 approval and launch-dispatch refusal."""

    def test_schema_is_valid(self) -> None:
        Draft202012Validator.check_schema(
            json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        )

    def test_checked_in_v8_package_is_consumed_and_historical(self) -> None:
        self.assertTrue((REPO_ROOT / V8_OUTPUT_PATH).is_dir())
        with self.assertRaisesRegex(
            B0RunAuthorizationV8Error, "source hash drifted.*matrix_runner"
        ):
            validate_authorization_v8(_authorization())

    def test_v8_package_preserves_decode_work_offset_contract(self) -> None:
        repair = _authorization()["decode_work_repair"]

        self.assertEqual(repair["measurement_currency"], "S_dec")
        self.assertEqual(repair["prefill_sampled_tokens_per_request"], 1)
        self.assertEqual(
            repair["frontend_total_output_tokens"],
            "generation.max_output_tokens + 1",
        )

    def test_v8_refusal_preserves_zero_gpu_work(self) -> None:
        failure = json.loads(V8_FAILURE_PATH.read_text(encoding="utf-8"))

        self.assertEqual(failure["attempt"]["complete_captures_emitted"], 0)
        self.assertEqual(failure["attempt"]["incomplete_captures_emitted"], 0)
        self.assertFalse(failure["attempt"]["gpu_model_executed"])
        self.assertFalse(failure["attempt"]["score_emitted"])
        self.assertEqual(
            failure["diagnostic"]["scope"],
            "reviewed_authorization_path_dispatch_omission",
        )
        self.assertTrue(failure["disposition"]["v8_consumed"])
        self.assertFalse(failure["disposition"]["retry_attempted"])

    def test_v7_failure_is_preserved_without_resume(self) -> None:
        failure = json.loads(V7_FAILURE_PATH.read_text(encoding="utf-8"))

        self.assertEqual(failure["attempt"]["complete_captures_emitted"], 0)
        self.assertEqual(failure["attempt"]["incomplete_captures_emitted"], 1)
        self.assertFalse(failure["attempt"]["score_emitted"])
        self.assertEqual(
            failure["diagnostic"]["scope"],
            "prefill_sample_vs_decode_only_fixed_work_off_by_one",
        )
        self.assertTrue(failure["disposition"]["v7_consumed"])
        self.assertFalse(failure["disposition"]["retry_attempted"])

    def test_repair_keeps_decode_currency_and_adds_only_prefill_offset(self) -> None:
        repair = _authorization()["decode_work_repair"]

        self.assertEqual(repair["measurement_currency"], "S_dec")
        self.assertEqual(repair["prefill_sampled_tokens_per_request"], 1)
        self.assertEqual(
            repair["frontend_total_output_tokens"],
            "generation.max_output_tokens + 1",
        )
        self.assertEqual(
            repair["recorder_completion_tokens"],
            "generation.max_output_tokens",
        )
        self.assertEqual(
            set(repair["applies_to_actions"]),
            {
                "off",
                "target-matching-k4",
                "target-matching-w512-masked-k4",
            },
        )

    def test_rejects_source_hash_drift(self) -> None:
        authorization = _authorization()
        authorization["source_artifacts"]["matrix_runner"]["sha256"] = "0" * 64

        with self.assertRaisesRegex(B0RunAuthorizationV8Error, "matrix_runner"):
            validate_authorization_v8(authorization)

    def test_rejects_prefill_sample_offset_drift(self) -> None:
        authorization = _authorization()
        authorization["decode_work_repair"]["prefill_sampled_tokens_per_request"] = 2

        with self.assertRaises(B0RunAuthorizationV8Error):
            validate_authorization_v8(authorization)

    def test_rejects_changed_output_path(self) -> None:
        authorization = _authorization()
        authorization["run_contract"]["invocation"]["output_dir"] += "-other"

        with self.assertRaises(B0RunAuthorizationV8Error):
            validate_authorization_v8(authorization)

    def test_rejects_fallback_retry_and_p4a_authority(self) -> None:
        mutations = (
            ("fallback_gpu_authorized", True),
            ("retry_allowed", True),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                authorization = copy.deepcopy(_authorization())
                authorization["execution_policy"][field] = value
                with self.assertRaises(B0RunAuthorizationV8Error):
                    validate_authorization_v8(authorization)

        authorization = copy.deepcopy(_authorization())
        authorization["authorizations"]["p4a_engineering"] = True
        with self.assertRaises(B0RunAuthorizationV8Error):
            validate_authorization_v8(authorization)


if __name__ == "__main__":
    unittest.main()
