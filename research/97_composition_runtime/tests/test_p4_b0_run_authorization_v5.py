"""CPU tests for the Phase 97 B0 full-prefill resource retry authorization."""

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
    APPROVED_OUTPUT_PATH,
    FULL_PREFILL_MAX_NUM_BATCHED_TOKENS,
    MEASUREMENT_GPU_MEMORY_UTILIZATION,
    SERVING_MAX_NUM_BATCHED_TOKENS,
    P4RunnerError,
    resolve_authorization_package,
)
from validate_p4_b0_run_authorization_v5 import (  # noqa: E402
    B0RunAuthorizationV5Error,
    validate_authorization_v5,
)

AUTHORIZATION_PATH = PHASE_DIR / "data" / "p4" / "p4_b0_run_authorization_v5.json"
SCHEMA_PATH = PHASE_DIR / "schemas" / "p4_b0_run_authorization_v5.schema.json"
V4_AUTHORIZATION_PATH = PHASE_DIR / "data" / "p4" / "p4_b0_run_authorization_v4.json"
V4_FAILURE_PATH = PHASE_DIR / "data" / "p4" / "run_b0_value_screen_v3" / "failure.json"
RESOURCE_RESULT_PATH = (
    PHASE_DIR / "data" / "p4" / "p4_b0_full_prefill_resource_repair_probe.json"
)
V5_FAILURE_PATH = PHASE_DIR / "data" / "p4" / "run_b0_value_screen_v4" / "failure.json"


def _authorization() -> dict:
    return json.loads(AUTHORIZATION_PATH.read_text(encoding="utf-8"))


class P4B0RunAuthorizationV5Tests(unittest.TestCase):
    """Preserve the consumed retry and its separate serving boundary."""

    def test_schema_is_valid(self) -> None:
        Draft202012Validator.check_schema(
            json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        )

    def test_checked_in_retry_is_consumed_and_historical(self) -> None:
        self.assertTrue((PHASE_DIR.parents[1] / APPROVED_OUTPUT_PATH).is_dir())
        with self.assertRaisesRegex(B0RunAuthorizationV5Error, "source hash drifted"):
            validate_authorization_v5(_authorization())

    def test_consumed_engine_keeps_registered_measurement_overrides(self) -> None:
        repair = _authorization()["measurement_repair"]["allowed_engine_overrides"]

        self.assertEqual(
            repair["max_num_batched_tokens"], FULL_PREFILL_MAX_NUM_BATCHED_TOKENS
        )
        self.assertTrue(repair["enable_chunked_prefill"])
        self.assertEqual(
            repair["gpu_memory_utilization"], MEASUREMENT_GPU_MEMORY_UTILIZATION
        )

    def test_resource_probe_pass_is_exact(self) -> None:
        result = json.loads(RESOURCE_RESULT_PATH.read_text(encoding="utf-8"))
        evidence = _authorization()["resource_probe"]

        self.assertEqual(result["decision"]["state"], "pass")
        self.assertEqual(result["shared_kv_capacity"]["num_gpu_blocks"], 22113)
        self.assertEqual(result["shared_kv_capacity"]["headroom_blocks"], 431)
        self.assertEqual(evidence["num_gpu_blocks"], 22113)
        self.assertEqual(evidence["requests_executed"], 0)

    def test_real_serving_8192_chunked_prefill_is_not_promoted(self) -> None:
        boundary = _authorization()["serving_chunked_prefill_boundary"]

        self.assertEqual(
            boundary["max_num_batched_tokens"], SERVING_MAX_NUM_BATCHED_TOKENS
        )
        self.assertTrue(boundary["chunked_prefill_enabled"])
        self.assertFalse(boundary["measurement_override_is_serving_default"])
        self.assertTrue(boundary["mixed_prefill_decode_diagnosis_required"])
        self.assertFalse(_authorization()["authorizations"]["serving_diagnosis"])

    def test_v4_failure_remains_immutable_and_unscored(self) -> None:
        failure = json.loads(V4_FAILURE_PATH.read_text(encoding="utf-8"))

        self.assertEqual(
            failure["authorization"]["package_id"],
            "p4-b0-value-screen-run-authorization-v4",
        )
        self.assertEqual(failure["attempt"]["captures_emitted"], 0)
        self.assertFalse(failure["attempt"]["score_emitted"])
        self.assertTrue(failure["output"]["preserve_without_overwrite_or_resume"])

    def test_v5_failure_preserves_zero_complete_captures(self) -> None:
        failure = json.loads(V5_FAILURE_PATH.read_text(encoding="utf-8"))

        self.assertEqual(
            failure["authorization"]["package_id"],
            "p4-b0-value-screen-run-authorization-v5",
        )
        self.assertEqual(failure["attempt"]["captures_emitted"], 0)
        self.assertEqual(failure["attempt"]["empty_capture_placeholders"], 1)
        self.assertFalse(failure["attempt"]["score_emitted"])
        self.assertTrue(failure["disposition"]["requires_fresh_authorization"])
        self.assertFalse(failure["disposition"]["scoring_allowed"])

    def test_rejects_source_hash_drift(self) -> None:
        authorization = _authorization()
        authorization["source_artifacts"]["matrix_runner"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(B0RunAuthorizationV5Error, "matrix_runner"):
            validate_authorization_v5(authorization)

    def test_rejects_prior_authorization_hash_drift(self) -> None:
        authorization = _authorization()
        authorization["prior_authorization"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(
            B0RunAuthorizationV5Error, "consumed V4 authorization"
        ):
            validate_authorization_v5(authorization)

    def test_rejects_failed_attempt_capture_inflation(self) -> None:
        authorization = _authorization()
        authorization["failed_attempt"]["capture_count"] = 1
        with self.assertRaises(B0RunAuthorizationV5Error):
            validate_authorization_v5(authorization)

    def test_rejects_resource_probe_inflation(self) -> None:
        authorization = _authorization()
        authorization["resource_probe"]["num_gpu_blocks"] += 1
        with self.assertRaises(B0RunAuthorizationV5Error):
            validate_authorization_v5(authorization)

    def test_rejects_measurement_memory_drift(self) -> None:
        authorization = _authorization()
        overrides = authorization["measurement_repair"]["allowed_engine_overrides"]
        overrides["gpu_memory_utilization"] = 0.95
        with self.assertRaises(B0RunAuthorizationV5Error):
            validate_authorization_v5(authorization)

    def test_rejects_serving_budget_promotion(self) -> None:
        authorization = _authorization()
        boundary = authorization["serving_chunked_prefill_boundary"]
        boundary["max_num_batched_tokens"] = 114688
        boundary["measurement_override_is_serving_default"] = True
        with self.assertRaises(B0RunAuthorizationV5Error):
            validate_authorization_v5(authorization)

    def test_rejects_changed_output_path(self) -> None:
        authorization = _authorization()
        authorization["run_contract"]["invocation"]["output_dir"] += "-other"
        with self.assertRaises(B0RunAuthorizationV5Error):
            validate_authorization_v5(authorization)

    def test_rejects_prior_output_reuse(self) -> None:
        authorization = copy.deepcopy(_authorization())
        authorization["execution_policy"]["prior_output_reuse_allowed"] = True
        with self.assertRaises(B0RunAuthorizationV5Error):
            validate_authorization_v5(authorization)

    def test_rejects_fallback_gpu(self) -> None:
        authorization = copy.deepcopy(_authorization())
        authorization["execution_policy"]["fallback_gpu_authorized"] = True
        with self.assertRaises(B0RunAuthorizationV5Error):
            validate_authorization_v5(authorization)

    def test_rejects_p4a_authority_inflation(self) -> None:
        authorization = _authorization()
        authorization["authorizations"]["p4a_engineering"] = True
        with self.assertRaises(B0RunAuthorizationV5Error):
            validate_authorization_v5(authorization)

    def test_consumed_v4_cannot_execute_directly(self) -> None:
        consumed = json.loads(V4_AUTHORIZATION_PATH.read_text(encoding="utf-8"))
        with self.assertRaises(P4RunnerError):
            resolve_authorization_package(consumed)


if __name__ == "__main__":
    unittest.main()
