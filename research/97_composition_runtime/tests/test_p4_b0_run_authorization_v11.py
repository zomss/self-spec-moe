"""CPU tests for the Phase 97 repaired-prefill V11 authorization."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from jsonschema import Draft202012Validator

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
sys.path.insert(0, str(PHASE_DIR / "scripts"))

from run_p4_b0_value_screen import (  # noqa: E402
    GPU4_UUID,
    SERVING_EFFECTIVE_MAX_NUM_SCHEDULED_TOKENS,
    SERVING_GPU_MEMORY_UTILIZATION,
    SERVING_MAX_NUM_BATCHED_TOKENS,
    V10_OUTPUT_PATH,
    V11_AUTHORIZATION_PATH,
    V11_OUTPUT_PATH,
    P4RunnerError,
    _capture_cohort_metadata,
    _reviewed_output_path,
    build_boot_specs,
    execute_run,
    resolve_authorization_package,
    validate_execution_authority,
)
from run_p4_b0_value_screen import (  # noqa: E402
    main as runner_main,
)
from validate_p4_b0_run_authorization_v11 import (  # noqa: E402
    B0RunAuthorizationV11Error,
    validate_authorization_v11,
)

AUTHORIZATION_PATH = PHASE_DIR / "data" / "p4" / "p4_b0_run_authorization_v11.json"
VALIDATION_PATH = (
    PHASE_DIR / "data" / "p4" / "p4_b0_run_authorization_v11_validation.json"
)
SCHEMA_PATH = PHASE_DIR / "schemas" / "p4_b0_run_authorization_v11.schema.json"
V10_FAILURE_PATH = PHASE_DIR / "data" / "p4" / "run_b0_value_screen_v9" / "failure.json"
V10_MANIFEST_PATH = (
    PHASE_DIR / "data" / "p4" / "run_b0_value_screen_v9" / "capture_manifest.json"
)
REPAIR_AUDIT_PATH = (
    PHASE_DIR
    / "data"
    / "p4"
    / "p4_b0_variable_prefill_repair_validation_attempt_v1.json"
)
REPAIR_CASE_ROOT = (
    PHASE_DIR / "data" / "p4" / "run_b0_variable_prefill_repair_validation_v1"
)


def _authorization() -> dict:
    return json.loads(AUTHORIZATION_PATH.read_text(encoding="utf-8"))


class P4B0RunAuthorizationV11Tests(unittest.TestCase):
    """Preserve the source-bound V11 value-screen boundary."""

    def test_schema_is_valid(self) -> None:
        Draft202012Validator.check_schema(
            json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        )

    def test_checked_in_v11_package_is_run_ready_without_gpu_execution(self) -> None:
        result = validate_authorization_v11(_authorization())

        self.assertEqual(
            result,
            json.loads(VALIDATION_PATH.read_text(encoding="utf-8")),
        )
        self.assertEqual(result["status"], "pass")
        self.assertFalse(result["gpu_executed"])
        self.assertEqual(result["gpu_uuid"], GPU4_UUID)
        self.assertTrue(result["v11_execution_authorized"])
        self.assertTrue(result["value_screen_scoring_authorized"])
        self.assertFalse(result["prior_outputs_reusable"])
        self.assertTrue(result["isolated_r8_gpu_case_passed"])
        self.assertTrue(result["r5cot_to_r8_gpu_case_passed"])
        self.assertFalse(result["repair_parent_aggregate_passed"])
        self.assertEqual(
            result["repair_parent_rejection_classification"],
            "observer_cardinality_bug",
        )
        self.assertFalse(result["score_grants_authority"])
        self.assertFalse(result["p4a_engineering_authorized"])
        self.assertFalse((REPO_ROOT / V11_OUTPUT_PATH).exists())

    def test_effective_v11_uses_only_bounded_chunked_prefill(self) -> None:
        effective = resolve_authorization_package(_authorization())

        validate_execution_authority(effective)
        specs = build_boot_specs(effective, REPO_ROOT / V11_OUTPUT_PATH)
        self.assertEqual(len(specs), 9)
        for spec in specs:
            self.assertEqual(
                spec["engine"]["max_num_batched_tokens"],
                SERVING_MAX_NUM_BATCHED_TOKENS,
            )
            self.assertEqual(
                spec["engine"]["gpu_memory_utilization"],
                SERVING_GPU_MEMORY_UTILIZATION,
            )
            self.assertNotIn("full_prefill_budget", spec)
            self.assertEqual(
                spec["chunked_prefill_budget"]["effective_scheduler_token_budget"],
                SERVING_EFFECTIVE_MAX_NUM_SCHEDULED_TOKENS,
            )
            self.assertFalse(
                spec["chunked_prefill_budget"]["full_microbatch_prefill_allowed"]
            )

    def test_every_prompt_slice_builds_the_exact_cohort_marker(self) -> None:
        effective = resolve_authorization_package(_authorization())
        specs = build_boot_specs(effective, REPO_ROOT / V11_OUTPUT_PATH)
        cohort_count = 0
        for spec in specs:
            for cell in spec["plan"]["cells"]:
                prompt_ids = cell["generation"]["prompt_record_ids"]
                batch = cell["generation"]["batch"]
                for start in range(0, len(prompt_ids), batch):
                    members = prompt_ids[start : start + batch]
                    marker = _capture_cohort_metadata(cell, members)
                    self.assertEqual(
                        marker["contract_id"],
                        "p4-capture-cohort-barrier-v1",
                    )
                    self.assertEqual(marker["frozen_request_ids"], members)
                    self.assertEqual(marker["action_id"], cell["matrix"]["action_id"])
                    cohort_count += 1
        self.assertEqual(cohort_count, 3384)

    def test_consumed_v10_and_case_level_repair_evidence_are_preserved(self) -> None:
        failure = json.loads(V10_FAILURE_PATH.read_text(encoding="utf-8"))
        manifest = json.loads(V10_MANIFEST_PATH.read_text(encoding="utf-8"))
        audit = json.loads(REPAIR_AUDIT_PATH.read_text(encoding="utf-8"))

        self.assertEqual(failure["attempt"]["complete_captures_emitted"], 72)
        self.assertFalse(failure["attempt"]["score_emitted"])
        self.assertTrue(failure["disposition"]["v10_consumed"])
        self.assertEqual(manifest["counts"]["complete_captures"], 72)
        self.assertTrue(
            manifest["invariants"]["preserve_without_overwrite_resume_or_reuse"]
        )
        self.assertTrue(audit["disposition"]["gpu_repair_validation_passed"])
        self.assertFalse(audit["disposition"]["parent_aggregate_passed"])
        self.assertFalse(audit["disposition"]["output_reuse_allowed"])
        self.assertEqual(
            audit["parent_rejection"]["classification"],
            "observer_cardinality_bug",
        )

        expected = {
            "isolated-r8": (1, 610, 612),
            "r5cot-to-r8": (9, 1273, 1289),
        }
        for case_id, (armed_count, event_count, trace_count) in expected.items():
            case = json.loads(
                (REPAIR_CASE_ROOT / case_id / "case_result.json").read_text(
                    encoding="utf-8"
                )
            )
            armed = case["instrumentation"]["armed_prefill_evidence"]
            exact_r8 = [
                row
                for row in armed
                if row["draft_step0_query_width"] is None
                and row["draft_step0_num_tokens"] == 2116
                and row["draft_step0_batch_size"] == 16
                and row["draft_output_shape"] == [16, 4]
            ]
            self.assertEqual(len(armed), armed_count)
            self.assertEqual(len(exact_r8), 1)
            self.assertEqual(case["recorder"]["event_count"], event_count)
            self.assertEqual(case["trace"]["record_count"], trace_count)
            self.assertIsNone(case["primary_exception"])
            self.assertIsNone(case["shutdown_exception"])

    def test_parent_execution_path_accepts_exact_v11_pair(self) -> None:
        events: list[str] = []
        with (
            mock.patch(
                "run_p4_b0_value_screen.validate_execution_authority",
                side_effect=lambda _: events.append("authority"),
            ),
            mock.patch(
                "run_p4_b0_value_screen._boot_child_environment",
                return_value={},
            ),
            mock.patch(
                "run_p4_b0_value_screen._preflight_native_sampler",
                side_effect=lambda _: events.append("sampler"),
            ),
            mock.patch(
                "run_p4_b0_value_screen._preflight_inprocess_engine_core",
                side_effect=lambda _: events.append("engine_core"),
            ),
            mock.patch(
                "run_p4_b0_value_screen._preflight_gpu4_identity_and_idle",
                side_effect=lambda: events.append("gpu"),
            ),
            mock.patch(
                "run_p4_b0_value_screen.prepare_run",
                side_effect=lambda *_: events.append("prepare") or [],
            ),
            mock.patch(
                "run_p4_b0_value_screen._adapt_and_score",
                side_effect=lambda _: events.append("score"),
            ),
        ):
            execute_run(
                REPO_ROOT / V11_AUTHORIZATION_PATH,
                {},
                REPO_ROOT / V11_OUTPUT_PATH,
            )

        self.assertEqual(
            events,
            ["authority", "sampler", "engine_core", "gpu", "prepare", "score"],
        )

    def test_child_execution_path_accepts_exact_v11_pair(self) -> None:
        child_spec = REPO_ROOT / V11_OUTPUT_PATH / "boot_specs" / "boot.json"
        args = Namespace(
            authorization=REPO_ROOT / V11_AUTHORIZATION_PATH,
            output_dir=REPO_ROOT / V11_OUTPUT_PATH,
            prepare_only=False,
            child_spec=child_spec,
        )
        with (
            mock.patch("run_p4_b0_value_screen.parse_args", return_value=args),
            mock.patch("run_p4_b0_value_screen._load_json", return_value={}),
            mock.patch(
                "run_p4_b0_value_screen.resolve_authorization_package",
                return_value={},
            ),
            mock.patch("run_p4_b0_value_screen.validate_execution_authority"),
            mock.patch("run_p4_b0_value_screen.run_boot_child") as run_child,
        ):
            self.assertEqual(runner_main(), 0)

        run_child.assert_called_once_with(child_spec)

    def test_dispatch_rejects_unknown_and_cross_paired_paths(self) -> None:
        with self.assertRaisesRegex(P4RunnerError, "reviewed authorization path"):
            _reviewed_output_path(Path("/tmp/copied-v11-authorization.json"))
        with self.assertRaisesRegex(P4RunnerError, "reviewed create-new path"):
            execute_run(
                REPO_ROOT / V11_AUTHORIZATION_PATH,
                {},
                REPO_ROOT / V10_OUTPUT_PATH,
            )

    def test_rejects_source_hash_drift(self) -> None:
        authorization = _authorization()
        authorization["source_artifacts"]["matrix_runner"]["sha256"] = "0" * 64

        with self.assertRaisesRegex(B0RunAuthorizationV11Error, "matrix_runner"):
            validate_authorization_v11(authorization)

    def test_rejects_consumed_attempt_drift(self) -> None:
        authorization = _authorization()
        authorization["consumed_attempt"]["failure"]["sha256"] = "0" * 64

        with self.assertRaises(B0RunAuthorizationV11Error):
            validate_authorization_v11(authorization)

    def test_rejects_repair_audit_drift(self) -> None:
        authorization = _authorization()
        repair = authorization["variable_prefill_repair_validation"]
        repair["audit"]["sha256"] = "0" * 64

        with self.assertRaisesRegex(B0RunAuthorizationV11Error, "repair evidence"):
            validate_authorization_v11(authorization)

    def test_rejects_full_prefill_reenable(self) -> None:
        authorization = _authorization()
        authorization["chunked_prefill_contract"]["full_microbatch_prefill_allowed"] = (
            True
        )

        with self.assertRaises(B0RunAuthorizationV11Error):
            validate_authorization_v11(authorization)

    def test_rejects_changed_output_path(self) -> None:
        authorization = _authorization()
        authorization["run_contract"]["invocation"]["output_dir"] += "-other"

        with self.assertRaises(B0RunAuthorizationV11Error):
            validate_authorization_v11(authorization)

    def test_rejects_reuse_retry_fallback_and_p4a_authority(self) -> None:
        mutations = (
            ("fallback_gpu_authorized", True),
            ("prior_output_reuse_allowed", True),
            ("repair_validation_output_reuse_allowed", True),
            ("retry_allowed", True),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                authorization = copy.deepcopy(_authorization())
                authorization["execution_policy"][field] = value
                with self.assertRaises(B0RunAuthorizationV11Error):
                    validate_authorization_v11(authorization)

        authorization = copy.deepcopy(_authorization())
        authorization["authorizations"]["p4a_engineering"] = True
        with self.assertRaises(B0RunAuthorizationV11Error):
            validate_authorization_v11(authorization)


if __name__ == "__main__":
    unittest.main()
