"""CPU tests for the Phase 97 bounded chunked-prefill V10 authorization."""

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
    V9_OUTPUT_PATH,
    V10_AUTHORIZATION_PATH,
    V10_OUTPUT_PATH,
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
from validate_p4_b0_run_authorization_v10 import (  # noqa: E402
    B0RunAuthorizationV10Error,
    validate_authorization_v10,
)

AUTHORIZATION_PATH = PHASE_DIR / "data" / "p4" / "p4_b0_run_authorization_v10.json"
VALIDATION_PATH = (
    PHASE_DIR / "data" / "p4" / "p4_b0_run_authorization_v10_validation.json"
)
SCHEMA_PATH = PHASE_DIR / "schemas" / "p4_b0_run_authorization_v10.schema.json"
V9_FAILURE_PATH = PHASE_DIR / "data" / "p4" / "run_b0_value_screen_v8" / "failure.json"
PROBE_AUTHORIZATION_PATH = (
    PHASE_DIR / "data" / "p4" / "p4_b0_chunked_prefill_probe_authorization_v5.json"
)
PROBE_RESULT_PATH = (
    PHASE_DIR / "data" / "p4" / "run_b0_chunked_prefill_probe_v5" / "probe_result.json"
)


def _authorization() -> dict:
    return json.loads(AUTHORIZATION_PATH.read_text(encoding="utf-8"))


class P4B0RunAuthorizationV10Tests(unittest.TestCase):
    """Preserve the source-bound V10 cohort value-screen boundary."""

    def test_schema_is_valid(self) -> None:
        Draft202012Validator.check_schema(
            json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        )

    def test_checked_in_v10_package_is_run_ready_without_gpu_execution(self) -> None:
        result = validate_authorization_v10(_authorization())

        self.assertEqual(
            result,
            json.loads(VALIDATION_PATH.read_text(encoding="utf-8")),
        )
        self.assertEqual(result["status"], "pass")
        self.assertFalse(result["gpu_executed"])
        self.assertEqual(result["gpu_uuid"], GPU4_UUID)
        self.assertTrue(result["v10_execution_authorized"])
        self.assertTrue(result["value_screen_scoring_authorized"])
        self.assertEqual(
            result["chunked_prefill_probe_action_id"],
            "target-matching-k4",
        )
        self.assertEqual(
            result["chunked_prefill_probe_regimes"],
            ["R4", "R5", "R5cot"],
        )
        self.assertTrue(result["chunked_prefill_k4_gpu_probe_passed"])
        self.assertNotIn("chunked_prefill_cohort_gpu_proven", result)
        self.assertFalse(result["score_grants_authority"])
        self.assertFalse(result["p4a_engineering_authorized"])
        self.assertFalse((REPO_ROOT / V10_OUTPUT_PATH).exists())

    def test_effective_v10_uses_only_bounded_chunked_prefill(self) -> None:
        effective = resolve_authorization_package(_authorization())

        validate_execution_authority(effective)
        specs = build_boot_specs(effective, REPO_ROOT / V10_OUTPUT_PATH)
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
        specs = build_boot_specs(effective, REPO_ROOT / V10_OUTPUT_PATH)
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

    def test_v9_oom_and_narrow_v5_k4_probe_pass_are_preserved(self) -> None:
        failure = json.loads(V9_FAILURE_PATH.read_text(encoding="utf-8"))
        probe_authorization = json.loads(
            PROBE_AUTHORIZATION_PATH.read_text(encoding="utf-8")
        )
        probe = json.loads(PROBE_RESULT_PATH.read_text(encoding="utf-8"))

        self.assertEqual(failure["attempt"]["complete_captures_emitted"], 8)
        self.assertEqual(
            failure["diagnostic"]["scope"],
            "full_prefill_transient_activation_hbm_underbound",
        )
        self.assertFalse(failure["attempt"]["score_emitted"])
        self.assertEqual(
            probe_authorization["run_contract"]["action"]["action_id"],
            "target-matching-k4",
        )
        self.assertEqual(
            probe_authorization["run_contract"]["engine"]["gpu_memory_utilization"],
            SERVING_GPU_MEMORY_UTILIZATION,
        )
        self.assertEqual(probe["status"], "pass")
        self.assertFalse(probe["scored"])
        self.assertEqual(probe["action_id"], "target-matching-k4")
        self.assertEqual(probe["regimes"], ["R4", "R5", "R5cot"])
        self.assertTrue(
            all(value is False for value in probe["authorizations"].values())
        )
        self.assertEqual(len(probe["cohorts"]), 3)
        self.assertTrue(all(row["state"] == "complete" for row in probe["cohorts"]))

    def test_parent_execution_path_accepts_exact_v10_pair(self) -> None:
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
                REPO_ROOT / V10_AUTHORIZATION_PATH,
                {},
                REPO_ROOT / V10_OUTPUT_PATH,
            )

        self.assertEqual(
            events,
            ["authority", "sampler", "engine_core", "gpu", "prepare", "score"],
        )

    def test_child_execution_path_accepts_exact_v10_pair(self) -> None:
        child_spec = REPO_ROOT / V10_OUTPUT_PATH / "boot_specs" / "boot.json"
        args = Namespace(
            authorization=REPO_ROOT / V10_AUTHORIZATION_PATH,
            output_dir=REPO_ROOT / V10_OUTPUT_PATH,
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
            _reviewed_output_path(Path("/tmp/copied-v10-authorization.json"))
        with self.assertRaisesRegex(P4RunnerError, "reviewed create-new path"):
            execute_run(
                REPO_ROOT / V10_AUTHORIZATION_PATH,
                {},
                REPO_ROOT / V9_OUTPUT_PATH,
            )

    def test_rejects_source_hash_drift(self) -> None:
        authorization = _authorization()
        authorization["source_artifacts"]["matrix_runner"]["sha256"] = "0" * 64

        with self.assertRaisesRegex(B0RunAuthorizationV10Error, "matrix_runner"):
            validate_authorization_v10(authorization)

    def test_rejects_probe_result_drift(self) -> None:
        authorization = _authorization()
        authorization["chunked_prefill_probe"]["result"]["sha256"] = "0" * 64

        with self.assertRaisesRegex(B0RunAuthorizationV10Error, "probe binding"):
            validate_authorization_v10(authorization)

    def test_rejects_full_prefill_reenable(self) -> None:
        authorization = _authorization()
        authorization["chunked_prefill_contract"]["full_microbatch_prefill_allowed"] = (
            True
        )

        with self.assertRaises(B0RunAuthorizationV10Error):
            validate_authorization_v10(authorization)

    def test_rejects_changed_output_path(self) -> None:
        authorization = _authorization()
        authorization["run_contract"]["invocation"]["output_dir"] += "-other"

        with self.assertRaises(B0RunAuthorizationV10Error):
            validate_authorization_v10(authorization)

    def test_rejects_fallback_retry_and_p4a_authority(self) -> None:
        mutations = (
            ("fallback_gpu_authorized", True),
            ("retry_allowed", True),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                authorization = copy.deepcopy(_authorization())
                authorization["execution_policy"][field] = value
                with self.assertRaises(B0RunAuthorizationV10Error):
                    validate_authorization_v10(authorization)

        authorization = copy.deepcopy(_authorization())
        authorization["authorizations"]["p4a_engineering"] = True
        with self.assertRaises(B0RunAuthorizationV10Error):
            validate_authorization_v10(authorization)


if __name__ == "__main__":
    unittest.main()
