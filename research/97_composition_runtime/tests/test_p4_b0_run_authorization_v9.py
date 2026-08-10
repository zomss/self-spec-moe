"""CPU tests for the Phase 97 B0 launch-dispatch V9 authorization."""

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
    V8_OUTPUT_PATH,
    V9_AUTHORIZATION_PATH,
    V9_OUTPUT_PATH,
    P4RunnerError,
    _reviewed_output_path,
    _total_output_tokens,
    build_boot_specs,
    execute_run,
    resolve_authorization_package,
    validate_execution_authority,
)
from run_p4_b0_value_screen import (  # noqa: E402
    main as runner_main,
)
from validate_p4_b0_run_authorization_v9 import (  # noqa: E402
    B0RunAuthorizationV9Error,
    validate_authorization_v9,
)

AUTHORIZATION_PATH = PHASE_DIR / "data" / "p4" / "p4_b0_run_authorization_v9.json"
VALIDATION_PATH = (
    PHASE_DIR / "data" / "p4" / "p4_b0_run_authorization_v9_validation.json"
)
SCHEMA_PATH = PHASE_DIR / "schemas" / "p4_b0_run_authorization_v9.schema.json"
V8_FAILURE_PATH = PHASE_DIR / "data" / "p4" / "run_b0_value_screen_v7" / "failure.json"


def _authorization() -> dict:
    return json.loads(AUTHORIZATION_PATH.read_text(encoding="utf-8"))


class P4B0RunAuthorizationV9Tests(unittest.TestCase):
    """Preserve the narrow, source-bound V9 execution boundary."""

    def test_schema_is_valid(self) -> None:
        Draft202012Validator.check_schema(
            json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        )

    def test_checked_in_v9_package_is_run_ready_without_gpu_execution(self) -> None:
        result = validate_authorization_v9(_authorization())

        self.assertEqual(
            result,
            json.loads(VALIDATION_PATH.read_text(encoding="utf-8")),
        )
        self.assertEqual(result["status"], "pass")
        self.assertFalse(result["gpu_executed"])
        self.assertTrue(result["launch_dispatch_repair_tested"])
        self.assertTrue(result["parent_dispatch_tested"])
        self.assertTrue(result["child_dispatch_tested"])
        self.assertEqual(result["gpu_uuid"], GPU4_UUID)
        self.assertTrue(result["v9_execution_authorized"])
        self.assertFalse(result["p4a_engineering_authorized"])
        self.assertFalse((REPO_ROOT / V9_OUTPUT_PATH).exists())

    def test_effective_v9_retains_v8_decode_work_contract(self) -> None:
        effective = resolve_authorization_package(_authorization())

        validate_execution_authority(effective)
        specs = build_boot_specs(effective, REPO_ROOT / V9_OUTPUT_PATH)
        for spec in specs:
            for cell in spec["plan"]["cells"]:
                measured = cell["generation"]["max_output_tokens"]
                self.assertEqual(_total_output_tokens(cell), measured + 1)

    def test_v8_dispatch_refusal_is_preserved_without_retry(self) -> None:
        failure = json.loads(V8_FAILURE_PATH.read_text(encoding="utf-8"))

        self.assertEqual(
            failure["authorization"]["package_id"],
            "p4-b0-value-screen-run-authorization-v8",
        )
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
        self.assertFalse(failure["output"]["runner_output_created"])

    def test_parent_execution_path_accepts_exact_v9_pair(self) -> None:
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
                REPO_ROOT / V9_AUTHORIZATION_PATH,
                {},
                REPO_ROOT / V9_OUTPUT_PATH,
            )

        self.assertEqual(
            events,
            ["authority", "sampler", "engine_core", "gpu", "prepare", "score"],
        )

    def test_child_execution_path_accepts_exact_v9_pair(self) -> None:
        child_spec = REPO_ROOT / V9_OUTPUT_PATH / "boot_specs" / "boot.json"
        args = Namespace(
            authorization=REPO_ROOT / V9_AUTHORIZATION_PATH,
            output_dir=REPO_ROOT / V9_OUTPUT_PATH,
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
            _reviewed_output_path(Path("/tmp/copied-v9-authorization.json"))
        with self.assertRaisesRegex(P4RunnerError, "reviewed create-new path"):
            execute_run(
                REPO_ROOT / V9_AUTHORIZATION_PATH,
                {},
                REPO_ROOT / V8_OUTPUT_PATH,
            )

    def test_rejects_source_hash_drift(self) -> None:
        authorization = _authorization()
        authorization["source_artifacts"]["matrix_runner"]["sha256"] = "0" * 64

        with self.assertRaisesRegex(B0RunAuthorizationV9Error, "matrix_runner"):
            validate_authorization_v9(authorization)

    def test_rejects_dispatch_contract_drift(self) -> None:
        authorization = _authorization()
        authorization["launch_dispatch_repair"]["child_dispatch_tested"] = False

        with self.assertRaises(B0RunAuthorizationV9Error):
            validate_authorization_v9(authorization)

    def test_rejects_changed_output_path(self) -> None:
        authorization = _authorization()
        authorization["run_contract"]["invocation"]["output_dir"] += "-other"

        with self.assertRaises(B0RunAuthorizationV9Error):
            validate_authorization_v9(authorization)

    def test_rejects_fallback_retry_and_p4a_authority(self) -> None:
        mutations = (
            ("fallback_gpu_authorized", True),
            ("retry_allowed", True),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                authorization = copy.deepcopy(_authorization())
                authorization["execution_policy"][field] = value
                with self.assertRaises(B0RunAuthorizationV9Error):
                    validate_authorization_v9(authorization)

        authorization = copy.deepcopy(_authorization())
        authorization["authorizations"]["p4a_engineering"] = True
        with self.assertRaises(B0RunAuthorizationV9Error):
            validate_authorization_v9(authorization)


if __name__ == "__main__":
    unittest.main()
