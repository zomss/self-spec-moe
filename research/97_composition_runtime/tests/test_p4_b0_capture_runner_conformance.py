"""CPU tests for the Phase 97 B0 capture-runner conformance package."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
sys.path.insert(0, str(PHASE_DIR / "scripts"))

from validate_p4_b0_capture_runner_conformance import (  # noqa: E402
    P4ConformanceError,
    validate_conformance,
)

CONFORMANCE_PATH = PHASE_DIR / "data" / "p4" / "p4_b0_capture_runner_conformance.json"
SCHEMA_PATH = PHASE_DIR / "schemas" / "p4_b0_capture_runner_conformance.schema.json"


def _conformance() -> dict:
    return json.loads(CONFORMANCE_PATH.read_text(encoding="utf-8"))


def _conformance_with_retry_sources() -> dict:
    conformance = _conformance()
    for artifact in conformance["source_artifacts"].values():
        artifact["sha256"] = hashlib.sha256(
            (REPO_ROOT / artifact["path"]).read_bytes()
        ).hexdigest()
    return conformance


class P4B0CaptureRunnerConformanceTests(unittest.TestCase):
    """Verify the six-check pass without GPU or downstream authority."""

    def test_schema_is_valid(self) -> None:
        Draft202012Validator.check_schema(
            json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        )

    def test_conformance_contracts_pass_with_retry_source_bindings(self) -> None:
        result = validate_conformance(_conformance_with_retry_sources())

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["implementation_checks_passed"], 6)
        self.assertEqual(result["physical_boot_count"], 9)
        self.assertEqual(result["cells_per_boot"], 48)
        self.assertEqual(result["capture_count"], 432)
        self.assertEqual(result["minimum_shared_kv_blocks"], 21682)
        self.assertEqual(result["gpu_commands_run"], 0)
        self.assertFalse(result["gpu_measurement_authorized"])
        self.assertFalse(result["p4a_engineering_authorized"])

    def test_rejects_source_hash_drift(self) -> None:
        conformance = _conformance_with_retry_sources()
        conformance["source_artifacts"]["matrix_runner"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(P4ConformanceError, "source hash mismatch"):
            validate_conformance(conformance)

    def test_rejects_missing_conformance_check(self) -> None:
        conformance = _conformance_with_retry_sources()
        conformance["implementation_checks"][0]["code"] = conformance[
            "implementation_checks"
        ][1]["code"]
        with self.assertRaisesRegex(P4ConformanceError, "missing, repeated"):
            validate_conformance(conformance)

    def test_rejects_gpu_authority(self) -> None:
        conformance = copy.deepcopy(_conformance_with_retry_sources())
        conformance["authorizations"]["gpu_measurement"] = True
        with self.assertRaises(P4ConformanceError):
            validate_conformance(conformance)


if __name__ == "__main__":
    unittest.main()
