"""CPU-only tests for the exact Phase 97 P4 prompt-token package."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR / "scripts"))

from validate_p4_prompt_manifest import (  # noqa: E402
    PromptManifestError,
    validate_manifest,
)

MANIFEST_PATH = PHASE_DIR / "data" / "p4" / "p4_b0_prompt_manifest.json"
VALIDATION_PATH = PHASE_DIR / "data" / "p4" / "p4_b0_prompt_manifest_validation.json"
BUNDLE_PATH = PHASE_DIR / "data" / "p4" / "p4_b0_prompt_tokens.jsonl.gz"


def valid_manifest() -> dict:
    """Return an independent copy of the checked-in prompt manifest."""
    return json.loads(MANIFEST_PATH.read_text())


class PromptManifestPositiveTests(unittest.TestCase):
    """Prove that the exact bundle is frozen without granting run authority."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.result = validate_manifest(valid_manifest())

    def test_checked_manifest_and_bundle_are_valid(self) -> None:
        self.assertEqual(self.result["status"], "pass")
        self.assertTrue(self.result["exact_prompt_manifest_frozen"])
        self.assertEqual(self.result["record_count"], 384)
        self.assertFalse(self.result["gpu_measurement_authorized"])
        self.assertFalse(self.result["p4a_engineering_authorized"])

    def test_exact_six_regime_plan_is_frozen(self) -> None:
        manifest = valid_manifest()

        self.assertEqual(
            manifest["prompt_plan"]["regime_order"],
            ["R4", "R5", "R5cot", "R8", "R1", "R6"],
        )
        self.assertEqual(manifest["prompt_plan"]["content_seeds"], [0, 1])
        self.assertEqual(manifest["prompt_plan"]["generation_seed"], 0)
        self.assertTrue(manifest["prompt_plan"]["ignore_eos"])

    def test_bundle_has_deterministic_gzip_header(self) -> None:
        compressed = BUNDLE_PATH.read_bytes()

        self.assertEqual(compressed[:4], b"\x1f\x8b\x08\x00")
        self.assertEqual(compressed[4:8], b"\x00\x00\x00\x00")

    def test_checked_validation_output_is_current(self) -> None:
        self.assertEqual(self.result, json.loads(VALIDATION_PATH.read_text()))


class PromptManifestFailClosedTests(unittest.TestCase):
    """Reject provenance, token-index, and authority drift."""

    def test_rejects_source_hash_drift(self) -> None:
        manifest = valid_manifest()
        manifest["source_artifacts"]["regime_loader"]["sha256"] = "0" * 64

        with self.assertRaisesRegex(PromptManifestError, "source hash mismatch"):
            validate_manifest(manifest)

    def test_rejects_tokenizer_revision_drift(self) -> None:
        manifest = valid_manifest()
        manifest["tokenizer"]["revision"] = "0" * 40

        with self.assertRaisesRegex(PromptManifestError, "tokenizer revision"):
            validate_manifest(manifest)

    def test_rejects_dataset_revision_drift(self) -> None:
        manifest = valid_manifest()
        manifest["datasets"][0]["revision"] = "0" * 40

        with self.assertRaisesRegex(PromptManifestError, "dataset provenance"):
            validate_manifest(manifest)

    def test_rejects_bundle_hash_drift(self) -> None:
        manifest = valid_manifest()
        manifest["bundle"]["sha256"] = "0" * 64

        with self.assertRaisesRegex(PromptManifestError, "bundle hash mismatch"):
            validate_manifest(manifest)

    def test_rejects_per_prompt_token_count_drift(self) -> None:
        manifest = valid_manifest()
        manifest["prompts"][0]["token_count"] += 1

        with self.assertRaisesRegex(PromptManifestError, "per-prompt manifest"):
            validate_manifest(manifest)

    def test_rejects_group_summary_drift(self) -> None:
        manifest = valid_manifest()
        manifest["groups"][0]["total_prompt_tokens"] += 1

        with self.assertRaisesRegex(PromptManifestError, "group summaries"):
            validate_manifest(manifest)

    def test_rejects_generation_seed_drift(self) -> None:
        manifest = valid_manifest()
        manifest["prompt_plan"]["generation_seed"] = 1

        with self.assertRaises(PromptManifestError):
            validate_manifest(manifest)

    def test_rejects_missing_prompt_index_row(self) -> None:
        manifest = valid_manifest()
        manifest["prompts"] = copy.deepcopy(manifest["prompts"][:-1])

        with self.assertRaises(PromptManifestError):
            validate_manifest(manifest)

    def test_rejects_downstream_authority(self) -> None:
        manifest = valid_manifest()
        manifest["authorizations"]["gpu_measurement"] = True

        with self.assertRaises(PromptManifestError):
            validate_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
