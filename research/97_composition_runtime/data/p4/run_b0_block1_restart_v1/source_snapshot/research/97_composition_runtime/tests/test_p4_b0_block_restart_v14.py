# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the Phase 97 single-block restart authorization."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
sys.path.insert(0, str(PHASE_DIR / "scripts"))

import run_p4_b0_value_screen as matrix  # noqa: E402
from validate_p4_b0_block_restart_authorization_v14 import (  # noqa: E402
    B0RunAuthorizationV14Error,
    validate_authorization_v14,
)

AUTHORIZATION_PATH = REPO_ROOT / matrix.V14_AUTHORIZATION_PATH


def _package() -> dict:
    with AUTHORIZATION_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


class SelectionIntegrityTests(unittest.TestCase):
    """The block choice must be declared, and not a choice about the outcome."""

    def test_package_validates(self) -> None:
        result = validate_authorization_v14(_package())
        self.assertEqual(result["status"], "pass")
        self.assertFalse(result["gpu_executed"])
        self.assertEqual(result["restart_block_id"], 1)
        self.assertEqual(result["rerun_capture_count"], 144)
        self.assertEqual(result["reused_capture_count"], 288)
        self.assertEqual(result["scored_capture_count"], 432)

    def test_selection_is_declared_before_the_rerun(self) -> None:
        selection = _package()["block_selection"]
        self.assertTrue(selection["declared_before_rerun"])
        self.assertTrue(selection["independent_of_certification_outcome"])
        self.assertEqual(selection["rule"], "highest_episode_rejection_count")

    def test_selected_block_has_the_highest_rejection_count(self) -> None:
        selection = _package()["block_selection"]
        counts = selection["rounds_below_95pct_floor_by_block"]
        selected = str(selection["selected_block_id"])
        self.assertTrue(
            all(counts[selected] >= v for k, v in counts.items() if k != selected)
        )

    def test_selection_counts_match_the_preserved_refusal_record(self) -> None:
        refusal = json.loads(
            (REPO_ROOT / matrix.V14_REFUSAL_PATH).read_text(encoding="utf-8")
        )
        self.assertEqual(
            refusal["episode_noise"]["rounds_below_95pct_floor_by_block"],
            _package()["block_selection"]["rounds_below_95pct_floor_by_block"],
        )

    def test_result_is_bound_before_it_exists(self) -> None:
        commitment = _package()["outcome_commitment"]
        self.assertTrue(commitment["result_used_regardless_of_certification"])
        self.assertTrue(commitment["rerun_until_pass_forbidden"])
        self.assertEqual(commitment["maximum_restarts_of_this_block"], 1)
        self.assertTrue(commitment["further_restart_requires_new_declared_rule"])

    def test_thresholds_and_cells_are_unchanged(self) -> None:
        commitment = _package()["outcome_commitment"]
        self.assertEqual(commitment["certification_bar_unchanged_fraction"], 0.02)
        self.assertEqual(commitment["episode_rule_unchanged_fraction"], 0.95)
        self.assertTrue(commitment["prompts_seeds_and_cells_identical_to_source_run"])

    def test_reused_blocks_are_not_remeasured(self) -> None:
        package = _package()
        self.assertFalse(package["claims"]["reused_blocks_remeasured"])
        self.assertEqual(package["execution_policy"]["reused_block_ids"], [2, 3])
        self.assertTrue(package["source_run"]["preserve_without_overwrite"])

    def test_a_further_restart_is_not_authorized(self) -> None:
        self.assertFalse(_package()["authorizations"]["additional_block_restart"])

    def test_downstream_authority_stays_false(self) -> None:
        package = _package()
        self.assertFalse(package["authorizations"]["p4a_engineering"])
        self.assertFalse(package["authorizations"]["action_admission"])
        self.assertFalse(package["authorizations"]["production_value_claim"])
        self.assertFalse(package["execution_policy"]["score_grants_authority"])


class FailClosedTests(unittest.TestCase):
    """Any weakening of the declared rule must be refused."""

    def _rejects(self, mutate) -> None:
        package = copy.deepcopy(_package())
        mutate(package)
        with self.assertRaises(B0RunAuthorizationV14Error):
            validate_authorization_v14(package)

    def test_selecting_a_quieter_block_is_rejected(self) -> None:
        def mutate(package: dict) -> None:
            package["block_selection"]["selected_block_id"] = 3

        self._rejects(mutate)

    def test_relaxing_the_certification_bar_is_rejected(self) -> None:
        def mutate(package: dict) -> None:
            package["outcome_commitment"]["certification_bar_unchanged_fraction"] = 0.05

        self._rejects(mutate)

    def test_relaxing_the_episode_rule_is_rejected(self) -> None:
        def mutate(package: dict) -> None:
            package["outcome_commitment"]["episode_rule_unchanged_fraction"] = 0.90

        self._rejects(mutate)

    def test_allowing_rerun_until_pass_is_rejected(self) -> None:
        def mutate(package: dict) -> None:
            package["outcome_commitment"]["rerun_until_pass_forbidden"] = False

        self._rejects(mutate)

    def test_discarding_the_result_on_failure_is_rejected(self) -> None:
        def mutate(package: dict) -> None:
            package["outcome_commitment"]["result_used_regardless_of_certification"] = (
                False
            )

        self._rejects(mutate)

    def test_more_than_one_restart_is_rejected(self) -> None:
        def mutate(package: dict) -> None:
            package["outcome_commitment"]["maximum_restarts_of_this_block"] = 3

        self._rejects(mutate)

    def test_fabricated_rejection_counts_are_rejected(self) -> None:
        def mutate(package: dict) -> None:
            package["block_selection"]["rounds_below_95pct_floor_by_block"] = {
                "1": 99,
                "2": 1,
                "3": 0,
            }

        self._rejects(mutate)

    def test_source_hash_drift_is_rejected(self) -> None:
        def mutate(package: dict) -> None:
            role = next(iter(package["source_artifacts"]))
            package["source_artifacts"][role]["sha256"] = "0" * 64

        self._rejects(mutate)

    def test_overwriting_the_source_run_is_rejected(self) -> None:
        def mutate(package: dict) -> None:
            package["run_contract"]["invocation"]["output_dir"] = (
                matrix.V14_SOURCE_RUN_PATH
            )

        self._rejects(mutate)

    def test_remeasuring_the_reused_blocks_is_rejected(self) -> None:
        def mutate(package: dict) -> None:
            package["claims"]["reused_blocks_remeasured"] = True

        self._rejects(mutate)


if __name__ == "__main__":
    unittest.main()
