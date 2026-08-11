# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the Phase 97 block-parallel two-lane V13 authorization."""

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
from validate_p4_b0_run_authorization_v13 import (  # noqa: E402
    B0RunAuthorizationV13Error,
    validate_authorization_v13,
)

AUTHORIZATION_PATH = REPO_ROOT / matrix.V13_AUTHORIZATION_PATH


def _package() -> dict:
    with AUTHORIZATION_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


class RegisteredPackageTests(unittest.TestCase):
    """The checked-in V13 package must validate against every frozen source."""

    def test_package_validates(self) -> None:
        result = validate_authorization_v13(_package())
        self.assertEqual(result["status"], "pass")
        self.assertFalse(result["gpu_executed"])
        self.assertEqual(result["physical_boot_count"], 9)
        self.assertEqual(result["capture_count"], 432)
        self.assertEqual(result["lane_count"], 2)

    def test_package_grants_no_downstream_authority(self) -> None:
        package = _package()
        self.assertFalse(package["authorizations"]["p4a_engineering"])
        self.assertFalse(package["authorizations"]["action_admission"])
        self.assertFalse(package["authorizations"]["production_value_claim"])
        self.assertFalse(package["execution_policy"]["score_grants_authority"])
        self.assertFalse(package["next_artifact"]["may_authorize_p4a"])
        self.assertFalse(package["next_artifact"]["may_admit_action"])

    def test_package_claims_no_contention_bound(self) -> None:
        package = _package()
        self.assertFalse(package["claims"]["contention_bound_measured"])
        self.assertFalse(
            package["contention_probe_disposition"]["dual_gpu_authority_from_probe"]
        )
        self.assertTrue(package["claims"]["lane_precedent_is_prior_phase_read_only"])

    def test_package_forbids_reusing_the_interrupted_attempt(self) -> None:
        package = _package()
        self.assertFalse(package["execution_policy"]["prior_output_reuse_allowed"])
        self.assertTrue(package["consumed_attempt"]["preserve_without_resume_or_reuse"])
        self.assertEqual(package["consumed_attempt"]["complete_capture_count"], 0)
        self.assertFalse(package["consumed_attempt"]["score_emitted"])

    def test_registered_output_is_fresh(self) -> None:
        self.assertEqual(
            matrix.V13_OUTPUT_PATH,
            "research/97_composition_runtime/data/p4/run_b0_value_screen_v12",
        )
        self.assertNotEqual(matrix.V13_OUTPUT_PATH, matrix.V12_OUTPUT_PATH)

    def test_lane_assignment_matches_the_runner(self) -> None:
        package = _package()
        self.assertEqual(
            package["execution_policy"]["lane_assignment"], matrix.lane_assignment()
        )


class FailClosedTests(unittest.TestCase):
    """Every registered contract must fail closed when it drifts."""

    def _rejects(self, mutate) -> None:
        package = copy.deepcopy(_package())
        mutate(package)
        with self.assertRaises(B0RunAuthorizationV13Error):
            validate_authorization_v13(package)

    def test_source_hash_drift_is_rejected(self) -> None:
        def mutate(package: dict) -> None:
            role = next(iter(package["source_artifacts"]))
            package["source_artifacts"][role]["sha256"] = "0" * 64

        self._rejects(mutate)

    def test_a_split_block_is_rejected(self) -> None:
        def mutate(package: dict) -> None:
            lanes = package["execution_policy"]["lane_assignment"]
            lanes[0]["block_ids"] = [1]

        self._rejects(mutate)

    def test_a_third_lane_is_rejected(self) -> None:
        def mutate(package: dict) -> None:
            lanes = package["execution_policy"]["lane_assignment"]
            lanes.append(copy.deepcopy(lanes[0]))

        self._rejects(mutate)

    def test_a_shared_gpu_is_rejected(self) -> None:
        def mutate(package: dict) -> None:
            lanes = package["execution_policy"]["lane_assignment"]
            lanes[1]["physical_gpu_index"] = lanes[0]["physical_gpu_index"]

        self._rejects(mutate)

    def test_dropping_the_source_snapshot_is_rejected(self) -> None:
        def mutate(package: dict) -> None:
            package["execution_policy"]["source_snapshot_required"] = False

        self._rejects(mutate)

    def test_enabling_retry_is_rejected(self) -> None:
        def mutate(package: dict) -> None:
            package["execution_policy"]["retry_allowed"] = True

        self._rejects(mutate)

    def test_reusing_the_interrupted_output_is_rejected(self) -> None:
        def mutate(package: dict) -> None:
            package["execution_policy"]["prior_output_reuse_allowed"] = True

        self._rejects(mutate)

    def test_claiming_a_contention_bound_is_rejected(self) -> None:
        def mutate(package: dict) -> None:
            package["contention_probe_disposition"][
                "measured_contention_bound_available"
            ] = True

        self._rejects(mutate)

    def test_admitting_an_action_is_rejected(self) -> None:
        def mutate(package: dict) -> None:
            package["authorizations"]["action_admission"] = True

        self._rejects(mutate)

    def test_reenabling_full_prefill_geometry_is_rejected(self) -> None:
        def mutate(package: dict) -> None:
            package["execution_policy"]["max_num_batched_tokens"] = 114688

        self._rejects(mutate)

    def test_a_foreign_output_directory_is_rejected(self) -> None:
        def mutate(package: dict) -> None:
            package["run_contract"]["invocation"]["output_dir"] = matrix.V12_OUTPUT_PATH

        self._rejects(mutate)

    def test_a_wrong_package_id_is_rejected(self) -> None:
        def mutate(package: dict) -> None:
            package["package_id"] = matrix.V12_PACKAGE_ID

        self._rejects(mutate)


if __name__ == "__main__":
    unittest.main()
