# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the non-scored CPU-pinning probe."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
sys.path.insert(0, str(PHASE_DIR / "scripts"))

import run_p4_b0_pinning_probe as probe  # noqa: E402
import run_p4_b0_value_screen as matrix  # noqa: E402


def _package() -> dict:
    path = REPO_ROOT / probe.PROBE_AUTHORIZATION_PATH
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


class ProbeScopeTests(unittest.TestCase):
    """The probe must target exactly the failing cells and claim nothing."""

    def test_package_validates(self) -> None:
        probe.validate_authorization(_package())

    def test_targets_are_the_four_cells_that_exceeded_five_percent(self) -> None:
        keys = {
            (c["action_id"], c["regime_id"], c["content_seed"])
            for c in probe.TARGET_CELLS
        }
        self.assertEqual(keys, set(probe.V13_WITHIN_CELL_SPREAD))
        self.assertEqual(len(keys), 4)
        for spread in probe.V13_WITHIN_CELL_SPREAD.values():
            self.assertGreater(spread, 0.05)

    def test_no_off_cell_is_targeted(self) -> None:
        self.assertNotIn("off", {c["action_id"] for c in probe.TARGET_CELLS})

    def test_probe_is_non_scored_and_grants_nothing(self) -> None:
        package = _package()
        auth = package["authorizations"]
        self.assertTrue(auth["pinning_probe_execution"])
        self.assertFalse(auth["value_screen_execution"])
        self.assertFalse(auth["value_screen_scoring"])
        self.assertFalse(auth["p4a_engineering"])
        self.assertFalse(auth["action_admission"])
        self.assertFalse(auth["production_value_claim"])
        self.assertFalse(package["next_artifact"]["scored"])
        self.assertFalse(package["next_artifact"]["may_authorize_rescreen"])

    def test_source_run_is_not_remeasured(self) -> None:
        self.assertFalse(_package()["comparison"]["source_run_remeasured"])

    def test_probe_runs_on_the_reserved_lane(self) -> None:
        lane = _package()["execution_policy"]["lane"]
        self.assertEqual(lane, matrix.lane_for_block(probe.PROBE_BLOCK_ID))
        self.assertEqual(lane["cpu_affinity"], matrix.LANE_A_CPUS)

    def test_lean_server_is_left_running(self) -> None:
        """The mitigation must be tested against the real disturbance."""
        self.assertTrue(_package()["hypothesis"]["lean_server_left_running"])


class ProbeFailClosedTests(unittest.TestCase):
    """Any drift or scope inflation must be refused."""

    def _rejects(self, mutate) -> None:
        package = copy.deepcopy(_package())
        mutate(package)
        with self.assertRaises(probe.PinningProbeError):
            probe.validate_authorization(package)

    def test_source_hash_drift_is_rejected(self) -> None:
        def mutate(p: dict) -> None:
            role = next(iter(p["source_artifacts"]))
            p["source_artifacts"][role]["sha256"] = "0" * 64

        self._rejects(mutate)

    def test_claiming_scoring_authority_is_rejected(self) -> None:
        def mutate(p: dict) -> None:
            p["authorizations"]["value_screen_scoring"] = True

        self._rejects(mutate)

    def test_claiming_rescreen_authority_is_rejected(self) -> None:
        def mutate(p: dict) -> None:
            p["next_artifact"]["may_authorize_rescreen"] = True

        self._rejects(mutate)

    def test_widening_the_target_cells_is_rejected(self) -> None:
        def mutate(p: dict) -> None:
            p["target_cells"].append(
                {"action_id": "off", "regime_id": "R1", "content_seed": 0}
            )

        self._rejects(mutate)

    def test_silencing_the_lean_server_claim_is_rejected(self) -> None:
        def mutate(p: dict) -> None:
            p["hypothesis"]["lean_server_left_running"] = False

        self._rejects(mutate)

    def test_remeasuring_the_source_run_is_rejected(self) -> None:
        def mutate(p: dict) -> None:
            p["comparison"]["source_run_remeasured"] = True

        self._rejects(mutate)

    def test_enabling_retry_is_rejected(self) -> None:
        def mutate(p: dict) -> None:
            p["execution_policy"]["retry_allowed"] = True

        self._rejects(mutate)

    def test_baseline_tampering_is_rejected(self) -> None:
        def mutate(p: dict) -> None:
            p["comparison"]["v13_block1_rejected_rounds"] = 0

        self._rejects(mutate)


if __name__ == "__main__":
    unittest.main()
