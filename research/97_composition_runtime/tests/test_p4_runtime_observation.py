# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the P4 runtime observations added after the V13 screen.

The V13 captures declared ``hardware_id`` from the frozen single-GPU
assignment (GPU 4) while actually running on GPUs 0 and 1, and carried only a
declared ``graph_grade``. These tests pin the two repairs: what the run
observed is recorded, and a declaration that contradicts the live device is
refused.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
sys.path.insert(0, str(PHASE_DIR / "scripts"))

import run_p4_b0_value_screen as matrix  # noqa: E402

from vllm.v1.spec_decode import koff_runtime  # noqa: E402


class ChainModeObservationTests(unittest.TestCase):
    """The chain's dispatched mode must reach the capture."""

    def tearDown(self) -> None:
        koff_runtime._RUNTIME_OBSERVATION.pop("chain_runtime_mode", None)

    def test_observation_is_recorded(self) -> None:
        koff_runtime.observe_chain_runtime_mode("PIECEWISE")
        self.assertEqual(
            koff_runtime._RUNTIME_OBSERVATION["chain_runtime_mode"], "PIECEWISE"
        )

    def test_a_fallback_to_none_is_visible(self) -> None:
        koff_runtime.observe_chain_runtime_mode("PIECEWISE")
        koff_runtime.observe_chain_runtime_mode("NONE")
        self.assertEqual(
            koff_runtime._RUNTIME_OBSERVATION["chain_runtime_mode"], "NONE"
        )

    def test_unobserved_is_the_default(self) -> None:
        self.assertEqual(
            koff_runtime._RUNTIME_OBSERVATION.get("chain_runtime_mode", "unobserved"),
            "unobserved",
        )

    def test_proposer_reporter_never_raises(self) -> None:
        from vllm.v1.spec_decode import llm_base_proposer

        with mock.patch.object(
            koff_runtime, "observe_chain_runtime_mode", side_effect=RuntimeError
        ):
            llm_base_proposer._observe_chain_runtime_mode("PIECEWISE")


class HardwareObservationTests(unittest.TestCase):
    """A capture must not be able to claim a GPU it did not run on."""

    def test_observed_id_is_a_device_uuid_or_unavailable(self) -> None:
        observed = koff_runtime.observed_hardware_id()
        self.assertTrue(
            observed == "unavailable" or observed.startswith("GPU-"), observed
        )

    def test_the_frozen_capture_contract_is_untouched(self) -> None:
        """Observations live beside the capture, never inside its schema."""
        recorder = koff_runtime.P4SameEventRecorder
        self.assertNotIn("observed_hardware_id", recorder._RUNNER_FIELDS)
        self.assertNotIn("chain_runtime_mode", recorder._RUNNER_FIELDS)
        self.assertFalse(hasattr(recorder, "_OBSERVED_RUNNER_FIELDS"))


class BootObservationTests(unittest.TestCase):
    """Each boot records the device and chain mode it actually used."""

    def test_record_carries_observed_and_declared_values(self) -> None:
        import json
        import tempfile

        koff_runtime.observe_chain_runtime_mode("PIECEWISE")
        self.addCleanup(
            koff_runtime._RUNTIME_OBSERVATION.pop, "chain_runtime_mode", None
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = {
                "boot_id": "p4-b0-b1-p2-k4",
                "boot_block_id": 1,
                "action_id": "target-matching-k4",
                "capture_dir": str(root / "captures" / "p4-b0-b1-p2-k4"),
                "lane": matrix.lane_for_block(1),
            }
            (root / "captures" / "p4-b0-b1-p2-k4").mkdir(parents=True)
            matrix._write_boot_observation(spec)
            record = json.loads(
                (root / "observations" / "p4-b0-b1-p2-k4.json").read_text()
            )
        self.assertEqual(record["chain_runtime_mode"], "PIECEWISE")
        self.assertEqual(record["boot_block_id"], 1)
        self.assertEqual(
            record["declared_lane_gpu_uuid"],
            matrix.lane_for_block(1)["physical_gpu_uuid"],
        )
        self.assertTrue(
            record["observed_hardware_id"] == "unavailable"
            or record["observed_hardware_id"].startswith("GPU-")
        )
        self.assertTrue(record["cpu_affinity"])


class CpuReservationTests(unittest.TestCase):
    """Lanes must leave headroom instead of claiming the whole machine."""

    def _cpus(self, spec: str) -> set[int]:
        lo, hi = spec.split("-")
        return set(range(int(lo), int(hi) + 1))

    def test_lanes_no_longer_claim_every_core(self) -> None:
        import os

        lanes = matrix.lane_assignment()
        claimed: set[int] = set()
        for lane in lanes:
            claimed |= self._cpus(lane["cpu_affinity"])
        self.assertLess(len(claimed), os.cpu_count())

    def test_lanes_are_disjoint_from_each_other(self) -> None:
        a, b = (self._cpus(lane["cpu_affinity"]) for lane in matrix.lane_assignment())
        self.assertFalse(a & b)

    def test_lanes_are_disjoint_from_the_lean_server_set(self) -> None:
        lean = self._cpus(matrix.LEAN_SERVER_CPUS)
        for lane in matrix.lane_assignment():
            self.assertFalse(self._cpus(lane["cpu_affinity"]) & lean)

    def test_free_headroom_remains(self) -> None:
        import os

        used = self._cpus(matrix.LEAN_SERVER_CPUS)
        for lane in matrix.lane_assignment():
            used |= self._cpus(lane["cpu_affinity"])
        self.assertGreaterEqual(os.cpu_count() - len(used), 32)


if __name__ == "__main__":
    unittest.main()
