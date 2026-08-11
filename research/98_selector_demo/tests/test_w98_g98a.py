# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the G98-A non-scored smoke authorization."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
sys.path.insert(0, str(PHASE_DIR / "scripts"))

import run_w98_g98a_smoke as gate  # noqa: E402


def _package() -> dict:
    path = REPO_ROOT / gate.AUTHORIZATION_PATH
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


class GateScopeTests(unittest.TestCase):
    """The gate must cover the axes that can invalidate the lattice."""

    def test_package_validates(self) -> None:
        gate.validate_authorization(_package())

    def test_five_boot_classes(self) -> None:
        self.assertEqual(len(gate.BOOT_CLASSES), 5)
        self.assertEqual(_package()["execution_policy"]["physical_boot_count"], 5)

    def test_both_quant_paths_are_booted(self) -> None:
        quants = {b["quant"] for b in gate.BOOT_CLASSES}
        self.assertEqual(quants, {"target-matching", "w4a16-quantized"})

    def test_quantized_boots_cannot_alias_weights(self) -> None:
        """_weight_sharing_enabled raises when the draft differs."""
        for boot in gate.BOOT_CLASSES:
            if boot["quant"] == "w4a16-quantized":
                self.assertFalse(boot["share_weights"])

    def test_every_boot_shares_target_kv(self) -> None:
        for boot in gate.BOOT_CLASSES:
            env = gate.boot_environment(boot)
            self.assertEqual(env["VLLM_SELF_SPEC_SHARED_KV"], "1")

    def test_quantized_env_disables_weight_sharing(self) -> None:
        quant = [b for b in gate.BOOT_CLASSES if not b["share_weights"]][0]
        env = gate.boot_environment(quant)
        self.assertEqual(env["VLLM_SELF_SPEC_SHARE_WEIGHTS"], "0")
        self.assertEqual(env["VLLM_SELF_SPEC_SHARED_KV"], "1")

    def test_window_and_skip_reach_the_environment(self) -> None:
        boot = [b for b in gate.BOOT_CLASSES if b["window"] and b["skip_count"]][0]
        env = gate.boot_environment(boot)
        self.assertEqual(env["VLLM_SELF_SPEC_DRAFT_KV_WINDOW"], str(boot["window"]))
        self.assertTrue(env["VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS"])
        self.assertEqual(env["VLLM_SELF_SPEC_DRAFT_KV_SINKS"], "16")

    def test_skip_zero_leaves_the_layer_set_empty(self) -> None:
        boot = [b for b in gate.BOOT_CLASSES if b["skip_count"] == 0][0]
        self.assertEqual(
            gate.boot_environment(boot)["VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS"], ""
        )

    def test_boots_run_on_the_reserved_lane(self) -> None:
        env = gate.boot_environment(gate.BOOT_CLASSES[0])
        self.assertEqual(
            env["CUDA_VISIBLE_DEVICES"], str(gate.LANE["physical_gpu_index"])
        )

    def test_quantized_checkpoint_is_present(self) -> None:
        self.assertTrue(Path(gate.QUANT_DRAFT_CKPT).is_dir())
        self.assertTrue(_package()["models"]["quantized_draft_present"])


class GateBoundaryTests(unittest.TestCase):
    """A smoke gate must claim nothing about acceptance or value."""

    def test_emits_no_captures(self) -> None:
        self.assertEqual(_package()["execution_policy"]["captures_emitted"], 0)

    def test_grants_no_round_authority(self) -> None:
        auth = _package()["authorizations"]
        self.assertTrue(auth["g98a_smoke_execution"])
        for denied in (
            "round1_execution",
            "round2_execution",
            "scoring",
            "acceptance_claim",
            "production_value_claim",
        ):
            self.assertFalse(auth[denied], denied)
        self.assertFalse(_package()["next_artifact"]["may_authorize_round1"])

    def test_records_the_lattice_consequence_of_failure(self) -> None:
        a = _package()["assumption_under_test"]
        self.assertEqual(a["prereg_status"], "unverified")
        self.assertIn("halves from 30 to 15", a["on_failure"])


class GateFailClosedTests(unittest.TestCase):
    """Drift and scope inflation must be refused."""

    def _rejects(self, mutate) -> None:
        package = copy.deepcopy(_package())
        mutate(package)
        with self.assertRaises(gate.G98ASmokeError):
            gate.validate_authorization(package)

    def test_source_hash_drift_is_rejected(self) -> None:
        def mutate(p: dict) -> None:
            role = next(iter(p["source_artifacts"]))
            p["source_artifacts"][role]["sha256"] = "0" * 64

        self._rejects(mutate)

    def test_claiming_round1_authority_is_rejected(self) -> None:
        def mutate(p: dict) -> None:
            p["authorizations"]["round1_execution"] = True

        self._rejects(mutate)

    def test_claiming_the_assumption_is_verified_is_rejected(self) -> None:
        def mutate(p: dict) -> None:
            p["assumption_under_test"]["prereg_status"] = "verified"

        self._rejects(mutate)

    def test_dropping_the_quantized_boots_is_rejected(self) -> None:
        def mutate(p: dict) -> None:
            p["boot_classes"] = [
                b for b in p["boot_classes"] if b["quant"] == "target-matching"
            ]

        self._rejects(mutate)

    def test_enabling_weight_sharing_for_quant_is_rejected(self) -> None:
        def mutate(p: dict) -> None:
            for b in p["boot_classes"]:
                if b["quant"] == "w4a16-quantized":
                    b["share_weights"] = True

        self._rejects(mutate)

    def test_enabling_retry_is_rejected(self) -> None:
        def mutate(p: dict) -> None:
            p["execution_policy"]["retry_allowed"] = True

        self._rejects(mutate)


if __name__ == "__main__":
    unittest.main()
