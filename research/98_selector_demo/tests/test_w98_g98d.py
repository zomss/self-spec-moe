# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for G98-D, centred on the screen commitment barrier.

D2(a)'s claim is only meaningful if the product-bound screen was fixed
BEFORE the admitted compositions were measured. These attack that barrier
from the directions a real run could breach it.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR / "scripts"))

import run_w98_g98d_campaign as gate  # noqa: E402


def _screen():
    return {
        "base_tau": {"R1": 8.0},
        "lever_factors": {"target-matching/w256/skip0": {"R1": 0.9}},
        "bridge_width_accepted_tokens": 0.375,
        "tolerance": gate.SCREEN_TOLERANCE,
        "admitted": [
            {
                "cell": "w4a16-quantized/w256/skip0",
                "config": {
                    "quant": "w4a16-quantized",
                    "window": 256,
                    "skip_count": 0,
                },
                "levers": ["target-matching/w256/skip0"],
                "bound_tau": {"R1": 7.2},
                "admitted": True,
            }
        ],
    }


def _confirmation(tmp: Path, tau_eff: float) -> None:
    confirm = tmp / gate.CONFIRM_DIR
    confirm.mkdir(parents=True, exist_ok=True)
    (confirm / "w4a16-quantized_w256_skip0_s4.json").write_text(
        json.dumps(
            {
                "config": {
                    "quant": "w4a16-quantized",
                    "window": 256,
                    "skip_count": 0,
                },
                "observations": {
                    "R1": {
                        "closure": {
                            "H": 10,
                            "D": 10,
                            "A": 60,
                            "C": 0,
                            "E": 70,
                            "tau_eff": tau_eff,
                        },
                        "tau_profile": {"buckets": {}},
                    }
                },
            }
        )
    )


class BarrierTests(unittest.TestCase):
    def test_screen_is_immutable_once_committed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            gate.commit_screen(tmp, _screen())
            with self.assertRaises(gate.G98DError):
                gate.commit_screen(tmp, _screen())

    def test_screen_cannot_be_committed_after_confirmations(self) -> None:
        """The whole point: no fitting the screen to measured compositions."""
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            _confirmation(tmp, 7.5)
            with self.assertRaises(gate.G98DError):
                gate.commit_screen(tmp, _screen())

    def test_scoring_refuses_without_a_committed_screen(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            _confirmation(tmp, 7.5)
            with self.assertRaises(gate.G98DError):
                gate.score_reveal(tmp)

    def test_a_tampered_screen_fails_its_digest(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            gate.commit_screen(tmp, _screen())
            path = tmp / gate.SCREEN_NAME
            record = json.loads(path.read_text())
            record["screen"]["admitted"][0]["bound_tau"]["R1"] = 1.0
            path.write_text(json.dumps(record))
            with self.assertRaises(gate.G98DError):
                gate.require_committed_screen(tmp)


class ScoringTests(unittest.TestCase):
    def test_a_confirmation_above_the_bound_is_honest(self) -> None:
        """Admit-only: exceeding the bound is exactly what should happen."""
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            gate.commit_screen(tmp, _screen())
            _confirmation(tmp, 7.9)
            result = gate.score_reveal(tmp)
            self.assertTrue(result["d2a_honest"])
            self.assertEqual(result["d2a_violations"], 0)

    def test_a_confirmation_far_below_the_bound_violates(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            gate.commit_screen(tmp, _screen())
            _confirmation(tmp, 5.0)
            result = gate.score_reveal(tmp)
            self.assertFalse(result["d2a_honest"])
            self.assertEqual(result["d2a_violations"], 1)

    def test_a_shortfall_inside_tolerance_is_honest(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            gate.commit_screen(tmp, _screen())
            _confirmation(tmp, 7.2 - 7.2 * gate.SCREEN_TOLERANCE / 2)
            result = gate.score_reveal(tmp)
            self.assertTrue(result["d2a_honest"])


class ScopeTests(unittest.TestCase):
    def test_compositions_vary_at_least_two_axes(self) -> None:
        for cfg in gate.composed_profiles():
            varied = (
                (cfg["quant"] != "target-matching")
                + (cfg["window"] != "off")
                + (cfg["skip_count"] != 0)
            )
            self.assertGreaterEqual(varied, 2, cfg)

    def test_singles_and_compositions_are_disjoint(self) -> None:
        singles = {gate._config_key(c) for c in gate.single_lever_profiles()}
        composed = {gate._config_key(c) for c in gate.composed_profiles()}
        self.assertEqual(singles & composed, set())

    def test_the_timing_gate_is_deliberately_absent(self) -> None:
        """Acceptance is clamp-immune; the shape gate would only reject data."""
        package = gate.expected_authorization()
        self.assertFalse(package["gates"]["measurement_shape_gate"])
        self.assertFalse(package["gates"]["repeat_boot_anchors"])

    def test_the_bridge_width_reaches_the_screen(self) -> None:
        self.assertGreater(gate.bridge_width(), 0.0)
        self.assertEqual(
            gate.expected_authorization()["screen"]["bridge_width_accepted_tokens"],
            gate.bridge_width(),
        )


if __name__ == "__main__":
    unittest.main()
