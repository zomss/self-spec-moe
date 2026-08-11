# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for G98-B Round 1, centred on the commitment barrier.

D1's soundness claim rests entirely on the held-out predictions having been
fixed before those configurations were measured. These tests attack that
barrier from every direction a real run could breach it.
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
sys.path.insert(0, str(PHASE_DIR / "scripts"))

import run_w98_g98b_round1 as gate  # noqa: E402


def _package() -> dict:
    path = REPO_ROOT / gate.AUTHORIZATION_PATH
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _heldout_keys() -> set[str]:
    path = REPO_ROOT / gate.PREREG_HELDOUT
    with path.open(encoding="utf-8") as handle:
        return {gate._triple_key(t) for t in json.load(handle)["heldout"]}


def _predictions() -> dict:
    return {key: {"q_lo": 0.9, "q_hi": 1.1} for key in _heldout_keys()}


class GateScopeTests(unittest.TestCase):
    """Round 1 fits from singles and reveals a frozen held-out set."""

    def test_package_validates(self) -> None:
        gate.validate_authorization(_package())

    def test_fit_uses_single_lever_profiles_only(self) -> None:
        singles = gate.single_lever_profiles()
        for cfg in singles:
            varied = sum(
                (
                    cfg["quant"] != "target-matching",
                    cfg["window"] != "off",
                    cfg["skip_count"] != 0,
                )
            )
            self.assertLessEqual(varied, 1, cfg)
        self.assertTrue(_package()["d1"]["fit_uses_single_lever_profiles_only"])

    def test_stages_place_the_commit_between_the_gpu_stages(self) -> None:
        stages = _package()["stages"]
        self.assertEqual([s["id"] for s in stages], ["B1", "B2", "B3", "B4"])
        self.assertTrue(stages[0]["gpu"])
        self.assertFalse(stages[1]["gpu"])
        self.assertTrue(stages[2]["gpu"])

    def test_requires_a_passing_g98a(self) -> None:
        self.assertTrue(_package()["prerequisite"]["passed"])
        self.assertTrue(
            _package()["prerequisite"]["quantized_draft_with_shared_kv_verified"]
        )

    def test_grants_no_round2_authority(self) -> None:
        auth = _package()["authorizations"]
        self.assertTrue(auth["round1_execution"])
        for denied in (
            "round2_execution",
            "round2_scoring",
            "d3_claim",
            "production_value_claim",
        ):
            self.assertFalse(auth[denied], denied)
        self.assertFalse(_package()["next_artifact"]["may_authorize_round2"])

    def test_records_the_w14d_fallback(self) -> None:
        fb = _package()["w14d_fallback"]
        self.assertFalse(fb["phase_96_w14d_scored_surface_present"])
        self.assertIn("NOT EXERCISED", fb["consequence"])

    def test_shapes_are_sized_against_the_quantized_headroom(self) -> None:
        note = _package()["resource_note"]
        self.assertEqual(note["shapes_sized_against"], "quantized")
        self.assertLess(note["headroom_fraction"], 0.05)

    def test_round1_avoids_the_capture_path(self) -> None:
        """The capture recorder still refuses quantized boots."""
        trace = _package()["trace_path"]
        self.assertFalse(trace["uses_p4_capture_recorder"])
        self.assertTrue(trace["scope_aware"])


class CommitmentBarrierTests(unittest.TestCase):
    """The barrier is the gate's reason for existing."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "g98_b"
        self.out.mkdir()

    def test_commit_then_reveal_is_permitted(self) -> None:
        gate.commit_predictions(self.out, _predictions())
        record = gate.require_committed_predictions(self.out)
        self.assertTrue(record["committed_before_reveal"])
        self.assertEqual(record["heldout_count"], len(_heldout_keys()))

    def test_reveal_without_commitment_is_refused(self) -> None:
        with self.assertRaises(gate.G98BError) as ctx:
            gate.require_committed_predictions(self.out)
        self.assertIn("unfalsifiable", str(ctx.exception))

    def test_committing_after_a_heldout_measurement_is_refused(self) -> None:
        """The exact way a run could launder a prediction into a postdiction."""
        heldout = self.out / gate.HELDOUT_DIR
        heldout.mkdir()
        (heldout / "target-matching_w128_skip4.json").write_text("{}")
        with self.assertRaises(gate.G98BError) as ctx:
            gate.commit_predictions(self.out, _predictions())
        self.assertIn("after held-out measurements exist", str(ctx.exception))

    def test_predictions_are_immutable_once_committed(self) -> None:
        gate.commit_predictions(self.out, _predictions())
        with self.assertRaises(gate.G98BError) as ctx:
            gate.commit_predictions(self.out, _predictions())
        self.assertIn("immutable", str(ctx.exception))

    def test_tampered_predictions_are_detected(self) -> None:
        gate.commit_predictions(self.out, _predictions())
        path = self.out / gate.PREDICTIONS_NAME
        record = json.loads(path.read_text())
        key = next(iter(record["predictions"]))
        record["predictions"][key]["q_lo"] = 0.0
        path.write_text(json.dumps(record))
        with self.assertRaises(gate.G98BError) as ctx:
            gate.require_committed_predictions(self.out)
        self.assertIn("modified after commitment", str(ctx.exception))

    def test_predictions_must_cover_exactly_the_registered_set(self) -> None:
        partial = _predictions()
        partial.pop(next(iter(partial)))
        with self.assertRaises(gate.G98BError) as ctx:
            gate.commit_predictions(self.out, partial)
        self.assertIn("exactly the registered held-out set", str(ctx.exception))

    def test_extra_predictions_are_refused(self) -> None:
        extra = _predictions()
        extra["invented/w999/skip99"] = {"q_lo": 1.0, "q_hi": 1.0}
        with self.assertRaises(gate.G98BError):
            gate.commit_predictions(self.out, extra)

    def test_a_forged_commitment_flag_is_detected(self) -> None:
        gate.commit_predictions(self.out, _predictions())
        path = self.out / gate.PREDICTIONS_NAME
        record = json.loads(path.read_text())
        record["committed_before_reveal"] = False
        path.write_text(json.dumps(record))
        with self.assertRaises(gate.G98BError) as ctx:
            gate.require_committed_predictions(self.out)
        self.assertIn("pre-reveal commitment", str(ctx.exception))


class FailClosedTests(unittest.TestCase):
    """Registered contracts must fail closed."""

    def _rejects(self, mutate) -> None:
        package = copy.deepcopy(_package())
        mutate(package)
        with self.assertRaises(gate.G98BError):
            gate.validate_authorization(package)

    def test_source_hash_drift_is_rejected(self) -> None:
        def mutate(p: dict) -> None:
            role = next(iter(p["source_artifacts"]))
            p["source_artifacts"][role]["sha256"] = "0" * 64

        self._rejects(mutate)

    def test_claiming_round2_authority_is_rejected(self) -> None:
        def mutate(p: dict) -> None:
            p["authorizations"]["round2_execution"] = True

        self._rejects(mutate)

    def test_relaxing_epsilon_arm_is_rejected(self) -> None:
        def mutate(p: dict) -> None:
            p["d1"]["epsilon_arm"] = 0.05

        self._rejects(mutate)

    def test_raising_the_false_elimination_budget_is_rejected(self) -> None:
        def mutate(p: dict) -> None:
            p["d1"]["false_elimination_budget"] = 1

        self._rejects(mutate)

    def test_disabling_the_barrier_is_rejected(self) -> None:
        def mutate(p: dict) -> None:
            p["commitment_barrier"]["predictions_committed_before_reveal"] = False

        self._rejects(mutate)

    def test_hiding_the_w14d_fallback_is_rejected(self) -> None:
        def mutate(p: dict) -> None:
            p["w14d_fallback"]["phase_96_w14d_scored_surface_present"] = True

        self._rejects(mutate)

    def test_enabling_retry_is_rejected(self) -> None:
        def mutate(p: dict) -> None:
            p["execution_policy"]["retry_allowed"] = True

        self._rejects(mutate)


if __name__ == "__main__":
    unittest.main()


class MeasurementWiringTests(unittest.TestCase):
    """The loops must express the model's inputs and the barrier's ordering."""

    def test_geometry_follows_the_factored_model(self) -> None:
        target = gate.lever_geometry(
            {"quant": "target-matching", "window": "off", "skip_count": 0}, 8000
        )
        quant = gate.lever_geometry(
            {"quant": "w4a16-quantized", "window": "off", "skip_count": 0}, 8000
        )
        self.assertLess(quant["weight_bytes"], target["weight_bytes"])
        self.assertEqual(quant["kv_bytes"], target["kv_bytes"])
        self.assertEqual(target["keep_frac"], 1.0)

    def test_a_window_bounds_the_kv_read(self) -> None:
        off = gate.lever_geometry(
            {"quant": "target-matching", "window": "off", "skip_count": 0}, 8000
        )
        win = gate.lever_geometry(
            {"quant": "target-matching", "window": 512, "skip_count": 0}, 8000
        )
        self.assertLess(win["kv_bytes"], off["kv_bytes"])

    def test_skip_reduces_keep_frac(self) -> None:
        for count in (0, 4, 8):
            geom = gate.lever_geometry(
                {"quant": "target-matching", "window": "off", "skip_count": count}, 8000
            )
            self.assertAlmostEqual(geom["keep_frac"], 1.0 - count / gate.DRAFT_LAYERS)

    def test_boot_environment_carries_the_scope_and_trace(self) -> None:
        env = gate.boot_environment(
            {"quant": "w4a16-quantized", "window": 512, "skip_count": 4},
            Path("/tmp/w98-trace.jsonl"),
        )
        self.assertEqual(env["VLLM_SELF_SPEC_BOOT_SCOPE"], "w98-lattice")
        self.assertEqual(env["VLLM_SELF_SPEC_SHARE_WEIGHTS"], "0")
        self.assertEqual(env["VLLM_SELF_SPEC_SHARED_KV"], "1")
        self.assertEqual(env["VLLM_SELF_SPEC_DRAFT_KV_WINDOW"], "512")
        self.assertEqual(env["VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS"], "2,4,7,16")
        self.assertTrue(env["VLLM_SELF_SPEC_KOFF_TRACE"])
        self.assertEqual(env["VLLM_SELF_SPEC_DRAFT_PARTIAL_REPLICA"], "")

    def test_target_matching_boots_keep_weight_sharing(self) -> None:
        env = gate.boot_environment(
            {"quant": "target-matching", "window": "off", "skip_count": 0},
            Path("/tmp/w98-trace.jsonl"),
        )
        self.assertEqual(env["VLLM_SELF_SPEC_SHARE_WEIGHTS"], "1")

    def test_fit_refuses_an_incomplete_single_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / gate.SINGLES_DIR).mkdir()
            with self.assertRaises(gate.G98BError) as ctx:
                gate.fit_and_predict(out)
            self.assertIn("single profiles", str(ctx.exception))

    def test_scoring_requires_the_committed_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / gate.HELDOUT_DIR).mkdir()
            with self.assertRaises(gate.G98BError):
                gate.score_reveal(out)

    def test_result_reports_d1_as_not_exercised_under_the_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / gate.HELDOUT_DIR).mkdir()
            gate.commit_predictions(out, _predictions())
            result = gate.score_reveal(out)
            self.assertFalse(result["d1_exercised"])
            self.assertFalse(result["may_authorize_round2"])
            self.assertTrue(result["predictions_committed_before_reveal"])


class SamplingDepthTests(unittest.TestCase):
    """The fit must be taken at each regime's registered state."""

    def test_sampling_is_state_indexed(self) -> None:
        sampling = _package()["sampling"]
        self.assertTrue(sampling["state_indexed"])
        self.assertEqual(
            sampling["prompts_per_regime"], "the regime's registered batch"
        )

    def test_token_depth_is_registered(self) -> None:
        self.assertEqual(_package()["sampling"]["measure_tokens"], gate.MEASURE_TOKENS)
        self.assertGreaterEqual(gate.MEASURE_TOKENS, 256)

    def test_a_thin_sample_yields_no_mean(self) -> None:
        """An unusable sample must not silently become a fitted point."""
        self.assertGreaterEqual(gate.MIN_ARMED_STEPS, 64)
