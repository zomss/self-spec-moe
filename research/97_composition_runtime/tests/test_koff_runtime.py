"""CPU-only tests for the Phase 97 minimal-B0 K/OFF runtime."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR / "scripts"))

from replay_koff_trace import (  # noqa: E402
    KOffReplayError,
    interval_tie_set,
    replay_trace,
)

PREFLIGHT_DIR = PHASE_DIR / "data" / "preflight"
P3_DIR = PHASE_DIR / "data" / "p3"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


class KOffRuntimeTests(unittest.TestCase):
    """Replay positive and fail-closed K/OFF transitions."""

    def setUp(self) -> None:
        self.environment = _load(PREFLIGHT_DIR / "environment_qwen3_8b_h100_tp1.json")
        self.workload = _load(PREFLIGHT_DIR / "workload_rl_capacity_v1.json")
        self.candidate = _load(PREFLIGHT_DIR / "candidate_b0_measured.json")
        self.boot = _load(P3_DIR / "boot_b0_minimal_k4.json")
        self.registry = _load(P3_DIR / "actions_b0_minimal_k4.json")
        self.runtime = _load(P3_DIR / "runtime_snapshot_b0_synthetic.json")
        self.trace = _load(P3_DIR / "passive_koff_trace_synthetic.json")

    def replay(self) -> dict:
        return replay_trace(
            self.environment,
            self.workload,
            self.candidate,
            self.boot,
            self.registry,
            self.runtime,
            self.trace,
        )

    def test_checked_in_trace_replays(self) -> None:
        result = self.replay()
        expected = _load(P3_DIR / "replay_koff_trace_result.json")
        self.assertEqual(result, expected)
        self.assertEqual(result["status"], "pass")
        self.assertFalse(result["scored"])
        self.assertEqual(result["total_target_steps"], 256)
        self.assertEqual(result["total_engine_steps"], 16)
        self.assertEqual(result["probe_target_steps"], 16)
        self.assertEqual(result["probe_target_step_fraction"], 0.0625)
        self.assertEqual(result["final_action_id"], "target-matching-k4")
        aggregates = {row["action_id"]: row for row in result["action_aggregates"]}
        self.assertEqual(aggregates["off"]["tau_eff"], 1.0)
        self.assertEqual(aggregates["target-matching-k4"]["tau_eff"], 3.0)

    def test_exact_resource_and_graph_binding_is_required(self) -> None:
        with self.subTest("resource closure"):
            self.boot["resources"]["target_fixed_bytes"] += 1
            with self.assertRaisesRegex(KOffReplayError, "HBM terms"):
                self.replay()

        self.setUp()
        with self.subTest("graph pool"):
            graph_ids = self.candidate["resident_objects"]["graph_ids"]
            graph_ids.remove("target-k1")
            with self.assertRaisesRegex(KOffReplayError, "graph pool"):
                self.replay()

        self.setUp()
        with self.subTest("max K"):
            self.boot["max_k"] = 5
            with self.assertRaisesRegex(KOffReplayError, "max_k=4"):
                self.replay()

    def test_registry_graph_roles_are_exact(self) -> None:
        action = self.registry["actions"][1]
        action["target_graph_descriptor"]["graph_id"] = "target-k1"
        with self.assertRaisesRegex(KOffReplayError, "target-k5"):
            self.replay()

    def test_shared_kv_state_cannot_change(self) -> None:
        cases = (
            ("binding_id", "other-binding", "KV binding changed"),
            ("pool_id", "other-pool", "KV pool changed"),
            ("true_slot_mapping_id", "other-slots", "slot mapping changed"),
        )
        for field, value, error in cases:
            with self.subTest(field=field):
                trace = copy.deepcopy(self.trace)
                trace["segments"][0]["runtime_state"][field] = value
                self.trace = trace
                with self.assertRaisesRegex(KOffReplayError, error):
                    self.replay()
                self.trace = _load(P3_DIR / "passive_koff_trace_synthetic.json")

    def test_target_matching_weight_version_cannot_diverge(self) -> None:
        state = self.trace["segments"][0]["runtime_state"]
        state["draft_weight_versions"]["target-matching"] = "target-v0"
        with self.assertRaisesRegex(KOffReplayError, "version diverged"):
            self.replay()

    def test_target_step_counter_closure_is_exact(self) -> None:
        self.trace["segments"][0]["counters"]["E_committed"] += 1
        with self.assertRaisesRegex(KOffReplayError, r"E\+C=A\+H"):
            self.replay()

    def test_action_specific_accounting_is_fail_closed(self) -> None:
        with self.subTest("accepted bound"):
            counters = self.trace["segments"][0]["counters"]
            counters["A_accepted"] = 129
            counters["E_committed"] = 161
            with self.assertRaisesRegex(KOffReplayError, "more than K"):
                self.replay()

        self.setUp()
        with self.subTest("OFF drafting"):
            self.trace["segments"][1]["counters"]["D_armed"] = 1
            with self.assertRaisesRegex(KOffReplayError, "drafting to OFF"):
                self.replay()

    def test_engine_trace_must_close_to_h(self) -> None:
        engine_step = self.trace["segments"][0]["engine_steps"][0]
        engine_step["active_request_count"] = 15
        with self.assertRaisesRegex(KOffReplayError, "exact engine-step trace"):
            self.replay()

    def test_engine_graph_must_match_selected_action(self) -> None:
        engine_step = self.trace["segments"][0]["engine_steps"][0]
        engine_step["target_graph_id"] = "target-k1"
        with self.assertRaisesRegex(KOffReplayError, "target graph"):
            self.replay()

    def test_switch_inside_target_step_is_rejected(self) -> None:
        self.trace["segments"][0]["target_step_boundary"] = False
        with self.assertRaisesRegex(KOffReplayError, "inside a target step"):
            self.replay()

    def test_off_fallback_must_remain_ready(self) -> None:
        graphs = self.trace["segments"][0]["runtime_state"]["available_graph_ids"]
        graphs.remove("target-k1")
        with self.assertRaisesRegex(KOffReplayError, "loses OFF fallback"):
            self.replay()

    def test_graph_unavailability_falls_back_to_off(self) -> None:
        result = self.replay()
        decision = result["decisions"][1]
        self.assertEqual(decision["selected_action_id"], "off")
        self.assertEqual(decision["fallback_reason"], "requested_ineligible")

    def test_stale_exploitation_falls_back_to_off(self) -> None:
        result = self.replay()
        decision = result["decisions"][3]
        self.assertEqual(decision["selected_action_id"], "off")
        self.assertEqual(decision["fallback_reason"], "stale_evidence")

    def test_stale_probe_refreshes_k4_evidence(self) -> None:
        result = self.replay()
        probe = result["decisions"][4]
        exploit = result["decisions"][5]
        self.assertEqual(probe["tie_set_action_ids"], ["off"])
        self.assertEqual(probe["selected_action_id"], "target-matching-k4")
        self.assertEqual(exploit["selected_action_id"], "target-matching-k4")
        self.assertIsNone(exploit["fallback_reason"])

    def test_fabricated_freshness_is_rejected(self) -> None:
        estimate = self.trace["segments"][5]["interval_estimates"][0]
        estimate["observed_at_target_step"] = 208
        with self.assertRaisesRegex(KOffReplayError, "unproven freshness"):
            self.replay()

    def test_interval_inversion_is_rejected(self) -> None:
        estimate = self.trace["segments"][0]["interval_estimates"][0]
        estimate["q_lo"] = estimate["q_hi"] + 0.1
        with self.assertRaisesRegex(KOffReplayError, "inverted q interval"):
            self.replay()

    def test_probe_duty_budget_is_enforced(self) -> None:
        self.trace["policy"]["max_probe_target_step_fraction"] = 0.05
        with self.assertRaisesRegex(KOffReplayError, "probe-duty budget"):
            self.replay()

    def test_interval_dominance_keeps_boundary_ties(self) -> None:
        intervals = {
            "winner": (2.0, 2.1),
            "boundary": (0.5, 1.8),
        }
        self.assertEqual(
            interval_tie_set(intervals, epsilon=0.1),
            ["boundary", "winner"],
        )
        self.assertEqual(
            interval_tie_set({"candidate": (1.01, 1.02)}, epsilon=0.015),
            ["candidate", "off"],
        )


if __name__ == "__main__":
    unittest.main()
