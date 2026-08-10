"""CPU-only tests for the Phase 97 B0 adapter and frozen scorer."""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR / "scripts"))

from adapt_p4_b0_same_event import B0AdapterError, adapt_capture  # noqa: E402
from score_p4_b0 import (  # noqa: E402
    B0ScorerError,
    score_rounds,
    validate_runner_scorer_contract,
)

CONTRACT_PATH = PHASE_DIR / "data" / "p4" / "p4_b0_runner_scorer_contract.json"
VALIDATION_PATH = PHASE_DIR / "data" / "p4" / "p4_b0_runner_scorer_validation.json"
MANIFEST_PATH = PHASE_DIR / "data" / "p4" / "p4_b0_prompt_manifest.json"
P3_TRACE_PATH = PHASE_DIR / "data" / "p3" / "live_smoke_20260809_trace.jsonl"

ACTION_ORDERS = {
    1: ["off", "target-matching-k4", "target-matching-w512-masked-k4"],
    2: ["target-matching-k4", "target-matching-w512-masked-k4", "off"],
    3: ["target-matching-w512-masked-k4", "off", "target-matching-k4"],
}
ACTION_REALIZATIONS = {
    "off": "live-b0-forced-off",
    "target-matching-k4": "live-b0-target-matching-k4",
    "target-matching-w512-masked-k4": ("boot-static-mask-equivalent-surrogate"),
}
REGIMES = {
    "R4": (8, 512, 0.0),
    "R5": (8, 512, 0.0),
    "R5cot": (8, 3072, 0.0),
    "R8": (16, 2048, 1.0),
    "R1": (1, 1024, 0.0),
    "R6": (32, 256, 0.0),
}


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def valid_contract() -> dict:
    """Return an independent copy of the frozen runner/scorer contract."""
    return json.loads(CONTRACT_PATH.read_text())


def prompt_ids(regime_id: str, content_seed: int) -> list[str]:
    """Return one canonical prompt group from the checked manifest."""
    manifest = json.loads(MANIFEST_PATH.read_text())
    rows = [
        row
        for row in manifest["prompts"]
        if row["regime_id"] == regime_id and row["content_seed"] == content_seed
    ]
    rows.sort(key=lambda row: row["prompt_index"])
    return [row["record_id"] for row in rows]


def _refresh_event(event: dict) -> None:
    rows = event["request_steps"]
    h_steps = len(rows)
    d_armed = sum(row["draft_armed"] for row in rows)
    accepted = sum(row["accepted_draft_tokens"] for row in rows)
    clipped = sum(row["clipped_tokens"] for row in rows)
    committed = sum(row["committed_tokens"] for row in rows)
    event["counters"] = {
        "H_target_steps": h_steps,
        "D_armed": d_armed,
        "A_accepted": accepted,
        "C_clipped": clipped,
        "E_committed": committed,
        "U_unarmed": h_steps - d_armed,
        "closure_holds": True,
        "draft_subset_holds": True,
    }


def valid_capture() -> dict:
    """Build one exact R6/K4 capture with 32 fixed-length requests."""
    ids = prompt_ids("R6", 0)
    events = []
    for step_index in range(52):
        accepted = 4 if step_index < 51 else 0
        rows = [
            {
                "request_id": request_id,
                "target_processed": True,
                "draft_armed": True,
                "accepted_draft_tokens": accepted,
                "raw_generated_tokens": accepted + 1,
                "committed_tokens": accepted + 1,
                "clipped_tokens": 0,
            }
            for request_id in ids
        ]
        event = {
            "schema_version": 1,
            "contract_id": "p4-same-event-target-step-v1",
            "event_id": f"r6-k4-event-{step_index}",
            "action_id": "target-matching-k4",
            "k": 4,
            "engine_step_index": 100 + step_index,
            "complete": True,
            "pure_decode": True,
            "source": {
                "scheduler_event_id": f"r6-k4-event-{step_index}",
                "verified_action_id": "target-matching-k4",
                "next_action_id": "target-matching-k4",
                "scheduler_output_observed": True,
                "model_runner_output_observed": True,
                "post_stop_commit_observed": True,
                "prometheus_interval_delta_used": False,
            },
            "quality": {
                "preemptions": 0,
                "recomputed_tokens": 0,
                "invalid_spec_tokens": 0,
            },
            "timing": {
                "engine_event_elapsed_s": 0.001,
                "request_decode_time_s": 0.032,
                "source": "same_scheduler_event_monotonic",
                "queue_time_included": False,
                "prefill_time_included": False,
            },
            "request_steps": rows,
            "counters": {},
            "score_eligible": True,
            "exclusion_reasons": [],
        }
        _refresh_event(event)
        events.append(event)

    manifest = json.loads(MANIFEST_PATH.read_text())
    return {
        "schema_version": 1,
        "capture_contract_id": "p4-b0-same-event-capture-v1",
        "capture_id": "capture-b1-k4-r6-s0-r1",
        "scored": False,
        "complete": True,
        "warmup_complete": True,
        "runner": {
            "preregistration_id": "p4-b0-off-k4-w512-value-screen-v1",
            "prompt_manifest_id": "p4-b0-six-regime-prompts-v1",
            "prompt_manifest_sha256": hashlib.sha256(
                MANIFEST_PATH.read_bytes()
            ).hexdigest(),
            "prompt_bundle_sha256": manifest["bundle"]["sha256"],
            "target_checkpoint_revision": manifest["tokenizer"]["revision"],
            "target_quantization": None,
            "target_kv_dtype": "bfloat16",
            "draft_weight_version": "target-alias-v1",
            "shared_kv_binding_id": "binding-b1-k4",
            "shared_kv_alias_proven": True,
            "true_slot_mapping_id": "slots-b1-k4",
            "true_slot_identity_proven": True,
            "hardware_id": "h100-80gb-gpu4",
            "parallel_layout": "tp1-pp1",
            "kernel_backend": "vllm-cuda",
            "graph_grade": "fullcg-piecewise",
            "warmup_policy": "registered-four-round",
            "measurement_currency": "S_dec",
        },
        "matrix": {
            "boot_block_id": 1,
            "boot_id": "boot-b1-k4",
            "action_id": "target-matching-k4",
            "action_position": 2,
            "action_realization": "live-b0-target-matching-k4",
            "regime_id": "R6",
            "content_seed": 0,
            "round_index": 1,
        },
        "generation": {
            "prompt_record_ids": ids,
            "generation_seed": 0,
            "batch": 32,
            "max_output_tokens": 256,
            "temperature": 0.0,
            "ignore_eos": True,
            "requested_output_tokens": 8192,
        },
        "events": events,
    }


def _tau_counters(work: int, action_id: str, w512_benefit: bool) -> dict:
    if action_id == "off":
        h_steps = work
        accepted = 0
        d_armed = 0
    elif action_id == "target-matching-k4" or not w512_benefit:
        h_steps = work // 4
        accepted = work - h_steps
        d_armed = h_steps
    else:
        h_steps = work * 125 // 512
        accepted = work - h_steps
        d_armed = h_steps
    return {
        "H_target_steps": h_steps,
        "D_armed": d_armed,
        "A_accepted": accepted,
        "C_clipped": 0,
        "E_committed": work,
        "U_unarmed": h_steps - d_armed,
        "closure_holds": True,
        "draft_subset_holds": True,
        "equal_work_holds": True,
    }


def synthetic_rounds(w512_benefit: bool = False) -> list[dict]:
    """Build an exact 432-round matrix with certified constant boots."""
    manifest = json.loads(MANIFEST_PATH.read_text())
    manifest_sha = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    prompt_hashes = {
        (regime_id, seed): _canonical_sha256(prompt_ids(regime_id, seed))
        for regime_id in REGIMES
        for seed in (0, 1)
    }
    rates = {
        "off": 100.0,
        "target-matching-k4": 110.0,
        "target-matching-w512-masked-k4": 108.0,
    }
    rows = []
    for block_id, order in ACTION_ORDERS.items():
        for action_id in (
            "off",
            "target-matching-k4",
            "target-matching-w512-masked-k4",
        ):
            position = order.index(action_id) + 1
            short_action = {
                "off": "off",
                "target-matching-k4": "k4",
                "target-matching-w512-masked-k4": "w512",
            }[action_id]
            for regime_id, (batch, max_output, temperature) in REGIMES.items():
                work = 32 * max_output
                for seed in (0, 1):
                    for round_index in range(1, 5):
                        counters = _tau_counters(work, action_id, w512_benefit)
                        request_time = work / rates[action_id]
                        key = (
                            f"b{block_id}-{short_action}-{regime_id}-"
                            f"s{seed}-r{round_index}"
                        )
                        rows.append(
                            {
                                "schema_version": 1,
                                "round_contract_id": "p4-b0-adapted-round-v1",
                                "adapter_contract_id": ("p4-b0-same-event-adapter-v1"),
                                "capture_id": f"capture-{key}",
                                "complete": True,
                                "score_eligible": True,
                                "matrix": {
                                    "boot_block_id": block_id,
                                    "boot_id": f"boot-b{block_id}-{short_action}",
                                    "action_id": action_id,
                                    "action_position": position,
                                    "action_realization": ACTION_REALIZATIONS[
                                        action_id
                                    ],
                                    "regime_id": regime_id,
                                    "content_seed": seed,
                                    "round_index": round_index,
                                },
                                "match": {
                                    "preregistration_id": (
                                        "p4-b0-off-k4-w512-value-screen-v1"
                                    ),
                                    "prompt_manifest_id": (
                                        "p4-b0-six-regime-prompts-v1"
                                    ),
                                    "prompt_manifest_sha256": manifest_sha,
                                    "prompt_bundle_sha256": manifest["bundle"][
                                        "sha256"
                                    ],
                                    "prompt_record_ids_sha256": prompt_hashes[
                                        (regime_id, seed)
                                    ],
                                    "prompt_count": 32,
                                    "generation_seed": 0,
                                    "batch": batch,
                                    "max_output_tokens": max_output,
                                    "temperature": temperature,
                                    "ignore_eos": True,
                                    "requested_output_tokens": work,
                                    "stable_config_sha256": "1" * 64,
                                },
                                "proofs": {
                                    "shared_kv_binding_id": (
                                        f"binding-b{block_id}-{short_action}"
                                    ),
                                    "shared_kv_alias_proven": True,
                                    "true_slot_mapping_id": (
                                        f"slots-b{block_id}-{short_action}"
                                    ),
                                    "true_slot_identity_proven": True,
                                },
                                "source": {
                                    "event_count": 1,
                                    "rejected_event_count": 0,
                                    "event_ids_sha256": hashlib.sha256(
                                        key.encode()
                                    ).hexdigest(),
                                    "first_engine_step_index": 1,
                                    "last_engine_step_index": 1,
                                },
                                "counters": counters,
                                "timing": {
                                    "request_decode_time_s": request_time,
                                    "source": (
                                        "sum_same_scheduler_event_request_decode_time"
                                    ),
                                },
                                "estimands": {
                                    "decode_rate_req": rates[action_id],
                                    "tau_raw": 1
                                    + counters["A_accepted"]
                                    / counters["H_target_steps"],
                                },
                            }
                        )
    return rows


def _set_rate(row: dict, rate: float) -> None:
    row["timing"]["request_decode_time_s"] = row["counters"]["E_committed"] / rate
    row["estimands"]["decode_rate_req"] = rate


def _find_round(
    rounds: list[dict],
    *,
    block: int,
    action: str,
    regime: str,
    seed: int,
    round_index: int,
) -> dict:
    for row in rounds:
        matrix = row["matrix"]
        if (
            matrix["boot_block_id"],
            matrix["action_id"],
            matrix["regime_id"],
            matrix["content_seed"],
            matrix["round_index"],
        ) == (block, action, regime, seed, round_index):
            return row
    raise AssertionError("synthetic round not found")


class RunnerScorerContractTests(unittest.TestCase):
    """Validate the frozen contract without granting run authority."""

    def test_checked_contract_is_frozen_but_unwired(self) -> None:
        result = validate_runner_scorer_contract(valid_contract())

        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["same_event_measurement_adapter_frozen"])
        self.assertTrue(result["runner_scorer_frozen"])
        self.assertFalse(result["same_event_live_recorder_wired"])
        self.assertEqual(result["expected_round_records"], 432)
        self.assertEqual(result["bootstrap_seed"], 20260809)
        self.assertFalse(result["gpu_measurement_authorized"])
        self.assertFalse(result["p4a_engineering_authorized"])

    def test_checked_validation_output_is_current(self) -> None:
        expected = json.loads(VALIDATION_PATH.read_text())

        self.assertEqual(validate_runner_scorer_contract(valid_contract()), expected)

    def test_rejects_bound_source_hash_drift(self) -> None:
        contract = valid_contract()
        contract["source_artifacts"]["measurement_adapter"]["sha256"] = "0" * 64

        with self.assertRaisesRegex(B0ScorerError, "source hash mismatch"):
            validate_runner_scorer_contract(contract)

    def test_rejects_authority(self) -> None:
        contract = valid_contract()
        contract["authorizations"]["gpu_measurement"] = True

        with self.assertRaises(B0ScorerError):
            validate_runner_scorer_contract(contract)


class SameEventAdapterTests(unittest.TestCase):
    """Accept explicit same-event rows and reject every ambiguous shortcut."""

    def test_adapts_exact_equal_work_capture(self) -> None:
        result = adapt_capture(valid_capture())

        self.assertTrue(result["score_eligible"])
        self.assertEqual(result["matrix"]["regime_id"], "R6")
        self.assertEqual(result["counters"]["E_committed"], 8192)
        self.assertEqual(result["counters"]["H_target_steps"], 1664)
        self.assertEqual(result["counters"]["A_accepted"], 6528)
        self.assertAlmostEqual(
            result["estimands"]["decode_rate_req"], 8192 / (52 * 0.032)
        )

    def test_rejects_transition_event(self) -> None:
        capture = valid_capture()
        capture["events"][0]["source"]["next_action_id"] = "off"

        with self.assertRaisesRegex(B0AdapterError, "steady-state"):
            adapt_capture(capture)

    def test_rejects_incomplete_event(self) -> None:
        capture = valid_capture()
        event = capture["events"][0]
        event["complete"] = False
        event["score_eligible"] = False
        event["exclusion_reasons"] = ["incomplete_event"]

        with self.assertRaisesRegex(B0AdapterError, "not score eligible"):
            adapt_capture(capture)

    def test_rejects_invalid_spec_tokens(self) -> None:
        capture = valid_capture()
        event = capture["events"][0]
        event["quality"]["invalid_spec_tokens"] = 1
        event["score_eligible"] = False
        event["exclusion_reasons"] = ["invalid_spec_tokens"]

        with self.assertRaisesRegex(B0AdapterError, "not score eligible"):
            adapt_capture(capture)

    def test_rejects_prompt_id_substitution(self) -> None:
        capture = valid_capture()
        capture["generation"]["prompt_record_ids"][-1] = "R6-s0-invented"

        with self.assertRaisesRegex(B0AdapterError, "prompt ids differ"):
            adapt_capture(capture)

    def test_rejects_short_equal_work(self) -> None:
        capture = valid_capture()
        row = capture["events"][-1]["request_steps"][0]
        row["committed_tokens"] = 0
        row["clipped_tokens"] = 1
        _refresh_event(capture["events"][-1])

        with self.assertRaisesRegex(B0AdapterError, "exact fixed output"):
            adapt_capture(capture)

    def test_rejects_legacy_p3_aggregate_instead_of_inferring_rows(self) -> None:
        capture = valid_capture()
        legacy = json.loads(P3_TRACE_PATH.read_text().splitlines()[1])
        legacy["event_id"] = "legacy-p3-event-1"
        capture["events"] = [legacy]

        with self.assertRaisesRegex(B0AdapterError, "not canonical"):
            adapt_capture(capture)


class FrozenScorerTests(unittest.TestCase):
    """Exercise episode handling, paired bootstrap, and the value gate."""

    def test_dominance_stops_equal_acceptance_candidate(self) -> None:
        result = score_rounds(valid_contract(), synthetic_rounds())

        self.assertTrue(result["decision"]["dominance_short_circuit"])
        self.assertFalse(result["decision"]["value_gate_pass"])
        self.assertFalse(result["decision"]["authority_granted"])

    def test_mean_branch_passes_value_only_with_acceptance_gain(self) -> None:
        result = score_rounds(valid_contract(), synthetic_rounds(w512_benefit=True))

        self.assertFalse(result["decision"]["dominance_short_circuit"])
        self.assertTrue(result["decision"]["mean_branch"])
        self.assertTrue(result["decision"]["value_gate_pass"])
        self.assertFalse(result["decision"]["authority_granted"])
        self.assertGreater(result["portfolio"]["weighted_gain_lcb"], 0.02)

    def test_bootstrap_is_reproducible(self) -> None:
        rounds = synthetic_rounds(w512_benefit=True)

        first = score_rounds(valid_contract(), rounds)
        second = score_rounds(valid_contract(), rounds)

        self.assertEqual(first, second)

    def test_rejects_one_slow_episode_round(self) -> None:
        rounds = synthetic_rounds(w512_benefit=True)
        row = _find_round(
            rounds,
            block=1,
            action="off",
            regime="R4",
            seed=0,
            round_index=1,
        )
        _set_rate(row, 80.0)

        result = score_rounds(valid_contract(), rounds)

        self.assertEqual(result["episode_rejected_rounds"], 1)

    def test_b1_second_highest_reference_tolerates_one_fast_round(self) -> None:
        rounds = synthetic_rounds(w512_benefit=True)
        row = _find_round(
            rounds,
            block=1,
            action="off",
            regime="R1",
            seed=0,
            round_index=1,
        )
        _set_rate(row, 106.0)

        result = score_rounds(valid_contract(), rounds)

        audit = [
            item
            for item in result["episode_audit"]
            if item["action_id"] == "off"
            and item["regime_id"] == "R1"
            and item["content_seed"] == 0
        ][0]
        self.assertEqual(audit["reference_rate"], 100.0)
        self.assertEqual(audit["rejected_rounds"], 0)

    def test_rejects_boot_with_fewer_than_two_surviving_rounds(self) -> None:
        rounds = synthetic_rounds()
        for round_index in (1, 2, 3):
            row = _find_round(
                rounds,
                block=1,
                action="off",
                regime="R4",
                seed=0,
                round_index=round_index,
            )
            _set_rate(row, 80.0)

        with self.assertRaisesRegex(B0ScorerError, "fewer than two"):
            score_rounds(valid_contract(), rounds)

    def test_rejects_cross_boot_disagreement_over_two_percent(self) -> None:
        rounds = synthetic_rounds()
        for round_index in range(1, 5):
            row = _find_round(
                rounds,
                block=1,
                action="off",
                regime="R4",
                seed=0,
                round_index=round_index,
            )
            _set_rate(row, 97.0)

        with self.assertRaisesRegex(B0ScorerError, "exceeded 2%"):
            score_rounds(valid_contract(), rounds)

    def test_rejects_missing_matrix_round(self) -> None:
        rounds = synthetic_rounds()
        rounds.pop()

        with self.assertRaisesRegex(B0ScorerError, "exactly 432"):
            score_rounds(valid_contract(), rounds)

    def test_rejects_prompt_hash_drift(self) -> None:
        rounds = synthetic_rounds()
        rounds[0]["match"]["prompt_record_ids_sha256"] = "0" * 64

        with self.assertRaisesRegex(B0ScorerError, "prompt-id hash drift"):
            score_rounds(valid_contract(), rounds)

    def test_rejects_non_raw_rate(self) -> None:
        rounds = synthetic_rounds()
        rounds[0]["estimands"]["decode_rate_req"] += 1

        with self.assertRaisesRegex(B0ScorerError, "not raw E/time"):
            score_rounds(valid_contract(), rounds)


if __name__ == "__main__":
    unittest.main()
