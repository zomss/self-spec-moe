"""CPU-only proof for the Phase 97 chunked-prefill cohort barrier."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR / "scripts"))

from validate_p4_b0_chunked_prefill_cohort_barrier import (  # noqa: E402
    CPU_PROOF_DECODE_TOKENS,
    FROZEN_IDS,
    OFF_ACTION_ID,
    CohortBarrierProofError,
    _new_barrier,
    _open_release,
    _runtime_ids,
    _states,
    run_cpu_proof,
    validate_artifact,
)

from vllm.v1.spec_decode.koff_runtime import (  # noqa: E402
    K4_ACTION_ID,
    KOffRuntimeError,
    P4CaptureCohortBarrier,
)

ARTIFACT_PATH = (
    PHASE_DIR
    / "data"
    / "p4"
    / "p4_b0_chunked_prefill_cohort_barrier_design_and_cpu_proof.json"
)
VALIDATION_PATH = (
    PHASE_DIR / "data" / "p4" / "p4_b0_chunked_prefill_cohort_barrier_validation.json"
)


def valid_artifact() -> dict:
    """Return an independent copy of the checked cohort-barrier artifact."""
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


class P4B0CohortBarrierStateMachineTests(unittest.TestCase):
    """Prove exact membership, early hold, release, abort, and accounting."""

    def test_complete_cpu_proof_covers_all_actions_and_failures(self) -> None:
        proof = run_cpu_proof()

        self.assertEqual(proof["status"], "pass")
        self.assertEqual(proof["positive_case_count"], 3)
        self.assertEqual(proof["rejection_case_count"], 10)
        self.assertEqual(proof["abort_case_count"], 1)
        self.assertFalse(proof["gpu_executed"])
        self.assertFalse(proof["state_leak_across_cohorts"])

    def test_early_prefill_is_held_without_finishing_request(self) -> None:
        barrier, runtime_ids = _new_barrier()

        gate = barrier.begin_step(_states(runtime_ids, [6, 4, 2], [1, 0, 0]))

        self.assertEqual(barrier.state, "prefilling")
        self.assertFalse(gate["release"])
        self.assertEqual(gate["held_request_ids"], (runtime_ids[0],))
        self.assertEqual(barrier.abort_request_ids, runtime_ids)

    def test_release_waits_for_all_and_preserves_one_prefill_sample(self) -> None:
        barrier, runtime_ids = _new_barrier()

        gate = _open_release(barrier, runtime_ids)

        self.assertTrue(gate["release"])
        self.assertEqual(barrier.state, "released")
        self.assertEqual(barrier.summary()["unmeasured_prefill_tokens_per_request"], 1)

    def test_member_abort_returns_every_unfinished_member(self) -> None:
        barrier, runtime_ids = _new_barrier()

        abort_ids = barrier.abort_member(runtime_ids[1])

        self.assertEqual(abort_ids, runtime_ids)
        self.assertEqual(barrier.state, "aborted")
        self.assertIn("member", barrier.abort_reason)

    def test_randomized_internal_id_must_keep_exact_queue_order(self) -> None:
        barrier = P4CaptureCohortBarrier(
            "order",
            FROZEN_IDS,
            action_id=OFF_ACTION_ID,
            measured_decode_tokens=CPU_PROOF_DECODE_TOKENS,
        )
        runtime_ids = _runtime_ids()

        with self.assertRaisesRegex(KOffRuntimeError, "order drifted"):
            barrier.register(runtime_ids[1])

        self.assertEqual(barrier.state, "aborted")

    def test_malformed_random_suffix_fails_closed(self) -> None:
        barrier = P4CaptureCohortBarrier(
            "suffix",
            FROZEN_IDS,
            action_id=OFF_ACTION_ID,
            measured_decode_tokens=CPU_PROOF_DECODE_TOKENS,
        )

        with self.assertRaisesRegex(KOffRuntimeError, "exactly eight lowercase"):
            barrier.register(f"{FROZEN_IDS[0]}-ABCDEF12")

        self.assertEqual(barrier.state, "aborted")

    def test_snapshot_preemption_aborts_before_release(self) -> None:
        barrier, runtime_ids = _new_barrier()

        with self.assertRaisesRegex(KOffRuntimeError, "was preempted"):
            barrier.begin_step(
                _states(
                    runtime_ids,
                    [0, 0, 0],
                    [0, 0, 0],
                    preemptions=[0, 1, 0],
                )
            )

        self.assertEqual(barrier.state, "aborted")

    def test_release_rejects_wrong_action_and_query_width(self) -> None:
        barrier, runtime_ids = _new_barrier()
        _open_release(barrier, runtime_ids)

        with self.assertRaisesRegex(KOffRuntimeError, "another action"):
            barrier.end_step(
                scheduled_request_ids=runtime_ids,
                pure_decode=True,
                action_id=K4_ACTION_ID,
                target_query_widths=dict.fromkeys(runtime_ids, 1),
                committed_tokens=dict.fromkeys(runtime_ids, 1),
            )

        barrier, runtime_ids = _new_barrier(K4_ACTION_ID)
        _open_release(barrier, runtime_ids)
        with self.assertRaisesRegex(KOffRuntimeError, "query width"):
            barrier.end_step(
                scheduled_request_ids=runtime_ids,
                pure_decode=True,
                action_id=K4_ACTION_ID,
                target_query_widths=dict.fromkeys(runtime_ids, 1),
                committed_tokens=dict.fromkeys(runtime_ids, 1),
            )

    def test_incomplete_close_is_terminal_failure(self) -> None:
        barrier, _ = _new_barrier()

        with self.assertRaisesRegex(KOffRuntimeError, "before exact work"):
            barrier.close()

        self.assertEqual(barrier.state, "aborted")

    def test_frontend_offset_must_be_exact_at_finish(self) -> None:
        barrier, runtime_ids = _new_barrier(measured_decode_tokens=1)
        _open_release(barrier, runtime_ids)
        barrier.end_step(
            scheduled_request_ids=runtime_ids,
            pure_decode=True,
            action_id=OFF_ACTION_ID,
            target_query_widths=dict.fromkeys(runtime_ids, 1),
            committed_tokens=dict.fromkeys(runtime_ids, 1),
        )

        with self.assertRaisesRegex(KOffRuntimeError, "prefill offset"):
            barrier.finish_request(runtime_ids[0], 1)


class P4B0CohortBarrierArtifactTests(unittest.TestCase):
    """Validate source closure and ensure the CPU proof grants no GPU authority."""

    def test_checked_artifact_passes_without_gpu_authority(self) -> None:
        result = validate_artifact(valid_artifact())

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["bounded_prefill_budget"], 8192)
        self.assertEqual(result["effective_scheduler_token_budget"], 8160)
        self.assertEqual(result["r5_minimum_prefill_events"], 14)
        self.assertEqual(result["r5cot_minimum_prefill_events"], 14)
        self.assertFalse(result["gpu_executed"])
        self.assertFalse(result["gpu_probe_authorized"])
        self.assertFalse(result["v10_authorized"])

    def test_checked_cpu_proof_is_reproducible(self) -> None:
        self.assertEqual(valid_artifact()["cpu_proof"], run_cpu_proof())

    def test_checked_validation_summary_is_current(self) -> None:
        expected = json.loads(VALIDATION_PATH.read_text(encoding="utf-8"))

        self.assertEqual(validate_artifact(valid_artifact()), expected)

    def test_rejects_source_hash_drift(self) -> None:
        artifact = valid_artifact()
        artifact["source_artifacts"]["v9_failure"]["sha256"] = "0" * 64

        with self.assertRaisesRegex(CohortBarrierProofError, "source hash"):
            validate_artifact(artifact)

    def test_rejects_full_microbatch_reenable(self) -> None:
        artifact = valid_artifact()
        artifact["bounded_prefill_contract"]["full_microbatch_prefill_allowed"] = True

        with self.assertRaisesRegex(CohortBarrierProofError, "bounded-prefill"):
            validate_artifact(artifact)

    def test_rejects_barrier_on_ordinary_serving(self) -> None:
        artifact = valid_artifact()
        artifact["ordinary_serving_boundary"]["cohort_barrier_enabled"] = True

        with self.assertRaisesRegex(CohortBarrierProofError, "ordinary-serving"):
            validate_artifact(artifact)

    def test_rejects_optimistic_live_wiring_claim(self) -> None:
        artifact = valid_artifact()
        artifact["claims"]["live_engine_wired"] = True

        with self.assertRaisesRegex(CohortBarrierProofError, "claims drifted"):
            validate_artifact(artifact)

    def test_rejects_gpu_or_v10_authority(self) -> None:
        for authority in ("gpu_probe", "v10_value_screen"):
            with self.subTest(authority=authority):
                artifact = copy.deepcopy(valid_artifact())
                artifact["authorizations"][authority] = True

                with self.assertRaisesRegex(CohortBarrierProofError, "must not grant"):
                    validate_artifact(artifact)

    def test_rejects_self_authorizing_next_probe(self) -> None:
        artifact = valid_artifact()
        artifact["next_artifact"]["separate_source_bound_authorization_required"] = (
            False
        )

        with self.assertRaisesRegex(CohortBarrierProofError, "next-artifact"):
            validate_artifact(artifact)


if __name__ == "__main__":
    unittest.main()
