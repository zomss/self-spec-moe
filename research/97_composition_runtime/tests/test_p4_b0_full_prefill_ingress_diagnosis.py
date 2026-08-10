"""CPU checks for the consumed full-prefill ingress diagnosis."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
sys.path.insert(0, str(PHASE_DIR / "scripts"))

from run_p4_b0_full_prefill_ingress_diagnosis import (  # noqa: E402
    EFFECTIVE_SCHEDULER_BUDGET,
    MAX_NUM_BATCHED_TOKENS,
    OUTPUT_PATH,
    FullPrefillIngressError,
    _analyze,
    validate_authorization,
)

AUTHORIZATION_PATH = (
    PHASE_DIR
    / "data"
    / "p4"
    / "p4_b0_full_prefill_ingress_diagnosis_authorization.json"
)


def _authorization() -> dict:
    return json.loads(AUTHORIZATION_PATH.read_text(encoding="utf-8"))


def _authorization_with_current_sources() -> dict:
    authorization = _authorization()
    for artifact in authorization["source_artifacts"].values():
        artifact["sha256"] = hashlib.sha256(
            (REPO_ROOT / artifact["path"]).read_bytes()
        ).hexdigest()
    return authorization


def _result() -> dict:
    return json.loads((OUTPUT_PATH / "diagnosis.json").read_text(encoding="utf-8"))


class P4B0FullPrefillIngressDiagnosisTests(unittest.TestCase):
    """Prove the one-boot result is exact, non-scored, and consumed."""

    def test_checked_in_authorization_is_historical_after_repair(self) -> None:
        with self.assertRaisesRegex(FullPrefillIngressError, "source hash drifted"):
            validate_authorization(_authorization(), require_output_absent=False)

        run = validate_authorization(
            _authorization_with_current_sources(),
            require_output_absent=False,
        )

        self.assertEqual(
            run["engine"]["max_num_batched_tokens"], MAX_NUM_BATCHED_TOKENS
        )
        self.assertEqual(
            run["engine"]["effective_scheduler_token_budget"],
            EFFECTIVE_SCHEDULER_BUDGET,
        )
        self.assertEqual(run["engine"]["gpu_memory_utilization"], 0.96)

    def test_create_new_authority_is_consumed(self) -> None:
        self.assertTrue(OUTPUT_PATH.is_dir())
        with self.assertRaisesRegex(FullPrefillIngressError, "output already exists"):
            validate_authorization(
                _authorization_with_current_sources(),
                require_output_absent=True,
            )

    def test_result_identifies_only_mixed_batch_exclusion(self) -> None:
        result = _result()

        self.assertEqual(result["status"], "pass")
        self.assertFalse(result["scored"])
        self.assertEqual(result["decision"], "v5_first_event_exclusion_identified")
        self.assertEqual(
            result["trace"]["first_decode_p4_exclusions"],
            ["prefill_or_mixed_batch"],
        )
        self.assertEqual(result["trace"]["preemptions"], 0)
        self.assertEqual(result["trace"]["recomputed_tokens"], 0)
        self.assertFalse(result["claims"]["value_screen_repaired"])

    def test_trace_proves_staggered_request_ingress(self) -> None:
        rows = [
            json.loads(line)
            for line in (OUTPUT_PATH / "koff_trace.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        steps = [row for row in rows if row["record_type"] == "koff_engine_step"]

        self.assertEqual(len(steps), 2)
        self.assertEqual(
            steps[0]["execution"]["diagnostic"]["num_scheduled_tokens"],
            [8077],
        )
        self.assertEqual(steps[0]["counters"]["H_target_steps"], 0)
        self.assertEqual(
            steps[1]["execution"]["diagnostic"]["num_scheduled_tokens"],
            [1, 7839, 8368, 8435, 7836, 7876, 8597, 8172],
        )
        self.assertEqual(steps[1]["counters"]["H_target_steps"], 1)

    def test_trace_reanalysis_matches_checked_in_result(self) -> None:
        result = _result()
        reanalyzed = _analyze(
            OUTPUT_PATH / "koff_trace.jsonl",
            OUTPUT_PATH / "captures",
            result["observed_failure"],
        )

        self.assertEqual(reanalyzed, result)

    def test_recorder_failed_closed_without_score(self) -> None:
        capture = OUTPUT_PATH / "captures" / "capture-b1-p1-off-r4-s0-r1.json"

        self.assertTrue(capture.is_file())
        self.assertEqual(capture.stat().st_size, 0)
        self.assertFalse((OUTPUT_PATH / "score.json").exists())
        self.assertEqual(_result()["capture"]["complete_capture_count"], 0)

    def test_shared_target_kv_and_weight_aliases_remained_live(self) -> None:
        invariants = _result()["invariants"]

        self.assertEqual(invariants["shared_target_kv_layer_count"], 36)
        self.assertFalse(invariants["private_draft_kv_allocated"])
        self.assertEqual(invariants["target_draft_weight_alias_count"], 291)

    def test_authority_does_not_include_screen_or_admission(self) -> None:
        authority = _authorization()["authorizations"]

        self.assertTrue(authority["ingress_diagnosis"])
        self.assertFalse(authority["value_screen"])
        self.assertFalse(authority["p4a_engineering"])
        self.assertFalse(authority["action_admission"])
        self.assertFalse(authority["production_value_claim"])

    def test_rejects_bound_source_hash_drift(self) -> None:
        authorization = copy.deepcopy(_authorization_with_current_sources())
        authorization["source_artifacts"]["scheduler"]["sha256"] = "0" * 64

        with self.assertRaisesRegex(FullPrefillIngressError, "scheduler"):
            validate_authorization(authorization, require_output_absent=False)


if __name__ == "__main__":
    unittest.main()
