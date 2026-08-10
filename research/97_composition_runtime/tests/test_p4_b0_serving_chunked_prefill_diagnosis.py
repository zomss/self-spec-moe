"""CPU checks for the consumed real-serving chunked-prefill diagnosis."""

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

from run_p4_b0_serving_chunked_prefill_diagnosis import (  # noqa: E402
    EFFECTIVE_SCHEDULER_BUDGET,
    MAX_NUM_BATCHED_TOKENS,
    MAX_OUTPUT_TOKENS,
    OUTPUT_PATH,
    PROMPT_IDS,
    ServingDiagnosisError,
    _analyze_trace,
    validate_authorization,
)

AUTHORIZATION_PATH = (
    PHASE_DIR
    / "data"
    / "p4"
    / "p4_b0_serving_chunked_prefill_diagnosis_authorization.json"
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


class P4B0ServingChunkedPrefillDiagnosisTests(unittest.TestCase):
    """Prove the diagnosis is exact, non-scored, and already consumed."""

    def test_checked_in_authorization_is_historical_after_repair(self) -> None:
        with self.assertRaisesRegex(ServingDiagnosisError, "source hash drifted"):
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
        self.assertTrue(run["engine"]["enable_chunked_prefill"])
        self.assertEqual(run["engine"]["gpu_memory_utilization"], 0.90)

    def test_create_new_authority_is_consumed(self) -> None:
        self.assertTrue(OUTPUT_PATH.is_dir())
        with self.assertRaisesRegex(ServingDiagnosisError, "output already exists"):
            validate_authorization(
                _authorization_with_current_sources(),
                require_output_absent=True,
            )

    def test_result_is_non_scored_pass(self) -> None:
        result = json.loads((OUTPUT_PATH / "diagnosis.json").read_text())

        self.assertEqual(result["status"], "pass")
        self.assertFalse(result["scored"])
        self.assertEqual(result["trace"]["mixed_step_count"], 8)
        self.assertEqual(result["trace"]["pure_decode_step_count"], 15)
        self.assertTrue(result["invariants"]["mixed_steps_force_q1_off"])
        self.assertTrue(result["invariants"]["mixed_steps_dispatch_no_draft"])
        self.assertFalse(result["claims"]["value_screen_repaired"])

    def test_trace_reanalysis_matches_checked_in_result(self) -> None:
        outputs = json.loads((OUTPUT_PATH / "request_outputs.json").read_text())
        counts = {
            request_id: len(token_ids)
            for request_id, token_ids in outputs["request_token_ids"].items()
        }
        result = _analyze_trace(OUTPUT_PATH / "koff_trace.jsonl", counts)

        self.assertEqual(counts, dict.fromkeys(PROMPT_IDS, MAX_OUTPUT_TOKENS))
        self.assertEqual(
            result,
            json.loads((OUTPUT_PATH / "diagnosis.json").read_text()),
        )

    def test_authority_does_not_include_value_screen_or_admission(self) -> None:
        authority = _authorization()["authorizations"]

        self.assertTrue(authority["serving_diagnosis"])
        self.assertFalse(authority["value_screen"])
        self.assertFalse(authority["p4a_engineering"])
        self.assertFalse(authority["action_admission"])
        self.assertFalse(authority["production_value_claim"])

    def test_rejects_source_hash_drift(self) -> None:
        authorization = copy.deepcopy(_authorization_with_current_sources())
        authorization["source_artifacts"]["scheduler"]["sha256"] = "0" * 64

        with self.assertRaisesRegex(ServingDiagnosisError, "scheduler"):
            validate_authorization(authorization, require_output_absent=False)

    def test_rejects_serving_budget_drift(self) -> None:
        authorization = copy.deepcopy(_authorization_with_current_sources())
        authorization["run_contract"]["engine"]["max_num_batched_tokens"] = 114688

        with self.assertRaisesRegex(ServingDiagnosisError, "run contract"):
            validate_authorization(authorization, require_output_absent=False)

    def test_rejects_fallback_gpu(self) -> None:
        authorization = copy.deepcopy(_authorization_with_current_sources())
        authorization["run_contract"]["gpu"]["fallback_authorized"] = True

        with self.assertRaisesRegex(ServingDiagnosisError, "run contract"):
            validate_authorization(authorization, require_output_absent=False)


if __name__ == "__main__":
    unittest.main()
