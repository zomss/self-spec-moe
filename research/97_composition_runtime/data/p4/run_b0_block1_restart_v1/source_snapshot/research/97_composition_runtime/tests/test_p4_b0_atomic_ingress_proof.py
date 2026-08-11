"""CPU checks for the Phase 97 atomic-ingress proof gate."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
sys.path.insert(0, str(PHASE_DIR / "scripts"))

from run_p4_b0_atomic_ingress_proof import (  # noqa: E402
    AUTHORIZATION_PATH,
    EFFECTIVE_SCHEDULER_BUDGET,
    ENGINE_CORE_CLASS,
    OUTPUT_PATH,
    PROMPT_COUNTS,
    PROMPT_IDS,
    AtomicIngressProofError,
    _analyze,
    _child_env,
    validate_authorization,
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


def _resources() -> dict:
    return {
        "binding_id": "binding",
        "pool_id": "pool",
        "shared_target_kv_block_capacity": 22090,
        "true_slot_mapping_id": "slots",
    }


def _execution(
    *, scheduled: list[int], computed: list[int], request_ids: list[str]
) -> dict:
    return {
        "diagnostic": {
            "request_ids": request_ids,
            "num_scheduled_tokens": scheduled,
            "num_computed_tokens": computed,
            "num_prompt_tokens": list(PROMPT_COUNTS),
        },
        "draft_dispatched": False,
        "shared_kv_layer_count": 36,
        "shared_kv_storage_alias_count": 36,
        "shared_weight_parameter_count": 291,
        "target_weight_version_id": "target-alias",
        "draft_weight_version_id": "target-alias",
    }


def _synthetic_rows() -> list[dict]:
    request_ids = [
        f"{record_id}-{index:08x}" for index, record_id in enumerate(PROMPT_IDS)
    ]
    return [
        {
            "record_type": "koff_runtime_header",
            "environment": {
                "max_num_batched_tokens": EFFECTIVE_SCHEDULER_BUDGET,
            },
        },
        {
            "record_type": "koff_engine_step",
            "engine_step_index": 0,
            "eligible_for_p3_replay": False,
            "exclusion_reasons": ["no_decode_action", "prefill_or_mixed_batch"],
            "verified_action_id": None,
            "next_action_id": "off",
            "counters": {
                "H_target_steps": 0,
                "preemptions": 0,
                "recomputed_tokens": 0,
            },
            "execution": _execution(
                scheduled=list(PROMPT_COUNTS),
                computed=[0] * len(PROMPT_IDS),
                request_ids=request_ids,
            ),
            "resources": _resources(),
        },
        {
            "record_type": "koff_engine_step",
            "engine_step_index": 1,
            "eligible_for_p3_replay": True,
            "exclusion_reasons": [],
            "verified_action_id": "off",
            "next_action_id": "off",
            "counters": {
                "H_target_steps": len(PROMPT_IDS),
                "preemptions": 0,
                "recomputed_tokens": 0,
            },
            "execution": _execution(
                scheduled=[1] * len(PROMPT_IDS),
                computed=list(PROMPT_COUNTS),
                request_ids=request_ids,
            ),
            "resources": _resources(),
        },
    ]


def _analyze_rows(rows: list[dict]) -> dict:
    with tempfile.TemporaryDirectory() as directory:
        trace_path = Path(directory) / "trace.jsonl"
        trace_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        return _analyze(trace_path, ENGINE_CORE_CLASS)


class P4B0AtomicIngressProofTests(unittest.TestCase):
    """Prove the gate is atomic, non-scored, exact, and fail-closed."""

    def test_inproc_client_defers_execution_until_get_output(self) -> None:
        from vllm.v1.engine.core_client import InprocClient

        class FakeEngineCore:
            def __init__(self) -> None:
                self.added: list[tuple[str, int]] = []
                self.step_calls = 0
                self.post_step_calls = 0

            def preprocess_add_request(self, request: str) -> tuple[str, int]:
                return f"prepared-{request}", 0

            def add_request(self, request: str, request_wave: int) -> None:
                self.added.append((request, request_wave))

            def step_fn(self) -> tuple[dict[int, str], bool]:
                self.step_calls += 1
                return {0: "output"}, True

            def post_step(self, *, model_executed: bool) -> None:
                self.assert_model_executed = model_executed
                self.post_step_calls += 1

        core = FakeEngineCore()
        client = InprocClient.__new__(InprocClient)
        client.engine_core = core

        client.add_request("r0")
        client.add_request("r1")

        self.assertEqual(core.added, [("prepared-r0", 0), ("prepared-r1", 0)])
        self.assertEqual(core.step_calls, 0)
        self.assertEqual(client.get_output(), "output")
        self.assertEqual(core.step_calls, 1)
        self.assertEqual(core.post_step_calls, 1)
        self.assertTrue(core.assert_model_executed)

    def test_child_environment_disables_multiprocessing_and_capture(self) -> None:
        environment = _child_env(Path("atomic-ingress-test-trace.jsonl"))

        self.assertEqual(environment["VLLM_ENABLE_V1_MULTIPROCESSING"], "0")
        self.assertEqual(environment["VLLM_SELF_SPEC_P4_CAPTURE_CONFIG"], "")
        self.assertEqual(environment["VLLM_SELF_SPEC_P4_CAPTURE_OUTPUT"], "")
        self.assertEqual(environment["VLLM_SELF_SPEC_P4_BOOT_ACTION"], "")
        self.assertEqual(environment["VLLM_SELF_SPEC_P4_MIN_KV_BLOCKS"], "0")

    def test_child_environment_parses_capture_conformance_as_disabled(self) -> None:
        from vllm.v1.spec_decode.koff_runtime import options_from_env

        environment = _child_env(Path("atomic-ingress-test-trace.jsonl"))
        with patch.dict(os.environ, environment, clear=False):
            options = options_from_env()

        self.assertEqual(options.p4_capture_config_path, "")
        self.assertEqual(options.p4_capture_output_path, "")
        self.assertEqual(options.p4_boot_action_id, "")
        self.assertEqual(options.p4_logical_weight_version, "")
        self.assertEqual(options.p4_min_kv_blocks, 0)

    def test_synthetic_atomic_trace_passes(self) -> None:
        result = _analyze_rows(_synthetic_rows())

        self.assertEqual(result["status"], "pass")
        self.assertFalse(result["scored"])
        self.assertTrue(result["claims"]["atomic_ingress_proven"])
        self.assertEqual(result["trace"]["initial_prefill_request_count"], 8)
        self.assertEqual(result["trace"]["first_decode_request_count"], 8)
        self.assertEqual(result["trace"]["first_decode_exclusion_reasons"], [])

    def test_exact_unsuffixed_request_ids_pass(self) -> None:
        rows = _synthetic_rows()
        rows[1]["execution"]["diagnostic"]["request_ids"] = list(PROMPT_IDS)
        rows[2]["execution"]["diagnostic"]["request_ids"] = list(PROMPT_IDS)

        self.assertEqual(_analyze_rows(rows)["status"], "pass")

    def test_loose_or_malformed_request_id_suffixes_fail_closed(self) -> None:
        suffixes = ("synthetic", "1234567", "123456789", "1234567g", "ABCDEF12")
        for suffix in suffixes:
            with self.subTest(suffix=suffix):
                rows = _synthetic_rows()
                rows[1]["execution"]["diagnostic"]["request_ids"][0] = (
                    f"{PROMPT_IDS[0]}-{suffix}"
                )

                with self.assertRaisesRegex(
                    AtomicIngressProofError,
                    "exactly eight lowercase hex",
                ):
                    _analyze_rows(rows)

    def test_unknown_request_id_fails_closed(self) -> None:
        rows = _synthetic_rows()
        rows[1]["execution"]["diagnostic"]["request_ids"][0] = "unknown-prompt-00000000"

        with self.assertRaisesRegex(AtomicIngressProofError, "request identity"):
            _analyze_rows(rows)

    def test_request_order_drift_fails_closed_after_canonicalization(self) -> None:
        rows = _synthetic_rows()
        request_ids = rows[1]["execution"]["diagnostic"]["request_ids"]
        request_ids[0], request_ids[1] = request_ids[1], request_ids[0]

        with self.assertRaisesRegex(AtomicIngressProofError, "request order drifted"):
            _analyze_rows(rows)

    def test_synthetic_mixed_decode_fails_closed(self) -> None:
        rows = _synthetic_rows()
        rows[2]["eligible_for_p3_replay"] = False
        rows[2]["exclusion_reasons"] = ["prefill_or_mixed_batch"]

        with self.assertRaisesRegex(
            AtomicIngressProofError, "first decode was score-ineligible"
        ):
            _analyze_rows(rows)

    @unittest.skipUnless(AUTHORIZATION_PATH.is_file(), "authorization not frozen")
    def test_checked_in_authorization_is_historical_after_repair(self) -> None:
        with self.assertRaisesRegex(
            AtomicIngressProofError,
            "source hash drifted for proof_runner",
        ):
            validate_authorization(_authorization(), require_output_absent=False)

        authorization = _authorization_with_current_sources()
        run = validate_authorization(authorization, require_output_absent=False)
        self.assertFalse(run["engine"]["v1_multiprocessing"])
        self.assertEqual(run["engine"]["engine_core_class"], ENGINE_CORE_CLASS)
        self.assertFalse(authorization["authorizations"]["value_screen"])

    @unittest.skipUnless(AUTHORIZATION_PATH.is_file(), "authorization not frozen")
    def test_authorization_rejects_multiprocess_drift(self) -> None:
        authorization = _authorization_with_current_sources()
        authorization["run_contract"]["engine"]["v1_multiprocessing"] = True

        with self.assertRaisesRegex(AtomicIngressProofError, "run contract"):
            validate_authorization(authorization, require_output_absent=False)

    @unittest.skipUnless(AUTHORIZATION_PATH.is_file(), "authorization not frozen")
    def test_authorization_rejects_source_hash_drift(self) -> None:
        authorization = _authorization_with_current_sources()
        authorization["source_artifacts"]["engine_core_client"]["sha256"] = "0" * 64

        with self.assertRaisesRegex(AtomicIngressProofError, "engine_core_client"):
            validate_authorization(authorization, require_output_absent=False)

    @unittest.skipUnless(OUTPUT_PATH.is_dir(), "proof not executed")
    def test_checked_in_proof_matches_live_trace(self) -> None:
        result = json.loads((OUTPUT_PATH / "proof.json").read_text(encoding="utf-8"))

        self.assertEqual(
            _analyze(OUTPUT_PATH / "koff_trace.jsonl", ENGINE_CORE_CLASS),
            result,
        )
        self.assertTrue(result["claims"]["atomic_ingress_proven"])
        self.assertFalse(result["claims"]["v6_authorized"])

    @unittest.skipUnless(OUTPUT_PATH.is_dir(), "proof not executed")
    def test_create_new_authority_is_consumed(self) -> None:
        with self.assertRaisesRegex(AtomicIngressProofError, "output already exists"):
            validate_authorization(
                _authorization_with_current_sources(),
                require_output_absent=True,
            )


if __name__ == "__main__":
    unittest.main()
