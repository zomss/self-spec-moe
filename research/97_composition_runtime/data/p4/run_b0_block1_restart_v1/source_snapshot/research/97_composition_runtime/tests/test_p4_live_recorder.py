"""CPU integration tests for the Phase 97 synchronous live recorder."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

import torch

PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR / "scripts"))

from adapt_p4_b0_same_event import adapt_capture  # noqa: E402

from vllm.v1.spec_decode.koff_runtime import (  # noqa: E402
    K4_ACTION_ID,
    OFF_ACTION_ID,
    P4_ACTION_ORDERS,
    P4_ACTION_REALIZATIONS,
    P4_MIN_SHARED_KV_BLOCKS,
    W512_ACTION_ID,
    KOffRunnerEvidence,
    P4SameEventRecorder,
    SharedKVIdentity,
    SharedWeightIdentity,
    build_same_event_record,
    make_runner_evidence,
    make_scheduler_metadata,
)

MANIFEST_PATH = PHASE_DIR / "data" / "p4" / "p4_b0_prompt_manifest.json"
LOGICAL_WEIGHT_VERSION = "target-matching-config-sha256-" + "b" * 64
REGIMES = ("R4", "R5", "R5cot", "R8", "R1", "R6")


def _prompt_ids() -> list[str]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    rows = [
        row
        for row in manifest["prompts"]
        if row["regime_id"] == "R6" and row["content_seed"] == 0
    ]
    rows.sort(key=lambda row: row["prompt_index"])
    return [row["record_id"] for row in rows]


def _capture_config(request_ids: list[str]) -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text())
    return {
        "schema_version": 1,
        "capture_contract_id": "p4-b0-same-event-capture-v1",
        "capture_id": "capture-live-b1-k4-r6-s0-r1",
        "scored": False,
        "complete": False,
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
            "draft_weight_version": "pending-live-proof",
            "shared_kv_binding_id": "pending-live-proof",
            "shared_kv_alias_proven": True,
            "true_slot_mapping_id": "pending-live-proof",
            "true_slot_identity_proven": True,
            "hardware_id": "cpu-contract-test",
            "parallel_layout": "tp1-pp1",
            "kernel_backend": "vllm-cuda",
            "graph_grade": "fullcg-piecewise",
            "warmup_policy": "registered-four-round",
            "measurement_currency": "S_dec",
        },
        "matrix": {
            "boot_block_id": 1,
            "boot_id": "boot-b1-k4",
            "action_id": K4_ACTION_ID,
            "action_position": 2,
            "action_realization": "live-b0-target-matching-k4",
            "regime_id": "R6",
            "content_seed": 0,
            "round_index": 1,
        },
        "generation": {
            "prompt_record_ids": request_ids,
            "generation_seed": 0,
            "batch": 32,
            "max_output_tokens": 256,
            "temperature": 0.0,
            "ignore_eos": True,
            "requested_output_tokens": 8192,
        },
        "events": [],
    }


def _action_plan(request_ids: list[str], action_id: str) -> dict:
    base = _capture_config(request_ids)
    action_position = P4_ACTION_ORDERS[1].index(action_id) + 1
    action_slug = {
        OFF_ACTION_ID: "off",
        K4_ACTION_ID: "k4",
        W512_ACTION_ID: "w512",
    }[action_id]
    boot_id = f"boot-b1-{action_slug}"
    cells = []
    for regime_id in REGIMES:
        for content_seed in (0, 1):
            for round_index in (1, 2, 3, 4):
                cell = copy.deepcopy(base)
                cell["capture_id"] = (
                    f"capture-b1-p{action_position}-{action_slug}-"
                    f"{regime_id.lower()}-"
                    f"s{content_seed}-r{round_index}"
                )
                cell["runner"]["draft_weight_version"] = LOGICAL_WEIGHT_VERSION
                cell["matrix"] = {
                    "boot_block_id": 1,
                    "boot_id": boot_id,
                    "action_id": action_id,
                    "action_position": action_position,
                    "action_realization": P4_ACTION_REALIZATIONS[action_id],
                    "regime_id": regime_id,
                    "content_seed": content_seed,
                    "round_index": round_index,
                }
                cell["generation"]["max_output_tokens"] = 1
                cell["generation"]["requested_output_tokens"] = 32
                cells.append(cell)
    return {
        "schema_version": 1,
        "capture_plan_contract_id": "p4-b0-same-boot-capture-plan-v1",
        "boot_id": boot_id,
        "boot_action_id": action_id,
        "logical_draft_weight_version": LOGICAL_WEIGHT_VERSION,
        "minimum_shared_kv_blocks": P4_MIN_SHARED_KV_BLOCKS,
        "cells": cells,
    }


def _w512_plan(request_ids: list[str]) -> dict:
    return _action_plan(request_ids, W512_ACTION_ID)


def _action_event(
    recorder: P4SameEventRecorder,
    request_ids: list[str],
    *,
    action_id: str,
    step_index: int,
) -> tuple[dict, KOffRunnerEvidence]:
    live_action_id = K4_ACTION_ID if action_id == W512_ACTION_ID else action_id
    draft_armed = live_action_id == K4_ACTION_ID
    next_k = 4 if draft_armed else 0
    metadata = make_scheduler_metadata(
        engine_step_index=step_index,
        next_k=next_k,
        selection_intent="exploit" if draft_armed else "force_off",
        decode_req_ids=request_ids,
        num_scheduled_tokens=dict.fromkeys(request_ids, next_k + 1),
        scheduled_spec_decode_tokens=(
            {request_id: [1, 2, 3, 4] for request_id in request_ids}
            if draft_armed
            else {}
        ),
        draft_action_by_req=dict.fromkeys(request_ids, live_action_id),
        context_tokens_by_req=dict.fromkeys(request_ids, 1024),
        generated_suffix_by_req=dict.fromkeys(request_ids, 0),
        shared_target_kv_blocks_in_use=1024,
        shared_target_kv_block_capacity=24529,
        preemptions=0,
        recomputed_tokens=0,
        scheduled_at_s=time.monotonic() - 0.001,
    )
    evidence = make_runner_evidence(
        metadata=metadata,
        target_runtime_mode="FULL",
        output=torch.empty((32, next_k), dtype=torch.int32),
        proposal_called=draft_armed,
        draft_step0_query_width=1 if draft_armed else None,
        draft_step0_num_tokens=32 if draft_armed else None,
        draft_step0_batch_size=32 if draft_armed else None,
        draft_step0_runtime_mode="PIECEWISE" if draft_armed else None,
        draft_chain_runtime_mode="PIECEWISE" if draft_armed else None,
        shared_kv=SharedKVIdentity("binding", "pool", 36, 36),
        shared_weights=SharedWeightIdentity(
            LOGICAL_WEIGHT_VERSION,
            LOGICAL_WEIGHT_VERSION,
            100,
            "pointer-binding",
        ),
        true_slot_mapping_id="slots",
    )
    event = build_same_event_record(
        capture_id=recorder.capture_id,
        metadata=metadata,
        evidence=evidence,
        accepted_draft_tokens=dict.fromkeys(request_ids, 0),
        raw_generated_tokens=dict.fromkeys(request_ids, 1),
        committed_tokens=dict.fromkeys(request_ids, 1),
        draft_armed=dict.fromkeys(request_ids, draft_armed),
        invalid_spec_tokens=dict.fromkeys(request_ids, 0),
        elapsed_s=0.001,
    )
    return event, evidence


def _live_event(
    recorder: P4SameEventRecorder,
    request_ids: list[str],
    *,
    step_index: int,
    generated_suffix: int,
    accepted: int,
) -> tuple[dict, KOffRunnerEvidence]:
    metadata = make_scheduler_metadata(
        engine_step_index=step_index,
        next_k=4,
        selection_intent="exploit",
        decode_req_ids=request_ids,
        num_scheduled_tokens=dict.fromkeys(request_ids, 5),
        scheduled_spec_decode_tokens={
            request_id: [1, 2, 3, 4] for request_id in request_ids
        },
        draft_action_by_req=dict.fromkeys(request_ids, K4_ACTION_ID),
        context_tokens_by_req=dict.fromkeys(request_ids, 1024),
        generated_suffix_by_req=dict.fromkeys(request_ids, generated_suffix),
        shared_target_kv_blocks_in_use=1024,
        shared_target_kv_block_capacity=24529,
        preemptions=0,
        recomputed_tokens=0,
        scheduled_at_s=time.monotonic() - 0.001,
    )
    evidence = make_runner_evidence(
        metadata=metadata,
        target_runtime_mode="FULL",
        output=torch.empty((len(request_ids), 4), dtype=torch.int32),
        proposal_called=True,
        draft_step0_query_width=1,
        draft_step0_num_tokens=len(request_ids),
        draft_step0_batch_size=len(request_ids),
        draft_step0_runtime_mode="PIECEWISE",
        draft_chain_runtime_mode="PIECEWISE",
        shared_kv=SharedKVIdentity("binding", "pool", 36, 36),
        shared_weights=SharedWeightIdentity("version", "version", 100),
        true_slot_mapping_id="slots",
    )
    event = build_same_event_record(
        capture_id=recorder.capture_id,
        metadata=metadata,
        evidence=evidence,
        accepted_draft_tokens=dict.fromkeys(request_ids, accepted),
        raw_generated_tokens=dict.fromkeys(request_ids, accepted + 1),
        committed_tokens=dict.fromkeys(request_ids, accepted + 1),
        draft_armed=dict.fromkeys(request_ids, True),
        invalid_spec_tokens=dict.fromkeys(request_ids, 0),
        elapsed_s=0.001,
    )
    return event, evidence


class LiveRecorderTests(unittest.TestCase):
    """Exercise the engine-side builder through the frozen CPU adapter."""

    def test_randomized_ids_emit_frozen_ids_in_scoreable_capture(self) -> None:
        frozen_ids = _prompt_ids()
        request_ids = [
            f"{request_id}-{index:08x}" for index, request_id in enumerate(frozen_ids)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "cell.json"
            output_path = root / "capture.json"
            config_path.write_text(json.dumps(_capture_config(frozen_ids)))
            recorder = P4SameEventRecorder(
                str(config_path),
                str(output_path),
            )

            for offset in range(52):
                accepted = 4 if offset < 51 else 0
                event, evidence = _live_event(
                    recorder,
                    request_ids,
                    step_index=100 + offset,
                    generated_suffix=offset * 5,
                    accepted=accepted,
                )
                recorder.record(event, evidence)

            capture = json.loads(output_path.read_text())
            adapted = adapt_capture(capture)
            self.assertTrue(capture["complete"])
            self.assertEqual(capture["runner"]["shared_kv_binding_id"], "binding")
            self.assertEqual(capture["runner"]["true_slot_mapping_id"], "slots")
            self.assertEqual(adapted["counters"]["E_committed"], 8192)
            self.assertEqual(adapted["source"]["event_count"], 52)
            self.assertTrue(adapted["counters"]["equal_work_holds"])
            for event in capture["events"]:
                self.assertEqual(
                    [row["request_id"] for row in event["request_steps"]],
                    frozen_ids,
                )

    def test_exact_unsuffixed_ids_remain_accepted(self) -> None:
        request_ids = _prompt_ids()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "cell.json"
            output_path = root / "capture.json"
            config_path.write_text(json.dumps(_capture_config(request_ids)))
            recorder = P4SameEventRecorder(str(config_path), str(output_path))
            event, evidence = _live_event(
                recorder,
                request_ids,
                step_index=100,
                generated_suffix=0,
                accepted=0,
            )

            recorder.record(event, evidence)
            recorder.close()

            capture = json.loads(output_path.read_text())
            self.assertEqual(
                [row["request_id"] for row in capture["events"][0]["request_steps"]],
                request_ids,
            )

    def test_invalid_or_colliding_internal_ids_fail_closed(self) -> None:
        frozen_ids = _prompt_ids()
        cases = {
            "malformed suffix": (
                [f"{frozen_ids[0]}-synthetic", *frozen_ids[1:]],
                "exactly eight lowercase hex",
            ),
            "unknown id": (
                ["unknown-prompt-00000000", *frozen_ids[1:]],
                "exactly eight lowercase hex",
            ),
            "canonical collision": (
                [
                    f"{frozen_ids[0]}-00000000",
                    f"{frozen_ids[0]}-00000001",
                    *frozen_ids[2:],
                ],
                "collide after canonicalization",
            ),
        }
        for label, (request_ids, message) in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config_path = root / "cell.json"
                output_path = root / "capture.json"
                config_path.write_text(json.dumps(_capture_config(frozen_ids)))
                recorder = P4SameEventRecorder(str(config_path), str(output_path))
                event, evidence = _live_event(
                    recorder,
                    request_ids,
                    step_index=100,
                    generated_suffix=0,
                    accepted=0,
                )

                with self.assertRaisesRegex(RuntimeError, message):
                    recorder.record(event, evidence)

    def test_all_actions_close_and_rotate_48_decode_only_cells(self) -> None:
        request_ids = _prompt_ids()
        for action_id in (OFF_ACTION_ID, K4_ACTION_ID, W512_ACTION_ID):
            with (
                self.subTest(action_id=action_id),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                config_path = root / "plan.json"
                output_dir = root / "captures"
                output_dir.mkdir()
                config_path.write_text(
                    json.dumps(_action_plan(request_ids, action_id)),
                    encoding="utf-8",
                )
                recorder = P4SameEventRecorder(
                    str(config_path),
                    str(output_dir),
                    boot_action_id=action_id,
                    logical_weight_version=LOGICAL_WEIGHT_VERSION,
                    minimum_shared_kv_blocks=P4_MIN_SHARED_KV_BLOCKS,
                )

                for step_index in range(48):
                    event, evidence = _action_event(
                        recorder,
                        request_ids,
                        action_id=action_id,
                        step_index=step_index,
                    )
                    recorder.record(event, evidence)

                captures = sorted(output_dir.glob("*.json"))
                self.assertEqual(recorder.completed_capture_count, 48)
                self.assertEqual(len(captures), 48)
                for path in captures:
                    capture = json.loads(path.read_text(encoding="utf-8"))
                    self.assertTrue(capture["complete"])
                    self.assertEqual(capture["matrix"]["action_id"], action_id)
                    self.assertEqual(capture["events"][0]["action_id"], action_id)
                    self.assertEqual(
                        capture["events"][0]["source"]["verified_action_id"],
                        action_id,
                    )
                    self.assertEqual(
                        capture["runner"]["draft_weight_version"],
                        LOGICAL_WEIGHT_VERSION,
                    )

    def test_same_boot_plan_rejects_out_of_order_cells_before_capture(self) -> None:
        plan = _w512_plan(_prompt_ids())
        plan["cells"][0], plan["cells"][1] = plan["cells"][1], plan["cells"][0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "plan.json"
            output_dir = root / "captures"
            output_dir.mkdir()
            config_path.write_text(json.dumps(plan), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "canonical order"):
                P4SameEventRecorder(
                    str(config_path),
                    str(output_dir),
                    boot_action_id=W512_ACTION_ID,
                    logical_weight_version=LOGICAL_WEIGHT_VERSION,
                    minimum_shared_kv_blocks=P4_MIN_SHARED_KV_BLOCKS,
                )
            self.assertEqual(list(output_dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
