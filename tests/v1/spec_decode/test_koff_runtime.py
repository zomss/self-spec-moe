# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the Phase 97 minimal-B0 live K4/OFF contract."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm import envs as vllm_envs
from vllm.v1.spec_decode.koff_runtime import (
    K4_ACTION_ID,
    KMAX8_ACTION_ID,
    OFF_ACTION_ID,
    P4_MIN_SHARED_KV_BLOCKS,
    W512_ACTION_ID,
    KOffBootOptions,
    KOffRuntimeError,
    KOffTraceWriter,
    P4SameEventRecorder,
    SharedKVIdentity,
    SharedWeightIdentity,
    abort_mixed_decode_drafts,
    action_for_k,
    build_live_step_record,
    build_same_event_record,
    make_runner_evidence,
    make_scheduler_metadata,
    make_trace_header,
    slot_mapping_identity,
    validate_action_transport,
    validate_boot_config,
    validate_draft_token_batch,
    validate_k_values,
    validate_p4_shared_kv_capacity,
    validate_shared_kv_aliases,
    validate_shared_weight_aliases,
)
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

pytestmark = pytest.mark.cpu_test
LOGICAL_WEIGHT_VERSION = "target-matching-config-sha256-" + "a" * 64


def _options(**overrides) -> KOffBootOptions:
    values = {
        "enabled": True,
        "trace_path": "",
        "p4_capture_config_path": "",
        "p4_capture_output_path": "",
        "p4_boot_action_id": "",
        "p4_logical_weight_version": "",
        "p4_min_kv_blocks": 0,
        "shared_kv": True,
        "shared_kv_step0_decode": True,
        "share_weights": True,
        "draft_kv_dtype": "",
        "draft_kv_window": 0,
        "draft_kv_sinks": 16,
        "draft_skip_layers": "",
        "draft_partial_replica": "",
        "ahead_chain": False,
        "consume_ahead": False,
    }
    values.update(overrides)
    return KOffBootOptions(**values)


def _config(**spec_overrides):
    target = SimpleNamespace(model="target", quantization=None)
    draft = SimpleNamespace(model="target", quantization=None)
    spec_values = {
        "method": "draft_model",
        "num_speculative_tokens": 4,
        "disable_padded_drafter_batch": False,
        "draft_model_config": draft,
    }
    spec_values.update(spec_overrides)
    return SimpleNamespace(
        model_config=target,
        speculative_config=SimpleNamespace(**spec_values),
    )


def _metadata(
    *,
    current: str = K4_ACTION_ID,
    next_k: int = 4,
):
    req_ids = ("r0", "r1")
    width = action_for_k(4 if current == K4_ACTION_ID else 0).k
    query_width = width + 1
    num_scheduled = {req_id: query_width for req_id in req_ids}
    scheduled_spec = {req_id: list(range(width)) for req_id in req_ids if width}
    return make_scheduler_metadata(
        engine_step_index=3,
        next_k=next_k,
        selection_intent="exploit",
        decode_req_ids=req_ids,
        num_scheduled_tokens=num_scheduled,
        scheduled_spec_decode_tokens=scheduled_spec,
        draft_action_by_req={req_id: current for req_id in req_ids},
        context_tokens_by_req={"r0": 100, "r1": 120},
        generated_suffix_by_req={"r0": 8, "r1": 10},
        shared_target_kv_blocks_in_use=20,
        shared_target_kv_block_capacity=100,
        preemptions=0,
        recomputed_tokens=0,
        scheduled_at_s=time.monotonic() - 0.01,
    )


def _identities():
    return (
        SharedKVIdentity("binding", "pool", 2, 2),
        SharedWeightIdentity("version", "version", 3),
    )


def _prefill_arm_metadata():
    return make_scheduler_metadata(
        engine_step_index=0,
        next_k=4,
        selection_intent="exploit",
        decode_req_ids=(),
        num_scheduled_tokens={"r0": 8077, "r1": 83},
        scheduled_spec_decode_tokens={},
        draft_action_by_req={},
        context_tokens_by_req={},
        generated_suffix_by_req={},
        shared_target_kv_blocks_in_use=513,
        shared_target_kv_block_capacity=24527,
        preemptions=0,
        recomputed_tokens=0,
        scheduled_at_s=time.monotonic() - 0.01,
        capture_cohort_arm=True,
    )


def _r8_prefill_arm_metadata():
    scheduled_widths = (
        83,
        84,
        61,
        247,
        68,
        50,
        102,
        47,
        54,
        73,
        262,
        98,
        144,
        286,
        343,
        98,
    )
    request_ids = tuple(f"R8-s0-p{index:03d}" for index in range(16))
    return make_scheduler_metadata(
        engine_step_index=27700,
        next_k=4,
        selection_intent="exploit",
        decode_req_ids=(),
        num_scheduled_tokens=dict(zip(request_ids, scheduled_widths, strict=True)),
        scheduled_spec_decode_tokens={},
        draft_action_by_req={},
        context_tokens_by_req={},
        generated_suffix_by_req={},
        shared_target_kv_blocks_in_use=145,
        shared_target_kv_block_capacity=24527,
        preemptions=0,
        recomputed_tokens=0,
        scheduled_at_s=time.monotonic() - 0.01,
        capture_cohort_arm=True,
    )


def _mixed_abort_metadata():
    decode_req_ids = ("r0", "r1")
    num_scheduled = {"r0": 5, "r1": 5, "prefill": 16}
    scheduled_spec = {
        "r0": [1, 2, 3, 4],
        "r1": [5, 6, 7, 8],
    }
    provenance = {req_id: K4_ACTION_ID for req_id in decode_req_ids}
    abort = abort_mixed_decode_drafts(
        decode_req_ids,
        num_scheduled,
        scheduled_spec,
        provenance,
    )
    metadata = make_scheduler_metadata(
        engine_step_index=4,
        next_k=0,
        selection_intent="force_off",
        decode_req_ids=decode_req_ids,
        num_scheduled_tokens=num_scheduled,
        scheduled_spec_decode_tokens=scheduled_spec,
        draft_action_by_req=provenance,
        context_tokens_by_req={"r0": 100, "r1": 120},
        generated_suffix_by_req={"r0": 8, "r1": 10},
        shared_target_kv_blocks_in_use=20,
        shared_target_kv_block_capacity=100,
        preemptions=0,
        recomputed_tokens=0,
        scheduled_at_s=time.monotonic() - 0.01,
        aborted_action_id=abort.action_id,
        discarded_draft_width=abort.draft_width,
        aborted_draft_request_count=abort.request_count,
    )
    return metadata, num_scheduled, scheduled_spec, provenance


@pytest.mark.parametrize("k,action_id", [(0, OFF_ACTION_ID), (4, K4_ACTION_ID)])
def test_closed_action_registry(k: int, action_id: str):
    assert action_for_k(k).action_id == action_id


def test_closed_action_registry_rejects_other_k():
    with pytest.raises(KOffRuntimeError, match="only K=0 or K=4"):
        action_for_k(2)


def test_boot_contract_accepts_only_target_matching_shared_kv():
    validate_boot_config(
        _config(),
        _options(),
        dynamic_k_values=(0, 4),
        policy=[{"options": [{"K": 4}]}],
    )


@pytest.mark.parametrize(
    "options,match",
    [
        (_options(shared_kv=False), "SHARED_KV=1"),
        (_options(shared_kv_step0_decode=False), "STEP0_DECODE=1"),
        (_options(share_weights=False), "SHARE_WEIGHTS=1"),
        (_options(draft_kv_dtype="fp8"), "no draft-only KV dtype"),
        (_options(draft_kv_window=512), "no draft KV window"),
        (_options(draft_skip_layers="2,4"), "no skipped draft layers"),
        (_options(draft_partial_replica="0,1"), "no partial draft replica"),
        (_options(ahead_chain=True), "no ahead draft chain"),
        (_options(consume_ahead=True), "no consumed ahead draft chain"),
    ],
)
def test_boot_contract_rejects_other_realizations(options, match: str):
    with pytest.raises(KOffRuntimeError, match=match):
        validate_boot_config(_config(), options)


def test_boot_contract_rejects_other_method_or_k():
    with pytest.raises(KOffRuntimeError, match="draft_model"):
        validate_boot_config(_config(method="ngram"), _options())
    with pytest.raises(KOffRuntimeError, match="num_speculative_tokens=4"):
        validate_boot_config(_config(num_speculative_tokens=3), _options())
    with pytest.raises(KOffRuntimeError, match="padded draft-model batch"):
        validate_boot_config(_config(disable_padded_drafter_batch=True), _options())


def test_trace_requires_explicit_runtime_enable():
    with pytest.raises(KOffRuntimeError, match="KOFF_RUNTIME=1"):
        validate_boot_config(
            _config(),
            _options(enabled=False, trace_path="unused.jsonl"),
        )


def test_p4_capture_boot_contract_requires_pair_and_synchronous_runtime():
    with pytest.raises(KOffRuntimeError, match="requires both"):
        validate_boot_config(
            _config(),
            _options(p4_capture_config_path="cell.json"),
        )
    with pytest.raises(KOffRuntimeError, match="KOFF_RUNTIME=1"):
        validate_boot_config(
            _config(),
            _options(
                enabled=False,
                p4_capture_config_path="cell.json",
                p4_capture_output_path="capture.json",
                p4_boot_action_id=K4_ACTION_ID,
                p4_logical_weight_version=LOGICAL_WEIGHT_VERSION,
                p4_min_kv_blocks=P4_MIN_SHARED_KV_BLOCKS,
            ),
        )
    config = _config()
    config.scheduler_config = SimpleNamespace(async_scheduling=True)
    with pytest.raises(KOffRuntimeError, match="synchronous scheduling"):
        validate_boot_config(
            config,
            _options(
                p4_capture_config_path="cell.json",
                p4_capture_output_path="capture.json",
                p4_boot_action_id=K4_ACTION_ID,
                p4_logical_weight_version=LOGICAL_WEIGHT_VERSION,
                p4_min_kv_blocks=P4_MIN_SHARED_KV_BLOCKS,
            ),
        )


def test_p4_capture_boot_contract_accepts_only_trusted_w512_shape():
    capture = {
        "p4_capture_config_path": "plan.json",
        "p4_capture_output_path": "captures",
        "p4_boot_action_id": W512_ACTION_ID,
        "p4_logical_weight_version": LOGICAL_WEIGHT_VERSION,
        "p4_min_kv_blocks": P4_MIN_SHARED_KV_BLOCKS,
        "draft_kv_window": 512,
        "draft_kv_sinks": 16,
    }
    validate_boot_config(_config(), _options(**capture), dynamic_k_values=(4,))

    with pytest.raises(KOffRuntimeError, match="window=512 and sinks=16"):
        validate_boot_config(
            _config(),
            _options(**(capture | {"draft_kv_sinks": 0})),
        )
    with pytest.raises(KOffRuntimeError, match="no draft KV window"):
        validate_boot_config(
            _config(),
            _options(**(capture | {"p4_boot_action_id": K4_ACTION_ID})),
        )


def test_p4_capture_boot_contract_rejects_unbound_conformance_fields():
    with pytest.raises(KOffRuntimeError, match="requires boot action"):
        validate_boot_config(
            _config(),
            _options(
                p4_capture_config_path="cell.json",
                p4_capture_output_path="capture.json",
            ),
        )
    with pytest.raises(KOffRuntimeError, match="require an explicit capture"):
        validate_boot_config(
            _config(),
            _options(
                p4_boot_action_id=K4_ACTION_ID,
                p4_logical_weight_version=LOGICAL_WEIGHT_VERSION,
                p4_min_kv_blocks=P4_MIN_SHARED_KV_BLOCKS,
            ),
        )
    with pytest.raises(KOffRuntimeError, match="logical weight version"):
        validate_boot_config(
            _config(),
            _options(
                p4_capture_config_path="cell.json",
                p4_capture_output_path="capture.json",
                p4_boot_action_id=K4_ACTION_ID,
                p4_logical_weight_version="pointer-derived",
                p4_min_kv_blocks=P4_MIN_SHARED_KV_BLOCKS,
            ),
        )


def test_p4_shared_kv_floor_is_exact_and_fail_closed():
    validate_p4_shared_kv_capacity(
        P4_MIN_SHARED_KV_BLOCKS,
        P4_MIN_SHARED_KV_BLOCKS,
    )
    with pytest.raises(KOffRuntimeError, match="below the 21682-block floor"):
        validate_p4_shared_kv_capacity(
            P4_MIN_SHARED_KV_BLOCKS - 1,
            P4_MIN_SHARED_KV_BLOCKS,
        )
    with pytest.raises(KOffRuntimeError, match="registered 21682-block floor"):
        validate_p4_shared_kv_capacity(30000, 21000)


def test_boot_contract_rejects_policy_with_k2():
    with pytest.raises(KOffRuntimeError, match="outside minimal-B0"):
        validate_boot_config(
            _config(),
            _options(),
            policy=[{"options": [{"K": 2}]}],
        )


def test_scheduler_metadata_preserves_current_and_next_actions():
    metadata = _metadata(current=OFF_ACTION_ID, next_k=4)
    assert metadata.verified_action_id == OFF_ACTION_ID
    assert metadata.next_action_id == K4_ACTION_ID
    assert metadata.pure_decode
    assert metadata.total_scheduled_kv_tokens == 220


def test_mixed_boundary_aborts_complete_k4_rows_to_q1_off():
    metadata, num_scheduled, scheduled_spec, provenance = _mixed_abort_metadata()

    assert num_scheduled == {"r0": 1, "r1": 1, "prefill": 16}
    assert scheduled_spec == {}
    assert provenance == {"r0": OFF_ACTION_ID, "r1": OFF_ACTION_ID}
    assert metadata.verified_action_id == OFF_ACTION_ID
    assert metadata.next_action_id == OFF_ACTION_ID
    assert metadata.aborted_action_id == K4_ACTION_ID
    assert metadata.discarded_draft_width == 4
    assert metadata.aborted_draft_request_count == 2
    assert not metadata.pure_decode


def test_mixed_boundary_abort_is_all_or_fail_closed():
    with pytest.raises(KOffRuntimeError, match="partially abort"):
        abort_mixed_decode_drafts(
            ("r0", "r1"),
            {"r0": 5, "r1": 1, "prefill": 16},
            {"r0": [1, 2, 3, 4]},
            {"r0": K4_ACTION_ID, "r1": OFF_ACTION_ID},
        )
    with pytest.raises(KOffRuntimeError, match="complete K4 dispatch"):
        abort_mixed_decode_drafts(
            ("r0",),
            {"r0": 3, "prefill": 16},
            {"r0": [1, 2]},
            {"r0": K4_ACTION_ID},
        )


def test_scheduler_metadata_rejects_mixed_action_or_bad_provenance():
    with pytest.raises(KOffRuntimeError, match="whole decode dispatch"):
        make_scheduler_metadata(
            engine_step_index=0,
            next_k=4,
            selection_intent="exploit",
            decode_req_ids=("r0", "r1"),
            num_scheduled_tokens={"r0": 5, "r1": 1},
            scheduled_spec_decode_tokens={"r0": [1, 2, 3, 4]},
            draft_action_by_req={"r0": K4_ACTION_ID, "r1": OFF_ACTION_ID},
            context_tokens_by_req={"r0": 1, "r1": 1},
            generated_suffix_by_req={"r0": 1, "r1": 1},
            shared_target_kv_blocks_in_use=1,
            shared_target_kv_block_capacity=2,
            preemptions=0,
            recomputed_tokens=0,
            scheduled_at_s=time.monotonic(),
        )
    with pytest.raises(KOffRuntimeError, match="draft provenance"):
        make_scheduler_metadata(
            engine_step_index=0,
            next_k=4,
            selection_intent="exploit",
            decode_req_ids=("r0",),
            num_scheduled_tokens={"r0": 5},
            scheduled_spec_decode_tokens={"r0": [1, 2, 3, 4]},
            draft_action_by_req={"r0": OFF_ACTION_ID},
            context_tokens_by_req={"r0": 1},
            generated_suffix_by_req={"r0": 1},
            shared_target_kv_blocks_in_use=1,
            shared_target_kv_block_capacity=2,
            preemptions=0,
            recomputed_tokens=0,
            scheduled_at_s=time.monotonic(),
        )


def test_worker_revalidates_action_transport():
    metadata = _metadata()
    validate_action_transport(
        metadata,
        next_k=4,
        num_scheduled_tokens={"r0": 5, "r1": 5},
        scheduled_spec_decode_tokens={
            "r0": [1, 2, 3, 4],
            "r1": [1, 2, 3, 4],
        },
    )
    with pytest.raises(KOffRuntimeError, match="next action"):
        validate_action_transport(
            metadata,
            next_k=0,
            num_scheduled_tokens={"r0": 5, "r1": 5},
            scheduled_spec_decode_tokens={
                "r0": [1, 2, 3, 4],
                "r1": [1, 2, 3, 4],
            },
        )


def test_worker_revalidates_mixed_boundary_abort():
    metadata, num_scheduled, scheduled_spec, _ = _mixed_abort_metadata()
    validate_action_transport(
        metadata,
        next_k=0,
        num_scheduled_tokens=num_scheduled,
        scheduled_spec_decode_tokens=scheduled_spec,
    )

    with pytest.raises(KOffRuntimeError, match="force next OFF"):
        validate_action_transport(
            replace(metadata, next_action_id=K4_ACTION_ID),
            next_k=4,
            num_scheduled_tokens=num_scheduled,
            scheduled_spec_decode_tokens=scheduled_spec,
        )
    with pytest.raises(KOffRuntimeError, match="expected 0"):
        validate_action_transport(
            metadata,
            next_k=0,
            num_scheduled_tokens={**num_scheduled, "r0": 5},
            scheduled_spec_decode_tokens={"r0": [1, 2, 3, 4]},
        )


def test_draft_token_batch_carries_action_provenance():
    validate_draft_token_batch(K4_ACTION_ID, [[1, 2, 3, 4], [5, 6, 7, 8]])
    validate_draft_token_batch(OFF_ACTION_ID, [[], []])
    with pytest.raises(KOffRuntimeError, match="missing their action tag"):
        validate_draft_token_batch(None, [[], []])
    with pytest.raises(KOffRuntimeError, match="row widths"):
        validate_draft_token_batch(K4_ACTION_ID, [[1, 2]])


def test_live_tensor_and_weight_alias_checks():
    target_cache = torch.empty(8)
    pool = object()
    kv_identity = validate_shared_kv_aliases(
        {"draft.layer": "target.layer"},
        {"draft.layer": target_cache, "target.layer": target_cache},
        pool,
    )
    assert kv_identity.layer_count == 1
    other_cache = torch.empty(8)
    other_identity = validate_shared_kv_aliases(
        {"draft.layer": "target.layer"},
        {"draft.layer": other_cache, "target.layer": other_cache},
        pool,
    )
    assert kv_identity.binding_id != other_identity.binding_id
    assert kv_identity.pool_id != other_identity.pool_id

    target = torch.nn.Linear(4, 3)
    draft = torch.nn.Linear(4, 3)
    draft._parameters["weight"] = target.weight
    draft._parameters["bias"] = target.bias
    weight_identity = validate_shared_weight_aliases(target, draft)
    assert weight_identity.target_version_id == weight_identity.draft_version_id
    assert weight_identity.parameter_alias_count == 2


def test_logical_weight_version_is_stable_but_pointer_binding_is_not():
    identities = []
    models = []
    for _ in range(2):
        target = torch.nn.Linear(4, 3)
        draft = torch.nn.Linear(4, 3)
        draft._parameters["weight"] = target.weight
        draft._parameters["bias"] = target.bias
        models.append((target, draft))
        identities.append(
            validate_shared_weight_aliases(
                target,
                draft,
                LOGICAL_WEIGHT_VERSION,
            )
        )
    assert {identity.target_version_id for identity in identities} == {
        LOGICAL_WEIGHT_VERSION
    }
    assert identities[0].alias_binding_id != identities[1].alias_binding_id


def test_live_alias_checks_reject_copies():
    with pytest.raises(KOffRuntimeError, match="is not target tensor"):
        validate_shared_kv_aliases(
            {"draft.layer": "target.layer"},
            {
                "draft.layer": torch.empty(8),
                "target.layer": torch.empty(8),
            },
            object(),
        )
    with pytest.raises(KOffRuntimeError, match="does not alias"):
        validate_shared_weight_aliases(torch.nn.Linear(4, 3), torch.nn.Linear(4, 3))

    target = torch.nn.Linear(4, 3)
    draft = torch.nn.Linear(4, 3, bias=False)
    draft._parameters["weight"] = target.weight
    with pytest.raises(KOffRuntimeError, match="parameter sets differ"):
        validate_shared_weight_aliases(target, draft)


def test_slot_mapping_id_is_stable_across_views():
    slots = torch.arange(16)
    assert slot_mapping_identity({"layer": slots[:4]}) == slot_mapping_identity(
        [{"layer": slots[4:8]}, {"layer": slots[:12]}]
    )
    assert slot_mapping_identity({"layer": slots}) != slot_mapping_identity(
        {"layer": slots.clone()}
    )


def test_shared_kv_step0_compacts_off_to_k_bootstrap():
    proposer = object.__new__(SpecDecodeBaseProposer)
    proposer._shared_kv_step0_decode = True
    proposer.num_speculative_tokens = 4
    proposer.net_num_new_slots_per_request = 1
    proposer.uses_mrope = False
    proposer.uses_xdrope_dim = 0
    proposer.supports_mm_inputs = False
    proposer.pass_hidden_states_to_model = False

    bootstrap = SimpleNamespace(max_query_len=2, num_actual_tokens=6, num_reqs=3)
    wider = SimpleNamespace(max_query_len=3, num_actual_tokens=9, num_reqs=3)
    assert proposer._step0_decode_active(bootstrap, None)
    assert not proposer._step0_decode_active(wider, None)


def test_shared_kv_bootstrap_compaction_needs_no_rejection_tensor():
    class _CommonAttentionMetadata(SimpleNamespace):
        def replace(self, **updates):
            return _CommonAttentionMetadata(**(vars(self) | updates))

    proposer = object.__new__(SpecDecodeBaseProposer)
    proposer.input_ids = torch.tensor([10, 11, 20, 21])
    proposer.positions = torch.tensor([0, 1, 4, 5])
    proposer.arange = torch.arange(8, dtype=torch.int32)
    proposer.token_arange_np = proposer.arange.numpy()
    metadata = _CommonAttentionMetadata(
        num_reqs=2,
        seq_lens=torch.tensor([2, 6], dtype=torch.int32),
        slot_mapping=torch.tensor([100, 101, 200, 201]),
    )

    compact, num_tokens, indices, appended_slots = proposer._compact_step0_decode(
        metadata,
        metadata,
        torch.tensor([1, 3]),
        None,
    )
    assert num_tokens == 2
    assert indices.tolist() == [0, 1]
    assert appended_slots.tolist() == [101, 201]
    assert compact.seq_lens.tolist() == [2, 6]
    assert proposer.input_ids[:2].tolist() == [11, 21]


@pytest.mark.parametrize(
    "prior_query_width",
    [None, 1],
    ids=["isolated-r8", "r5cot-to-r8"],
)
def test_step0_work_evidence_preserves_variable_r8_shape(prior_query_width):
    proposer = object.__new__(SpecDecodeBaseProposer)
    proposer._last_step0_query_width = prior_query_width

    proposer._record_step0_work_evidence(
        num_tokens=2116,
        batch_size=16,
        max_query_len=344,
        step0_decode=False,
    )

    assert proposer._last_step0_query_width is None
    assert proposer._last_step0_num_tokens == 2116
    assert proposer._last_step0_batch_size == 16


def test_runner_evidence_accepts_r8_variable_prefill_work():
    shared_kv, shared_weights = _identities()
    evidence = make_runner_evidence(
        metadata=_r8_prefill_arm_metadata(),
        target_runtime_mode="NONE",
        output=torch.empty((16, 4), dtype=torch.int32),
        proposal_called=True,
        draft_step0_query_width=None,
        draft_step0_num_tokens=2116,
        draft_step0_batch_size=16,
        draft_step0_runtime_mode="NONE",
        draft_chain_runtime_mode="PIECEWISE",
        shared_kv=shared_kv,
        shared_weights=shared_weights,
        true_slot_mapping_id="slots",
    )

    assert evidence.draft_step0_query_width is None
    assert evidence.draft_step0_num_tokens == 2116
    assert evidence.draft_step0_batch_size == 16
    assert evidence.produced_draft_width == 4


def test_runner_evidence_rejects_r8_output_batch_mismatch():
    shared_kv, shared_weights = _identities()
    with pytest.raises(KOffRuntimeError, match="produced 15 draft rows"):
        make_runner_evidence(
            metadata=_r8_prefill_arm_metadata(),
            target_runtime_mode="NONE",
            output=torch.empty((15, 4), dtype=torch.int32),
            proposal_called=True,
            draft_step0_query_width=None,
            draft_step0_num_tokens=2116,
            draft_step0_batch_size=16,
            draft_step0_runtime_mode="NONE",
            draft_chain_runtime_mode="PIECEWISE",
            shared_kv=shared_kv,
            shared_weights=shared_weights,
            true_slot_mapping_id="slots",
        )


def test_runner_evidence_rejects_inexact_uniform_width_fact():
    shared_kv, shared_weights = _identities()
    with pytest.raises(KOffRuntimeError, match="uniform query width"):
        make_runner_evidence(
            metadata=_r8_prefill_arm_metadata(),
            target_runtime_mode="NONE",
            output=torch.empty((16, 4), dtype=torch.int32),
            proposal_called=True,
            draft_step0_query_width=132,
            draft_step0_num_tokens=2116,
            draft_step0_batch_size=16,
            draft_step0_runtime_mode="NONE",
            draft_chain_runtime_mode="PIECEWISE",
            shared_kv=shared_kv,
            shared_weights=shared_weights,
            true_slot_mapping_id="slots",
        )


def test_runner_evidence_rejects_variable_width_for_pure_decode():
    shared_kv, shared_weights = _identities()
    with pytest.raises(KOffRuntimeError, match="query width None, expected 1"):
        make_runner_evidence(
            metadata=_metadata(),
            target_runtime_mode="FULL",
            output=torch.empty((2, 4), dtype=torch.int32),
            proposal_called=True,
            draft_step0_query_width=None,
            draft_step0_num_tokens=2,
            draft_step0_batch_size=2,
            draft_step0_runtime_mode="PIECEWISE",
            draft_chain_runtime_mode="PIECEWISE",
            shared_kv=shared_kv,
            shared_weights=shared_weights,
            true_slot_mapping_id="slots",
        )


def test_runner_evidence_rejects_unarmed_variable_prefill():
    shared_kv, shared_weights = _identities()
    metadata = replace(_r8_prefill_arm_metadata(), capture_cohort_arm=False)
    with pytest.raises(KOffRuntimeError, match="pure-prefill cohort arming"):
        make_runner_evidence(
            metadata=metadata,
            target_runtime_mode="NONE",
            output=torch.empty((16, 4), dtype=torch.int32),
            proposal_called=True,
            draft_step0_query_width=None,
            draft_step0_num_tokens=2116,
            draft_step0_batch_size=16,
            draft_step0_runtime_mode="NONE",
            draft_chain_runtime_mode="PIECEWISE",
            shared_kv=shared_kv,
            shared_weights=shared_weights,
            true_slot_mapping_id="slots",
        )


def test_runner_evidence_and_passive_counters_close():
    metadata = _metadata()
    shared_kv, shared_weights = _identities()
    evidence = make_runner_evidence(
        metadata=metadata,
        target_runtime_mode="FULL",
        output=torch.empty((2, 4), dtype=torch.int32),
        proposal_called=True,
        draft_step0_query_width=1,
        draft_step0_num_tokens=2,
        draft_step0_batch_size=2,
        draft_step0_runtime_mode="PIECEWISE",
        draft_chain_runtime_mode="PIECEWISE",
        shared_kv=shared_kv,
        shared_weights=shared_weights,
        true_slot_mapping_id="slots",
    )
    record = build_live_step_record(
        metadata=metadata,
        evidence=evidence,
        raw_generated_lengths={"r0": 3, "r1": 2},
        committed_lengths={"r0": 3, "r1": 2},
        invalid_spec_tokens=0,
        elapsed_s=0.02,
    )
    assert record["eligible_for_p3_replay"]
    assert record["counters"]["H_target_steps"] == 2
    assert record["counters"]["A_accepted"] == 3
    assert record["counters"]["E_committed"] == 5
    assert record["counters"]["closure_holds"]


def test_runner_evidence_accepts_wide_step0_only_for_prefill_arm():
    metadata = _prefill_arm_metadata()
    shared_kv, shared_weights = _identities()

    evidence = make_runner_evidence(
        metadata=metadata,
        target_runtime_mode="PIECEWISE",
        output=torch.empty((2, 4), dtype=torch.int32),
        proposal_called=True,
        draft_step0_query_width=4081,
        draft_step0_num_tokens=8162,
        draft_step0_batch_size=2,
        draft_step0_runtime_mode="PIECEWISE",
        draft_chain_runtime_mode="PIECEWISE",
        shared_kv=shared_kv,
        shared_weights=shared_weights,
        true_slot_mapping_id="slots",
    )

    assert not metadata.pure_decode
    assert metadata.capture_cohort_arm
    assert evidence.draft_step0_query_width == 4081
    assert evidence.produced_draft_width == 4
    assert evidence.draft_dispatched


def test_runner_evidence_rejects_wide_step0_for_pure_decode():
    shared_kv, shared_weights = _identities()

    with pytest.raises(KOffRuntimeError, match="query width 4081, expected 1"):
        make_runner_evidence(
            metadata=_metadata(),
            target_runtime_mode="FULL",
            output=torch.empty((2, 4), dtype=torch.int32),
            proposal_called=True,
            draft_step0_query_width=4081,
            draft_step0_num_tokens=8162,
            draft_step0_batch_size=2,
            draft_step0_runtime_mode="PIECEWISE",
            draft_chain_runtime_mode="PIECEWISE",
            shared_kv=shared_kv,
            shared_weights=shared_weights,
            true_slot_mapping_id="slots",
        )


@pytest.mark.parametrize(
    ("num_tokens", "batch_size"),
    [(None, 2), (8162, None), (0, 2), (8162, 0)],
)
def test_runner_evidence_rejects_missing_prefill_work(num_tokens, batch_size):
    shared_kv, shared_weights = _identities()

    with pytest.raises(KOffRuntimeError, match="positive draft step-0 work"):
        make_runner_evidence(
            metadata=_prefill_arm_metadata(),
            target_runtime_mode="PIECEWISE",
            output=torch.empty((2, 4), dtype=torch.int32),
            proposal_called=True,
            draft_step0_query_width=None,
            draft_step0_num_tokens=num_tokens,
            draft_step0_batch_size=batch_size,
            draft_step0_runtime_mode="PIECEWISE",
            draft_chain_runtime_mode="PIECEWISE",
            shared_kv=shared_kv,
            shared_weights=shared_weights,
            true_slot_mapping_id="slots",
        )


def test_same_event_builder_requires_explicit_per_request_rows():
    metadata = _metadata()
    shared_kv, shared_weights = _identities()
    evidence = make_runner_evidence(
        metadata=metadata,
        target_runtime_mode="FULL",
        output=torch.empty((2, 4), dtype=torch.int32),
        proposal_called=True,
        draft_step0_query_width=1,
        draft_step0_num_tokens=2,
        draft_step0_batch_size=2,
        draft_step0_runtime_mode="PIECEWISE",
        draft_chain_runtime_mode="PIECEWISE",
        shared_kv=shared_kv,
        shared_weights=shared_weights,
        true_slot_mapping_id="slots",
    )
    event = build_same_event_record(
        capture_id="capture-b0-k4",
        metadata=metadata,
        evidence=evidence,
        accepted_draft_tokens={"r0": 2, "r1": 1},
        raw_generated_tokens={"r0": 3, "r1": 2},
        committed_tokens={"r0": 3, "r1": 2},
        draft_armed={"r0": True, "r1": True},
        invalid_spec_tokens={"r0": 0, "r1": 0},
        elapsed_s=0.02,
    )
    assert event["source"]["scheduler_event_id"] == event["event_id"]
    assert event["request_steps"][0] == {
        "request_id": "r0",
        "target_processed": True,
        "draft_armed": True,
        "accepted_draft_tokens": 2,
        "raw_generated_tokens": 3,
        "committed_tokens": 3,
        "clipped_tokens": 0,
    }
    assert event["counters"]["A_accepted"] == 3
    assert event["timing"]["request_decode_time_s"] == pytest.approx(0.04)

    with pytest.raises(KOffRuntimeError, match="exactly cover"):
        build_same_event_record(
            capture_id="capture-b0-k4",
            metadata=metadata,
            evidence=evidence,
            accepted_draft_tokens={"r0": 2, "r1": 1},
            raw_generated_tokens={"r0": 3, "r1": 2},
            committed_tokens={"r0": 3},
            draft_armed={"r0": True, "r1": True},
            invalid_spec_tokens={"r0": 0, "r1": 0},
            elapsed_s=0.02,
        )


def test_p4_recorder_emits_create_new_complete_capture(tmp_path):
    request_ids = tuple(f"r{index:02d}" for index in range(32))
    num_scheduled = dict.fromkeys(request_ids, 1)
    metadata = make_scheduler_metadata(
        engine_step_index=7,
        next_k=0,
        selection_intent="force_off",
        decode_req_ids=request_ids,
        num_scheduled_tokens=num_scheduled,
        scheduled_spec_decode_tokens={},
        draft_action_by_req=dict.fromkeys(request_ids, OFF_ACTION_ID),
        context_tokens_by_req=dict.fromkeys(request_ids, 100),
        generated_suffix_by_req=dict.fromkeys(request_ids, 0),
        shared_target_kv_blocks_in_use=20,
        shared_target_kv_block_capacity=100,
        preemptions=0,
        recomputed_tokens=0,
        scheduled_at_s=time.monotonic() - 0.01,
    )
    shared_kv, shared_weights = _identities()
    evidence = make_runner_evidence(
        metadata=metadata,
        target_runtime_mode="FULL",
        output=torch.empty((32, 0), dtype=torch.int32),
        proposal_called=False,
        draft_step0_query_width=None,
        draft_step0_num_tokens=None,
        draft_step0_batch_size=None,
        draft_step0_runtime_mode=None,
        draft_chain_runtime_mode=None,
        shared_kv=shared_kv,
        shared_weights=shared_weights,
        true_slot_mapping_id="slots",
    )
    event = build_same_event_record(
        capture_id="capture-b0-off",
        metadata=metadata,
        evidence=evidence,
        accepted_draft_tokens=dict.fromkeys(request_ids, 0),
        raw_generated_tokens=dict.fromkeys(request_ids, 1),
        committed_tokens=dict.fromkeys(request_ids, 1),
        draft_armed=dict.fromkeys(request_ids, False),
        invalid_spec_tokens=dict.fromkeys(request_ids, 0),
        elapsed_s=0.01,
    )
    config = {
        "schema_version": 1,
        "capture_contract_id": "p4-b0-same-event-capture-v1",
        "capture_id": "capture-b0-off",
        "scored": False,
        "complete": False,
        "warmup_complete": True,
        "runner": {
            "preregistration_id": "p4-b0-off-k4-w512-value-screen-v1",
            "prompt_manifest_id": "p4-b0-six-regime-prompts-v1",
            "prompt_manifest_sha256": "1" * 64,
            "prompt_bundle_sha256": "2" * 64,
            "target_checkpoint_revision": "revision",
            "target_quantization": None,
            "target_kv_dtype": "bfloat16",
            "draft_weight_version": "pending",
            "shared_kv_binding_id": "pending",
            "shared_kv_alias_proven": True,
            "true_slot_mapping_id": "pending",
            "true_slot_identity_proven": True,
            "hardware_id": "test-gpu",
            "parallel_layout": "tp1-pp1",
            "kernel_backend": "test",
            "graph_grade": "test",
            "warmup_policy": "test",
            "measurement_currency": "S_dec",
        },
        "matrix": {
            "boot_block_id": 1,
            "boot_id": "boot-test",
            "action_id": OFF_ACTION_ID,
            "action_position": 1,
            "action_realization": "live-b0-forced-off",
            "regime_id": "R6",
            "content_seed": 0,
            "round_index": 1,
        },
        "generation": {
            "prompt_record_ids": list(request_ids),
            "generation_seed": 0,
            "batch": 32,
            "max_output_tokens": 1,
            "temperature": 0.0,
            "requested_output_tokens": 32,
            "ignore_eos": True,
        },
        "events": [],
    }
    config_path = tmp_path / "cell.json"
    output_path = tmp_path / "capture.json"
    config_path.write_text(json.dumps(config))
    recorder = P4SameEventRecorder(str(config_path), str(output_path))
    recorder.record(event, evidence)

    capture = json.loads(output_path.read_text())
    assert capture["complete"]
    assert capture["runner"]["shared_kv_binding_id"] == "binding"
    assert capture["runner"]["true_slot_mapping_id"] == "slots"
    assert capture["runner"]["draft_weight_version"] == "version"
    assert capture["events"] == [event]


def test_aborted_mixed_batch_is_unscored_and_does_not_count_drafts():
    metadata, _, _, _ = _mixed_abort_metadata()
    shared_kv, shared_weights = _identities()
    evidence = make_runner_evidence(
        metadata=metadata,
        target_runtime_mode="PIECEWISE",
        output=torch.empty((3, 0), dtype=torch.int32),
        proposal_called=False,
        draft_step0_query_width=None,
        draft_step0_num_tokens=None,
        draft_step0_batch_size=None,
        draft_step0_runtime_mode=None,
        draft_chain_runtime_mode=None,
        shared_kv=shared_kv,
        shared_weights=shared_weights,
        true_slot_mapping_id="slots",
    )
    record = build_live_step_record(
        metadata=metadata,
        evidence=evidence,
        raw_generated_lengths={"r0": 1, "r1": 1},
        committed_lengths={"r0": 1, "r1": 1},
        invalid_spec_tokens=0,
        elapsed_s=0.02,
    )
    assert not record["eligible_for_p3_replay"]
    assert "prefill_or_mixed_batch" in record["exclusion_reasons"]
    assert "aborted_mixed_boundary_draft" in record["exclusion_reasons"]
    assert record["aborted_action_id"] == K4_ACTION_ID
    assert record["discarded_draft_width"] == 4
    assert record["aborted_draft_request_count"] == 2
    assert record["counters"]["D_armed"] == 0
    assert record["counters"]["A_accepted"] == 0
    assert record["counters"]["closure_holds"]


def test_trace_writer_is_create_new_and_append_only(tmp_path):
    path = tmp_path / "live.jsonl"
    header = make_trace_header({"model": "target"})
    writer = KOffTraceWriter(str(path), header)
    writer.write({"schema_version": 1, "record_type": "koff_engine_step"})
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["record_type"] for record in records] == [
        "koff_runtime_header",
        "koff_engine_step",
    ]
    with pytest.raises(KOffRuntimeError, match="refusing to overwrite"):
        KOffTraceWriter(str(path), header)


# --- w98-d2 acceptance scope (G98-D) ---------------------------------------
#
# The scope adds K=8 unconditional arming and per-request acceptance rows for
# the D2 campaign. Rounds 1-2 were measured under minimal-b0/w98-lattice, so
# the first requirement of these tests is that those scopes did not move.


def _d2_scope(monkeypatch):
    monkeypatch.setattr(vllm_envs, "VLLM_SELF_SPEC_BOOT_SCOPE", "w98-d2")


def test_kmax8_is_unresolvable_outside_the_d2_scope():
    """K=8 must not exist for minimal-b0 or w98-lattice."""
    with pytest.raises(KOffRuntimeError):
        action_for_k(8)
    with pytest.raises(KOffRuntimeError):
        validate_k_values([0, 8], "schedule")


def test_kmax8_resolves_only_under_the_d2_scope(monkeypatch):
    _d2_scope(monkeypatch)
    assert action_for_k(8).action_id == KMAX8_ACTION_ID
    assert action_for_k(8).target_query_width == 9
    validate_k_values([8], "schedule")
    # The closed values still resolve; the scope extends, never replaces.
    assert action_for_k(0).action_id == OFF_ACTION_ID
    assert action_for_k(4).action_id == K4_ACTION_ID
    with pytest.raises(KOffRuntimeError):
        action_for_k(2)


def test_d2_requires_an_unconditionally_armed_schedule(monkeypatch):
    _d2_scope(monkeypatch)
    metadata = _metadata()
    shared_kv, shared_weights = _identities()
    evidence = make_runner_evidence(
        metadata=metadata,
        target_runtime_mode="FULL",
        output=torch.empty((2, 4), dtype=torch.int32),
        proposal_called=True,
        draft_step0_query_width=1,
        draft_step0_num_tokens=2,
        draft_step0_batch_size=2,
        draft_step0_runtime_mode="PIECEWISE",
        draft_chain_runtime_mode="PIECEWISE",
        shared_kv=shared_kv,
        shared_weights=shared_weights,
        true_slot_mapping_id="slots",
    )
    record = build_live_step_record(
        metadata=metadata,
        evidence=evidence,
        raw_generated_lengths={"r0": 3, "r1": 2},
        committed_lengths={"r0": 3, "r1": 2},
        invalid_spec_tokens=0,
        elapsed_s=0.02,
        boot_scope="w98-d2",
    )
    rows = record["acceptance_rows"]
    assert [row["request_id"] for row in rows] == ["r0", "r1"]
    # Acceptance is a prefix: raw generated minus the target's own token.
    assert [row["accepted_draft_tokens"] for row in rows] == [2, 1]
    # Each request carries its OWN suffix length, not the batch min/max.
    assert [row["generated_suffix_len"] for row in rows] == [8, 10]
    assert record["engine"]["generated_suffix_min"] == 8
    assert record["engine"]["generated_suffix_max"] == 10


def test_other_scopes_emit_no_acceptance_rows():
    metadata = _metadata()
    shared_kv, shared_weights = _identities()
    evidence = make_runner_evidence(
        metadata=metadata,
        target_runtime_mode="FULL",
        output=torch.empty((2, 4), dtype=torch.int32),
        proposal_called=True,
        draft_step0_query_width=1,
        draft_step0_num_tokens=2,
        draft_step0_batch_size=2,
        draft_step0_runtime_mode="PIECEWISE",
        draft_chain_runtime_mode="PIECEWISE",
        shared_kv=shared_kv,
        shared_weights=shared_weights,
        true_slot_mapping_id="slots",
    )
    for scope in ("minimal-b0", "w98-lattice"):
        record = build_live_step_record(
            metadata=metadata,
            evidence=evidence,
            raw_generated_lengths={"r0": 3, "r1": 2},
            committed_lengths={"r0": 3, "r1": 2},
            invalid_spec_tokens=0,
            elapsed_s=0.02,
            boot_scope=scope,
        )
        assert "acceptance_rows" not in record
