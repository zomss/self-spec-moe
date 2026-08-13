# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the w98-lattice boot scope.

The scope exists because Phase 97's minimal-B0 contract forbids Phase 98's
lattice outright: it hard-requires SHARE_WEIGHTS=1, window 0, and no skipped
layers. These tests pin both halves — that minimal-b0 is unchanged, and that
w98-lattice relaxes exactly three axes and nothing else.
"""

from __future__ import annotations

import dataclasses
import sys
import unittest
from pathlib import Path

from vllm.v1.spec_decode.koff_runtime import (
    BOOT_SCOPE_MINIMAL_B0,
    BOOT_SCOPE_W98_LATTICE,
    BOOT_SCOPES,
    W98_SKIP_COUNTS,
    W98_WINDOWS,
    KOffBootOptions,
    KOffRuntimeError,
    validate_boot_config,
)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


class _ModelConfig:
    def __init__(self, model: str = "target", quantization: str | None = None):
        self.model = model
        self.quantization = quantization


class _SpecConfig:
    method = "draft_model"
    num_speculative_tokens = 4
    disable_padded_drafter_batch = False

    def __init__(self, draft: _ModelConfig | None = None) -> None:
        self.draft_model_config = draft or _ModelConfig()


class _VllmConfig:
    scheduler_config = None

    def __init__(self, draft: _ModelConfig | None = None) -> None:
        self.speculative_config = _SpecConfig(draft)
        self.model_config = _ModelConfig()


def _options(**overrides) -> KOffBootOptions:
    base = {
        "enabled": True,
        "boot_scope": BOOT_SCOPE_W98_LATTICE,
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
        "draft_kv_sinks": 0,
        "draft_skip_layers": "",
        "draft_partial_replica": "",
        "ahead_chain": False,
        "consume_ahead": False,
    }
    base.update(overrides)
    return KOffBootOptions(**base)


def _validate(draft: _ModelConfig | None = None, **overrides) -> None:
    validate_boot_config(_VllmConfig(draft), _options(**overrides))


class ScopeRegistryTests(unittest.TestCase):
    """Scopes are closed and default to the Phase 97 behaviour."""

    def test_two_registered_scopes(self) -> None:
        self.assertEqual(BOOT_SCOPES, {BOOT_SCOPE_MINIMAL_B0, BOOT_SCOPE_W98_LATTICE})

    def test_unknown_scope_fails_closed(self) -> None:
        with self.assertRaises(KOffRuntimeError) as ctx:
            _validate(boot_scope="anything-else")
        self.assertIn("unknown boot scope", str(ctx.exception))

    def test_options_record_carries_the_scope(self) -> None:
        names = {f.name for f in dataclasses.fields(KOffBootOptions)}
        self.assertIn("boot_scope", names)


class MinimalB0UnchangedTests(unittest.TestCase):
    """The default scope must behave exactly as before."""

    def test_phase_97_configuration_still_passes(self) -> None:
        _validate(boot_scope=BOOT_SCOPE_MINIMAL_B0)

    def test_still_requires_weight_sharing(self) -> None:
        with self.assertRaises(KOffRuntimeError) as ctx:
            _validate(boot_scope=BOOT_SCOPE_MINIMAL_B0, share_weights=False)
        self.assertIn("SHARE_WEIGHTS=1", str(ctx.exception))

    def test_still_forbids_a_window(self) -> None:
        with self.assertRaises(KOffRuntimeError) as ctx:
            _validate(
                boot_scope=BOOT_SCOPE_MINIMAL_B0,
                draft_kv_window=512,
                draft_kv_sinks=16,
            )
        self.assertIn("no draft KV window", str(ctx.exception))

    def test_still_forbids_skipped_layers(self) -> None:
        with self.assertRaises(KOffRuntimeError) as ctx:
            _validate(boot_scope=BOOT_SCOPE_MINIMAL_B0, draft_skip_layers="2,4")
        self.assertIn("no skipped draft layers", str(ctx.exception))


class W98RelaxationTests(unittest.TestCase):
    """w98-lattice admits exactly the three lattice axes."""

    def test_quantized_draft_may_disable_weight_sharing(self) -> None:
        _validate(
            draft=_ModelConfig("quantized-ckpt", "compressed-tensors"),
            share_weights=False,
        )

    def test_a_bare_differing_checkpoint_is_still_refused(self) -> None:
        """Relaxing the quant axis must not admit any unrelated draft."""
        with self.assertRaises(KOffRuntimeError) as ctx:
            _validate(draft=_ModelConfig("some-other-model"), share_weights=False)
        self.assertIn("declares no", str(ctx.exception))

    def test_every_registered_window_is_admitted(self) -> None:
        for window in sorted(W98_WINDOWS):
            _validate(draft_kv_window=window, draft_kv_sinks=16 if window else 0)

    def test_every_registered_skip_count_is_admitted(self) -> None:
        """Drive the sets from the runner so the two cannot drift apart."""
        sys.path.insert(0, str(SCRIPTS))
        from run_w98_g98b_round1 import SKIP_SETS

        self.assertEqual(set(SKIP_SETS), set(W98_SKIP_COUNTS))
        for count in sorted(W98_SKIP_COUNTS):
            _validate(draft_skip_layers=SKIP_SETS[count])

    def test_skip_sets_are_nested_so_keep_frac_is_a_scalar(self) -> None:
        """skip4 subset skip8 subset skip16.

        A non-nested set would confound "how many layers are skipped" with
        "which ones", and X1 measured that confound at up to 5%.
        """
        sys.path.insert(0, str(SCRIPTS))
        from run_w98_g98b_round1 import SKIP_SETS

        def indices(count: int) -> set[int]:
            return {int(x) for x in SKIP_SETS[count].split(",") if x.strip()}

        ordered = sorted(c for c in SKIP_SETS if c)
        for smaller, larger in zip(ordered, ordered[1:]):
            self.assertLess(indices(smaller), indices(larger))
        for count in ordered:
            self.assertEqual(len(indices(count)), count)
            self.assertTrue(all(0 <= i < 36 for i in indices(count)))

    def test_skip16_widens_the_keep_range_that_identifies_F(self) -> None:
        """The reason skip16 was admitted, pinned as a test.

        Round 1 spanned keep 0.778-1.0 (22%), too narrow to separate an
        intercept from a slope at the measured reproducibility.
        """
        keeps = sorted(1 - c / 36 for c in W98_SKIP_COUNTS)
        self.assertAlmostEqual(min(keeps), 1 - 16 / 36, places=6)
        self.assertGreater(max(keeps) - min(keeps), 0.4)

    def test_the_full_composition_is_admitted(self) -> None:
        _validate(
            draft=_ModelConfig("quantized-ckpt", "compressed-tensors"),
            share_weights=False,
            draft_kv_window=512,
            draft_kv_sinks=16,
            draft_skip_layers="2,4,7,16",
        )


class W98BoundaryTests(unittest.TestCase):
    """Relaxed does not mean unbounded."""

    def test_an_unregistered_window_is_refused(self) -> None:
        with self.assertRaises(KOffRuntimeError) as ctx:
            _validate(draft_kv_window=2048, draft_kv_sinks=16)
        self.assertIn("window must be one of", str(ctx.exception))

    def test_wrong_sinks_for_a_window_is_refused(self) -> None:
        with self.assertRaises(KOffRuntimeError) as ctx:
            _validate(draft_kv_window=512, draft_kv_sinks=0)
        self.assertIn("requires sinks=16", str(ctx.exception))

    def test_sinks_without_a_window_is_refused(self) -> None:
        with self.assertRaises(KOffRuntimeError):
            _validate(draft_kv_window=0, draft_kv_sinks=16)

    def test_an_unregistered_skip_count_is_refused(self) -> None:
        with self.assertRaises(KOffRuntimeError) as ctx:
            _validate(draft_skip_layers="2,4,7")
        self.assertIn("skip count must be one of", str(ctx.exception))

    def test_non_integer_skip_layers_are_refused(self) -> None:
        with self.assertRaises(KOffRuntimeError) as ctx:
            _validate(draft_skip_layers="two,four,seven,sixteen")
        self.assertIn("comma list of integers", str(ctx.exception))


class W98InvariantTests(unittest.TestCase):
    """Shared target KV and the forbidden mechanisms are never relaxed."""

    def test_shared_kv_is_still_required(self) -> None:
        with self.assertRaises(KOffRuntimeError) as ctx:
            _validate(shared_kv=False)
        self.assertIn("SHARED_KV=1", str(ctx.exception))

    def test_step0_decode_is_still_required(self) -> None:
        with self.assertRaises(KOffRuntimeError):
            _validate(shared_kv_step0_decode=False)

    def test_partial_replica_is_still_forbidden(self) -> None:
        with self.assertRaises(KOffRuntimeError) as ctx:
            _validate(draft_partial_replica="0,1")
        self.assertIn("no partial draft replica", str(ctx.exception))

    def test_draft_only_kv_dtype_is_still_forbidden(self) -> None:
        with self.assertRaises(KOffRuntimeError):
            _validate(draft_kv_dtype="fp8")

    def test_ahead_chain_is_still_forbidden(self) -> None:
        with self.assertRaises(KOffRuntimeError):
            _validate(ahead_chain=True)

    def test_consumed_ahead_chain_is_still_forbidden(self) -> None:
        with self.assertRaises(KOffRuntimeError):
            _validate(consume_ahead=True)

    def test_violation_message_names_the_scope(self) -> None:
        with self.assertRaises(KOffRuntimeError) as ctx:
            _validate(shared_kv=False)
        self.assertIn(BOOT_SCOPE_W98_LATTICE, str(ctx.exception))


if __name__ == "__main__":
    unittest.main()


class _Model:
    def __init__(self, params: dict) -> None:
        self._params = params

    def named_parameters(self):  # noqa: ANN201 - duck-typed for the proof
        return list(self._params.items())


def _param():
    """A real storage-backed parameter; the proof inspects storage identity."""
    import torch

    return torch.nn.Parameter(torch.zeros(4), requires_grad=False)


def _target_model(layers=(0, 1, 2)) -> _Model:
    return _Model({f"model.layers.{i}.mlp.down_proj.weight": _param() for i in layers})


class AliasProofScopeTests(unittest.TestCase):
    """The alias proof weakens per scope, and never silently disappears."""

    def test_minimal_b0_still_demands_set_equality(self) -> None:
        from vllm.v1.spec_decode.koff_runtime import validate_shared_weight_aliases

        target = _target_model()
        draft = _Model(dict(list(target.named_parameters())[:2]))
        with self.assertRaises(KOffRuntimeError) as ctx:
            validate_shared_weight_aliases(target, draft)
        self.assertIn("parameter sets differ", str(ctx.exception))

    def test_w98_admits_exactly_the_declared_skip_set(self) -> None:
        from vllm.v1.spec_decode.koff_runtime import validate_shared_weight_aliases

        target = _target_model()
        draft = _Model(
            {
                k: v
                for k, v in target.named_parameters()
                if not k.startswith("model.layers.2.")
            }
        )
        validate_shared_weight_aliases(
            target,
            draft,
            boot_scope=BOOT_SCOPE_W98_LATTICE,
            skip_layers="2",
        )

    def test_w98_refuses_absences_outside_the_skip_set(self) -> None:
        from vllm.v1.spec_decode.koff_runtime import validate_shared_weight_aliases

        target = _target_model()
        draft = _Model(
            {
                k: v
                for k, v in target.named_parameters()
                if not k.startswith("model.layers.1.")
            }
        )
        with self.assertRaises(KOffRuntimeError) as ctx:
            validate_shared_weight_aliases(
                target, draft, boot_scope=BOOT_SCOPE_W98_LATTICE, skip_layers="2"
            )
        self.assertIn("outside its declared skip set", str(ctx.exception))

    def test_extra_draft_parameters_are_refused_in_every_scope(self) -> None:
        from vllm.v1.spec_decode.koff_runtime import validate_shared_weight_aliases

        target = _target_model()
        draft = _Model({**dict(target.named_parameters()), "rogue.weight": _param()})
        for scope in (BOOT_SCOPE_MINIMAL_B0, BOOT_SCOPE_W98_LATTICE):
            with self.assertRaises(KOffRuntimeError) as ctx:
                validate_shared_weight_aliases(target, draft, boot_scope=scope)
            self.assertIn("adds parameters", str(ctx.exception))

    def test_independent_draft_proof_accepts_disjoint_weights(self) -> None:
        from vllm.v1.spec_decode.koff_runtime import (
            validate_independent_draft_weights,
        )

        target = _target_model()
        draft = _Model({"model.layers.0.mlp.down_proj.weight_packed": _param()})
        identity = validate_independent_draft_weights(target, draft)
        self.assertEqual(identity.parameter_alias_count, 0)
        self.assertTrue(identity.draft_version_id.startswith("w98-independent-draft-"))

    def test_independent_draft_proof_refuses_a_half_aliased_draft(self) -> None:
        """A partly-shared draft must not pass as 'independent'."""
        from vllm.v1.spec_decode.koff_runtime import (
            validate_independent_draft_weights,
        )

        target = _target_model()
        shared = dict(target.named_parameters())["model.layers.0.mlp.down_proj.weight"]
        draft = _Model({"model.layers.0.mlp.down_proj.weight_packed": shared})
        with self.assertRaises(KOffRuntimeError) as ctx:
            validate_independent_draft_weights(target, draft)
        self.assertIn("shares target weight storage", str(ctx.exception))


class StepEvidenceScopeTests(unittest.TestCase):
    """Per-step evidence records weight divergence instead of asserting equality."""

    def _record(self, scope: str, draft_version: str) -> dict:
        from vllm.v1.spec_decode.koff_runtime import (
            KOffRunnerEvidence,
            KOffSchedulerMetadata,
            build_live_step_record,
        )

        meta = KOffSchedulerMetadata(
            engine_step_index=1,
            verified_action_id="off",
            next_action_id="off",
            selection_intent="registered",
            capture_cohort_arm=False,
            decode_req_ids=("r0",),
            pure_decode=True,
            total_scheduled_kv_tokens=1,
            generated_suffix_min=1,
            generated_suffix_max=1,
            shared_target_kv_blocks_in_use=1,
            shared_target_kv_block_capacity=2,
            preemptions=0,
            recomputed_tokens=0,
            scheduled_at_s=0.0,
            aborted_action_id=None,
            discarded_draft_width=0,
            aborted_draft_request_count=0,
        )
        evidence = KOffRunnerEvidence(
            verified_action_id="off",
            next_action_id="off",
            target_graph_id="target-k1",
            verified_draft_graph_id=None,
            next_draft_graph_id=None,
            target_query_width=1,
            target_runtime_mode="FULL",
            draft_step0_query_width=None,
            draft_step0_num_tokens=None,
            draft_step0_batch_size=None,
            draft_step0_runtime_mode=None,
            draft_chain_runtime_mode=None,
            produced_draft_width=0,
            draft_dispatched=False,
            binding_id="live-kv-binding-x",
            pool_id="target-kv-pool-x",
            true_slot_mapping_id="target-true-slots-x",
            shared_kv_layer_count=1,
            shared_kv_storage_alias_count=1,
            target_weight_version_id="target-v1",
            draft_weight_version_id=draft_version,
            shared_weight_binding_id="target-alias-x",
            shared_weight_parameter_count=1,
            aborted_action_id=None,
            discarded_draft_width=0,
            aborted_draft_request_count=0,
            diagnostic=None,
        )
        return build_live_step_record(
            metadata=meta,
            evidence=evidence,
            raw_generated_lengths={"r0": 1},
            committed_lengths={"r0": 1},
            invalid_spec_tokens=0,
            elapsed_s=0.01,
            boot_scope=scope,
        )

    def test_minimal_b0_still_refuses_divergent_weights(self) -> None:
        with self.assertRaises(KOffRuntimeError) as ctx:
            self._record(BOOT_SCOPE_MINIMAL_B0, "draft-v2")
        self.assertIn("weight version diverged", str(ctx.exception))

    def test_w98_records_divergence_rather_than_refusing(self) -> None:
        record = self._record(BOOT_SCOPE_W98_LATTICE, "draft-v2")
        self.assertFalse(record["target_matching_weights"])
        self.assertEqual(record["target_weight_version_id"], "target-v1")
        self.assertEqual(record["draft_weight_version_id"], "draft-v2")
        self.assertEqual(record["boot_scope"], BOOT_SCOPE_W98_LATTICE)

    def test_matching_weights_are_still_reported_as_matching(self) -> None:
        for scope in (BOOT_SCOPE_MINIMAL_B0, BOOT_SCOPE_W98_LATTICE):
            record = self._record(scope, "target-v1")
            self.assertTrue(record["target_matching_weights"])
