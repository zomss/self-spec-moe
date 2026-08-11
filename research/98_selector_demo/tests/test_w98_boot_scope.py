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
import unittest

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
        sets = {0: "", 4: "2,4,7,16", 8: "2,4,7,11,16,20,25,30"}
        for count in sorted(W98_SKIP_COUNTS):
            _validate(draft_skip_layers=sets[count])

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
