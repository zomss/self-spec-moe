"""CPU-only tests for the Phase 97 w512 acceptance-equivalence proof."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

import torch

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.spec_decode.koff_runtime import (
    slot_mapping_identity,
    validate_shared_kv_aliases,
    validate_shared_weight_aliases,
)
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR / "scripts"))

from validate_p4_w512_equivalence import (  # noqa: E402
    W512EquivalenceError,
    compute_mask_coverage,
    validate_proof,
)

PROOF_PATH = PHASE_DIR / "data" / "p4" / "p4_w512_acceptance_equivalence.json"


def valid_proof() -> dict:
    """Return an independent copy of the checked-in equivalence proof."""
    return json.loads(PROOF_PATH.read_text(encoding="utf-8"))


def _proposer() -> SpecDecodeBaseProposer:
    proposer = object.__new__(SpecDecodeBaseProposer)
    proposer.block_size = 16
    proposer._kv_window = 512
    proposer._kv_window_sinks = 16
    proposer.max_model_len = 20480
    proposer.max_batch_size = 1
    proposer.device = torch.device("cpu")
    proposer._win_block_table = None
    proposer._win_seq_lens = None
    proposer._win_col_arange = None
    proposer._win_src_cols = None
    proposer._win_col_ge_sink = None
    proposer._kv_window_debug = False
    proposer._kv_window_calls = 0
    return proposer


def _metadata(seq_len: int) -> CommonAttentionMetadata:
    max_cols = 20480 // 16
    block_table = torch.arange(max_cols, dtype=torch.int32).unsqueeze(0)
    return CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([seq_len], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=1,
        max_query_len=1,
        max_seq_len=seq_len,
        block_table_tensor=block_table,
        slot_mapping=torch.tensor([123], dtype=torch.int64),
    )


class P4W512EquivalencePositiveTests(unittest.TestCase):
    """Prove the exact acceptance-only semantic boundary."""

    def test_checked_in_proof_remains_a_valid_historical_snapshot(self) -> None:
        result = validate_proof(valid_proof(), enforce_current_symbols=False)

        self.assertEqual(result["status"], "pass")
        self.assertFalse(result["implementation_symbols_current"])
        self.assertTrue(result["acceptance_surrogate_eligible"])
        self.assertEqual(result["mask_cases_checked"], 163830)
        self.assertEqual(result["shared_kv_layer_count"], 36)
        self.assertFalse(result["latency_transfer_allowed"])
        self.assertFalse(result["cost_credit_allowed"])
        self.assertFalse(result["gpu_measurement_authorized"])
        self.assertFalse(result["p4a_engineering_authorized"])
        self.assertEqual(
            result["remaining_blockers"],
            ["conservative_resource_bound_missing"],
        )

    def test_checked_in_proof_detects_changed_implementation_symbols(self) -> None:
        with self.assertRaisesRegex(
            W512EquivalenceError,
            "implementation symbol hash mismatch",
        ):
            validate_proof(valid_proof())

    def test_exhaustive_mask_digest_is_stable(self) -> None:
        count, digest = compute_mask_coverage(20480)

        self.assertEqual(count, 163830)
        self.assertEqual(
            digest,
            "6134578e29da4b30884c4de1ecfb6b2f7bcc338393e1a29f4d2334fa1b77d62a",
        )

    def test_live_window_transform_keeps_exact_registered_pages(self) -> None:
        cases = (
            (1, 0),
            (528, 0),
            (529, 0),
            (544, 1),
            (545, 1),
            (20480, 0),
            (20480, 3),
        )
        for seq_len, query_offset in cases:
            with self.subTest(seq_len=seq_len, query_offset=query_offset):
                proposer = _proposer()
                metadata = _metadata(seq_len)
                original_blocks = metadata.block_table_tensor.clone()
                original_seq_lens = metadata.seq_lens.clone()
                original_slots = metadata.slot_mapping

                windowed = proposer._apply_draft_kv_window(
                    metadata, tokens_drafted=query_offset
                )

                n_total = (seq_len + 15) // 16
                n_last = (512 + query_offset + 15) // 16
                start_last = min(max(n_total - n_last, 1), n_total)
                dropped = max(start_last - 1, 0)
                if dropped:
                    expected_blocks = [0, *range(start_last, n_total)]
                else:
                    expected_blocks = list(range(n_total))
                observed_blocks = windowed.block_table_tensor[
                    0, : len(expected_blocks)
                ].tolist()

                self.assertEqual(observed_blocks, expected_blocks)
                self.assertEqual(windowed.seq_lens.item(), seq_len - dropped * 16)
                self.assertIs(windowed.slot_mapping, original_slots)
                self.assertTrue(
                    torch.equal(metadata.block_table_tensor, original_blocks)
                )
                self.assertTrue(torch.equal(metadata.seq_lens, original_seq_lens))

    def test_shared_kv_and_true_slot_identity_are_object_exact(self) -> None:
        layer_mapping: dict[str, str] = {}
        kv_caches: dict[str, torch.Tensor] = {}
        for layer_idx in range(36):
            target_name = f"model.layers.{layer_idx}.self_attn"
            draft_name = f"draft_model.model.layers.{layer_idx}.self_attn"
            cache = torch.empty(2, 16, 2, 8)
            layer_mapping[draft_name] = target_name
            kv_caches[target_name] = cache
            kv_caches[draft_name] = cache

        identity = validate_shared_kv_aliases(
            layer_mapping, kv_caches, pool_object=object()
        )
        slots = torch.arange(64, dtype=torch.int64)
        before = slot_mapping_identity({"target": slots})
        after = slot_mapping_identity({name: slots[:1] for name in layer_mapping})

        self.assertEqual(identity.layer_count, 36)
        self.assertEqual(identity.storage_alias_count, 36)
        self.assertEqual(before, after)

    def test_target_matching_weights_are_exact_parameter_aliases(self) -> None:
        target = torch.nn.Linear(4, 3)
        identity = validate_shared_weight_aliases(target, target)

        self.assertEqual(identity.target_version_id, identity.draft_version_id)
        self.assertEqual(identity.parameter_alias_count, 2)


class P4W512EquivalenceFailClosedTests(unittest.TestCase):
    """Reject source drift, semantic drift, and authority inflation."""

    def test_rejects_upstream_artifact_hash_drift(self) -> None:
        proof = valid_proof()
        proof["source_artifacts"]["base_boot"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(W512EquivalenceError, "artifact hash mismatch"):
            validate_proof(proof)

    def test_rejects_implementation_symbol_drift(self) -> None:
        proof = valid_proof()
        proof["implementation_symbols"][1]["source_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            W512EquivalenceError, "implementation symbol hash mismatch"
        ):
            validate_proof(proof)

    def test_rejects_an_alternate_window(self) -> None:
        proof = valid_proof()
        proof["surrogate_contract"]["required_environment"][
            "VLLM_SELF_SPEC_DRAFT_KV_WINDOW"
        ] = "2048"
        with self.assertRaisesRegex(W512EquivalenceError, "environment"):
            validate_proof(proof, enforce_current_symbols=False)

    def test_rejects_truncated_sequence_coverage(self) -> None:
        proof = valid_proof()
        proof["proof"]["mask"]["sequence_lengths_checked"][1] = 2048
        with self.assertRaisesRegex(W512EquivalenceError, "length interval"):
            validate_proof(proof, enforce_current_symbols=False)

    def test_rejects_mask_digest_drift(self) -> None:
        proof = valid_proof()
        proof["proof"]["mask"]["coverage_sha256"] = "0" * 64
        with self.assertRaisesRegex(W512EquivalenceError, "coverage digest"):
            validate_proof(proof, enforce_current_symbols=False)

    def test_rejects_boundary_interval_drift(self) -> None:
        proof = valid_proof()
        proof["proof"]["mask"]["boundary_observations"][2]["visible_key_intervals"][1][
            0
        ] = 16
        with self.assertRaisesRegex(W512EquivalenceError, "boundary observation"):
            validate_proof(proof, enforce_current_symbols=False)

    def test_rejects_noncanonical_true_slots(self) -> None:
        proof = valid_proof()
        proof["proof"]["canonical_true_slots"]["mapping_id"] = "other-slots"
        with self.assertRaisesRegex(W512EquivalenceError, "noncanonical"):
            validate_proof(proof, enforce_current_symbols=False)

    def test_rejects_window_cost_credit(self) -> None:
        proof = valid_proof()
        proof["claims"]["window_cost_credit_allowed"] = True
        with self.assertRaises(W512EquivalenceError):
            validate_proof(proof, enforce_current_symbols=False)

    def test_rejects_executable_action_claim(self) -> None:
        proof = valid_proof()
        proof["claims"]["executable_masked_action_exists"] = True
        with self.assertRaises(W512EquivalenceError):
            validate_proof(proof, enforce_current_symbols=False)

    def test_rejects_gpu_authorization(self) -> None:
        proof = copy.deepcopy(valid_proof())
        proof["authorizations"]["gpu_measurement"] = True
        with self.assertRaises(W512EquivalenceError):
            validate_proof(proof, enforce_current_symbols=False)


if __name__ == "__main__":
    unittest.main()
