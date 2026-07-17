# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy

import torch
import torch.nn as nn
from typing_extensions import override

from vllm import envs
from vllm.config import VllmConfig
from vllm.config.quantization import resolve_quantization_config
from vllm.config.utils import replace
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

logger = init_logger(__name__)


class DraftModelProposer(SpecDecodeBaseProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            pass_hidden_states_to_model=False,
            runner=runner,
        )
        self._raise_if_vocab_size_mismatch()
        self._raise_if_draft_tp_mismatch()

    def _raise_if_vocab_size_mismatch(self):
        self.speculative_config.verify_equal_vocab_size_if_draft_model()

    def _raise_if_draft_tp_mismatch(self):
        # Note(Tomas Ruiz) If we run the target model with TP > 1 and
        # the draft model with TP = 1, then the different TP ranks collide.
        # Specifically when all ranks compile the draft model on rank 0
        # (because TP=1), then the torch compile cache is overwritten and corrupted.
        # We need a mechanism like this: https://github.com/vllm-project/vllm/pull/5414
        # To prevent this error, we assert that both TP sizes must be the same.
        spec_cfg = self.speculative_config
        tgt_tp = spec_cfg.target_parallel_config.tensor_parallel_size
        draft_tp = spec_cfg.draft_parallel_config.tensor_parallel_size
        if draft_tp != tgt_tp:
            raise ValueError(
                f"Currently, 'draft_tensor_parallel_size' and 'tensor_parallel_size' "
                f"must be the same. Got {draft_tp} and {tgt_tp}. "
                "Please pass 'draft_tensor_parallel_size' in the speculative_config."
            )

    @override
    def _create_draft_vllm_config(self) -> VllmConfig:
        base = super()._create_draft_vllm_config()
        spec = self.speculative_config

        # By default the draft runs unquantized (bf16). If the speculative
        # config requests a draft quantization (e.g. "fp8"), derive the draft's
        # quant_config from the draft model_config (which already carries that
        # quantization). This lets the DRAFT use on-the-fly FP8 (halving its
        # expert weight-load and FLOPs) while the TARGET stays bf16. Default
        # (no draft quantization) keeps the original quant_config=None behavior.
        quant_config = None
        if spec.draft_model_config.quantization is not None:
            load_config = spec.draft_load_config or self.vllm_config.load_config
            # get_quant_config reads hf_overrides as a dict when the checkpoint
            # has no embedded quant config (the on-the-fly FP8 path). The draft
            # model_config carries hf_overrides as the SpeculativeConfig callable,
            # so normalize it to {} on a copy (get_quantization_config deep-copies
            # anyway, but be explicit so the callable is never indexed).
            quant_model_config = copy.copy(spec.draft_model_config)
            if not isinstance(quant_model_config.hf_overrides, dict):
                quant_model_config.hf_overrides = {}
            # Online-quant shorthands ("fp8_per_block", "fp8_per_channel", ...)
            # carry no checkpoint config; they must be desugared into
            # quantization_config, which EngineArgs only does for the TARGET.
            # Returns None for checkpoint-based names ("fp8"), leaving those
            # unchanged.
            quant_model_config.quantization_config = resolve_quantization_config(
                quant_model_config.quantization,
                quant_model_config.quantization_config,
            )
            quant_config = VllmConfig.get_quantization_config(
                quant_model_config, load_config
            )
            logger.info(
                "Draft model quantization: %s (target stays %s)",
                spec.draft_model_config.quantization,
                self.vllm_config.model_config.quantization,
            )

        return replace(
            base,
            quant_config=quant_config,
            parallel_config=replace(
                spec.draft_parallel_config,
                rank=self.vllm_config.parallel_config.rank,
            ),
            model_config=spec.draft_model_config,
        )

    @override
    def _get_model(self) -> nn.Module:
        from contextlib import contextmanager

        from vllm.compilation.backends import set_model_tag
        from vllm.config.compilation import CompilationMode, CUDAGraphMode

        @contextmanager
        def _maybe_uncompiled_draft(cc):
            # W7-dp: build the draft UNCOMPILED. For non-MLA MoE self-spec
            # (Qwen2/Qwen3-MoE under FLASH_ATTN) torch.compile produces an MoE
            # forward that is numerically INCONSISTENT between the separately
            # compiled draft ("draft_model" tag) and verify ("backbone" tag):
            # with compile on the draft's per-position greedy tokens diverge from
            # the verify's and acceptance collapses (Qwen1.5-MoE K=4: per-token
            # 0.94 eager vs 0.14 compiled; even the EP-shard draft and the verify
            # disagree). Dense and MLA-MoE drafts are unaffected. The verify model
            # is built BEFORE the draft, so flipping these flags now affects only
            # the draft's per-module do_not_compile decision; we restore them
            # after so nothing else sees the change. We deliberately mutate the
            # SHARED compilation_config (not a replace()) so static_forward_context
            # stays the same dict and the draft's attn/MoE layers register where
            # the proposer's runtime forward context looks them up.
            # NOTE: this is a partial mitigation -- it makes the draft eager but
            # the verify stays compiled, so they still partially disagree (accept
            # recovers only ~1.6->1.9 on Qwen1.5-MoE). Full recovery (->4.8)
            # needs the WHOLE engine eager (enforce_eager), so draft and verify
            # share the eager numerics. Default off; see research/36_dp_accept.
            if not envs.VLLM_SELF_SPEC_DRAFT_EAGER:
                yield
                return
            logger.info_once(
                "Self-spec: building draft model UNCOMPILED "
                "(VLLM_SELF_SPEC_DRAFT_EAGER=1); verify model stays compiled."
            )
            old_mode, old_cg = cc.mode, cc.cudagraph_mode
            cc.mode = CompilationMode.NONE
            cc.cudagraph_mode = CUDAGraphMode.NONE
            try:
                yield
            finally:
                cc.mode, cc.cudagraph_mode = old_mode, old_cg

        draft_vllm_config = self._create_draft_vllm_config()
        # Phase 83: draft PARTIAL replica -- activate the build context so
        # FusedMoE construction installs the frequency-profiled per-layer
        # expert maps (non-resident experts are never loaded).
        from vllm.model_executor.layers.fused_moe.expert_map_manager import (
            draft_partial_replica_build,
        )

        with set_model_tag("draft_model"), _maybe_uncompiled_draft(
            draft_vllm_config.compilation_config
        ), draft_partial_replica_build():
            model = get_model(
                vllm_config=draft_vllm_config,
                prefix="draft_model",
            )
        self._log_draft_moe_ep_status(model)
        return model

    @staticmethod
    def _log_draft_moe_ep_status(model: nn.Module) -> None:
        """One-time check (W0 validation rung ii): confirm the draft's MoE
        layers build with expert parallelism (use_ep=True, non-None expert_map).
        Without EP propagated into the draft parallel config, use_ep=False and
        the comm-free local-routing path is a silent no-op on the draft.
        """
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

        for name, module in model.named_modules():
            if isinstance(module, RoutedExperts):
                logger.info(
                    "Draft MoE EP status [%s]: use_ep=%s expert_map=%s",
                    name,
                    module.use_ep,
                    "None" if module.expert_map is None else "set",
                )
                return

    @override
    def _maybe_share_embeddings(self, target_language_model: nn.Module) -> None:
        # Draft models don't share embeddings with the target model
        pass

    @override
    def _maybe_share_lm_head(self, target_language_model: nn.Module) -> None:
        # Draft models don't share lm_head with the target model
        pass
