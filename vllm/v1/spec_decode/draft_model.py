# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy

import torch
import torch.nn as nn
from typing_extensions import override

from vllm.config import VllmConfig
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
        from vllm.compilation.backends import set_model_tag

        draft_vllm_config = self._create_draft_vllm_config()
        with set_model_tag("draft_model"):
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
