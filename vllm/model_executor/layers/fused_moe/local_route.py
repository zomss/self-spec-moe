# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Self-spec W1: CORRECT comm-free local-routing MoE forward helpers.

When the local-routing flag is set, each rank routes its LOCAL tokens to only
its RESIDENT experts (the EP shard, via ``expert_map``) and skips BOTH the AgRs
dispatch (allgather) and combine (reducescatter). This is the correct,
non-garbage counterpart to ``VLLM_SELF_SPEC_SKIP_A2A`` (which is timing-only and
tiles the local chunk to keep the gathered shape).

The flag rides on ``ForwardContext.additional_kwargs[SELF_SPEC_LOCAL_ROUTE_KEY]``
(the signal channel set by the lockstep driver's draft forward); the env var
``VLLM_SELF_SPEC_LOCAL_ROUTE`` is the default when that key is absent. The flag
reader lives in ``vllm.forward_context`` (low-level, no MoE deps) and is
re-exported here as ``local_route_enabled`` for the routing code.

The router masking matches ``local_routing_ref.local_route``: mask the gate
probabilities to the resident experts (set non-resident logits to the dtype min
so softmax/sigmoid -> 0), top-k over the survivors, renormalize over the
selected top-k. At resident = all experts this is identical to full routing.
"""

import torch

import vllm.envs as envs
from vllm.forward_context import (
    SELF_SPEC_LOCAL_ROUTE_KEY,
    self_spec_local_route_enabled as local_route_enabled,
)

__all__ = [
    "SELF_SPEC_LOCAL_ROUTE_KEY",
    "local_route_enabled",
    "mask_router_logits_to_resident",
    "draft_topc",
    "prune_topk_to_topc",
]


def draft_topc() -> int:
    """The draft top-C prune width (VLLM_SELF_SPEC_DRAFT_TOPC); 0 = off."""
    return envs.VLLM_SELF_SPEC_DRAFT_TOPC


def prune_topk_to_topc(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    c: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prune the model's top_k selection to the C largest-gate-weight experts.

    A *smart* prune: for each token, keep the C experts with the largest gate
    weights (drop the smallest), renormalize the kept weights to sum to 1, and
    narrow the tensors to width C so the fused kernel physically computes only C
    experts per token. Used by the comm-free self-spec DRAFT forward to trade a
    little acceptance for draft compute; the full-top_k VERIFY corrects, so the
    output stays lossless regardless of C.

    Args:
        topk_weights: ``[n_tokens, top_k]`` renormalized gate weights.
        topk_ids: ``[n_tokens, top_k]`` selected expert ids.
        c: Number of experts to keep per token (``1 <= c <= top_k``). Values
            ``>= top_k`` or ``<= 0`` return the inputs unchanged.

    Returns:
        ``(weights_c, ids_c)`` of width C, weights renormalized over the kept C.
        The original dtypes are preserved.
    """
    top_k = topk_weights.shape[-1]
    if c <= 0 or c >= top_k:
        return topk_weights, topk_ids
    # Largest-C by gate weight (values), gather the matching ids.
    kept_w, kept_pos = torch.topk(topk_weights, c, dim=-1)
    kept_ids = torch.gather(topk_ids, -1, kept_pos)
    # Renormalize the kept weights over the retained C (sum to 1), matching the
    # full-top_k renormalization semantics but over the C survivors. Guard the
    # degenerate all-zero row (never happens post-softmax but keep it safe).
    denom = kept_w.sum(dim=-1, keepdim=True)
    denom = torch.where(denom > 0, denom, torch.ones_like(denom))
    kept_w = (kept_w / denom).to(topk_weights.dtype)
    return kept_w, kept_ids.to(topk_ids.dtype)


def mask_router_logits_to_resident(
    router_logits: torch.Tensor,
    expert_map: torch.Tensor | None,
) -> torch.Tensor:
    """Mask router logits to the resident experts (skip-cold), additively.

    Sets the logits of non-resident experts to the dtype min so that any
    downstream softmax/sigmoid scoring drives their probability to 0, so they
    are never selected by top-k. The top-k *selection* among the resident
    experts and the renormalized top-k weights are then identical to masking the
    post-softmax probabilities (the ``local_routing_ref.local_route``
    semantics), because renormalization divides by the sum over the selected
    survivors either way.

    Args:
        router_logits: ``[n_tokens, global_num_experts]`` gate logits.
        expert_map: ``[global_num_experts]`` int tensor, ``-1`` for non-resident
            experts and the local slot otherwise. ``None`` means every expert is
            resident (no EP shard) -> no-op, preserving the sanity invariant
            that resident=all matches full routing exactly.

    Returns:
        The masked logits (a new tensor) when ``expert_map`` is not None,
        otherwise ``router_logits`` unchanged.
    """
    if expert_map is None:
        return router_logits
    resident = expert_map.to(router_logits.device) >= 0  # [E] bool
    neg_inf = torch.finfo(router_logits.dtype).min
    return router_logits.masked_fill(~resident.unsqueeze(0), neg_inf)
