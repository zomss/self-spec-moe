# SPDX-License-Identifier: Apache-2.0
"""Window-scratchpad dense attention for the self-spec draft chain (Phase 69).

The paged FA3 (GQA) decode kernel freezes its host-side work distribution at
CUDA-graph capture and does not re-derive it from the live ``seqused_k`` at
replay, so the draft chain's growing sequence cannot replay a captured FULL
graph -> the chain is forced to eager attention + PIECEWISE dispatch
(~98 graph-piece launches / step, the ~16 ms host-side launch bubble).

When ``VLLM_SELF_SPEC_DRAFT_KV_WINDOW`` bounds the draft's KV read to a small
FIXED set (sinks S + trailing window W + drafted <= K), each chain step attends
over a cycle-constant number of pages. This module materialises that windowed
key set into a FIXED-shape dense scratchpad and runs a plain masked attention
(gather + matmul + softmax + matmul) -- pure shape-driven tensor ops with no
host-side scheduling, hence fully CUDA-graph-capturable. The whole per-step
draft forward then replays as ONE FULL graph.

Correctness: the gathered key set is the SAME sinks+window pages the paged
windowed FA3 reads (same compacted block table, same bf16 cached RoPE'd KV),
the decode causal mask (query attends to all valid past keys) is reproduced by
masking padded scratchpad slots to -inf, and the softmax scale is the layer's
own ``impl.scale``. Greedy accept is therefore bit-exact to the paged windowed
attention modulo attention-kernel ULP (which does not flip argmax in practice).
"""

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

# Phase 81 E1b: the SDPA path at q_len=1 with an additive mask dispatches
# torch's MEM-EFFICIENT backend -> a gemv/elementwise decomposition costing
# ~0.25 ms/layer (~7-9 ms/draft-step, the 81-E1 `sp` arm's 0.43x). The fused
# FA3 varlen kernel does the same work in ~30 us -- and it is capture-safe
# HERE because the scratchpad geometry is CONSTANT (max_seqlen_k = cap,
# batch fixed): the P35 freeze needs a GROWING sequence; per-row validity is
# read on-device from the persistent seq_lens buffer via ``seqused_k``.
# Set VLLM_SELF_SPEC_SCRATCHPAD_SDPA=1 to restore the old SDPA path.
_USE_SDPA = os.environ.get("VLLM_SELF_SPEC_SCRATCHPAD_SDPA", "0") == "1"
try:
    from vllm.vllm_flash_attn import flash_attn_varlen_func as _fa_varlen
except ImportError:  # pragma: no cover
    _fa_varlen = None
_CU_CACHE: dict = {}


def _cu_seqlens(bs: int, stride: int, device) -> torch.Tensor:
    key = (bs, stride, device)
    t = _CU_CACHE.get(key)
    if t is None:
        t = torch.arange(0, (bs + 1) * stride, stride,
                         dtype=torch.int32, device=device)
        _CU_CACHE[key] = t
    return t


@dataclass
class DraftScratchpadCtx:
    """Per-cycle handle carried on the draft chain forward context.

    Holds references to the proposer's PERSISTENT window buffers (updated
    in place each chain step, so a captured graph reads the live data through
    the capture-time pointers) plus the fixed gather geometry.
    """

    # Full persistent compacted-window block table [rows, n_cols]; row r column
    # c holds the physical block id of the c-th kept (sink/trailing) page for
    # request r. Only the first ``n_kept_blocks`` columns are gathered.
    block_table: torch.Tensor
    # Full persistent per-request valid key length [rows] (sinks + window +
    # drafted-so-far); slots >= this are padding and get masked to -inf.
    seq_lens: torch.Tensor
    # Persistent [1, cap] arange for the padding mask (cap = n_kept*block_size).
    col_arange: torch.Tensor
    block_size: int
    n_kept_blocks: int
    # Number of real (unpadded) requests in this batch.
    num_reqs: int


def scratchpad_attention(
    ctx: DraftScratchpadCtx,
    layer: torch.nn.Module,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Fixed-shape dense windowed attention for one draft attention layer.

    Args:
        ctx: window geometry + persistent buffers (see DraftScratchpadCtx).
        layer: the Attention layer (for ``impl.scale`` / head counts).
        query: [num_tokens, num_heads, head_dim], RoPE'd (num_tokens == batch,
            one decode query per request; padded rows past num_reqs ignored).
        kv_cache: [num_blocks, 2, block_size, num_kv_heads, head_dim] paged
            cache (shared with the target; the current token's K/V were already
            written by the preceding unified_kv_cache_update).
        output: [num_tokens, num_heads, head_dim] to write attention out into.
    """
    bs = ctx.num_reqs
    key_cache, value_cache = kv_cache.unbind(1)
    # Logical [num_blocks, block_size, num_kv_heads, head_dim] (index_select
    # respects strides, so this is correct for HND/NHD physical layouts).
    num_kv_heads = key_cache.shape[2]
    head_dim = key_cache.shape[3]
    n_kept = ctx.n_kept_blocks
    cap = n_kept * ctx.block_size

    if _fa_varlen is not None and not _USE_SDPA:
        # E1b fused path: PAGED FA3 varlen over the compacted window block
        # table -- no gather at all (the dense scratchpad becomes unnecessary;
        # FA3 reads the kept pages directly). Geometry is cycle-constant
        # (total_q = bs, max_seqlen_k = cap, table width n_kept), so the
        # host-side schedule captured into the FULL graph stays valid across
        # replays; live per-row lengths and page ids are device-read from the
        # PERSISTENT seq_lens / block_table buffers (updated in place each
        # step -- the captured pointers see the live data).
        if output.shape[0] > bs:
            output[bs:].zero_()
        _fa_varlen(
            q=query[:bs].to(key_cache.dtype),
            k=key_cache,
            v=value_cache,
            max_seqlen_q=1,
            cu_seqlens_q=_cu_seqlens(bs, 1, query.device),
            max_seqlen_k=cap,
            seqused_k=ctx.seq_lens[:bs],
            block_table=ctx.block_table[:bs, :n_kept],
            softmax_scale=layer.impl.scale,
            causal=False,
            out=output[:bs],
        )
        return

    # SDPA fallback: gather the kept pages into a dense scratchpad,
    # [bs, cap, num_kv_heads, head_dim].
    idx = ctx.block_table[:bs, :n_kept].reshape(-1)
    k = key_cache.index_select(0, idx).view(bs, cap, num_kv_heads, head_dim)
    v = value_cache.index_select(0, idx).view(bs, cap, num_kv_heads, head_dim)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    num_heads = query.shape[1]
    # [bs, num_heads, 1, head_dim]
    q = query[:bs].to(k.dtype).unsqueeze(2)

    # Additive padding mask [bs, 1, 1, cap] (-inf on padded slots). The
    # compaction lays sinks then trailing pages contiguously with padding only
    # at the tail, so a simple (col >= seq_len) test is exact. Every row keeps
    # >= sinks valid slots, so no row is fully masked (no softmax NaN).
    invalid = ctx.col_arange[:, :cap] >= ctx.seq_lens[:bs].unsqueeze(1)  # [bs, cap]
    attn_mask = torch.zeros(
        (bs, 1, 1, cap), dtype=q.dtype, device=q.device
    ).masked_fill_(invalid[:, None, None, :], float("-inf"))
    # Fused GQA attention: no head materialisation (enable_gqa), fp32-accumulate
    # softmax inside the kernel -> the paged windowed FA3's key set, done as a
    # fixed-shape CUDA-graph-capturable op.
    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, scale=layer.impl.scale, enable_gqa=True
    )  # [bs, num_heads, 1, head_dim]
    # Padded (num_reqs..num_tokens) rows have no valid window -> zero them so
    # their (discarded) logits stay finite; real rows get the attention output.
    if output.shape[0] > bs:
        output[bs:].zero_()
    output[:bs] = out.squeeze(2).to(output.dtype)
