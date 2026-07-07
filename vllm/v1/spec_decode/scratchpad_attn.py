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

from dataclasses import dataclass

import torch


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

    # Gather the kept (sink + trailing-window) pages into a dense scratchpad.
    idx = ctx.block_table[:bs, :n_kept].reshape(-1)
    k = key_cache.index_select(0, idx).view(bs, cap, num_kv_heads, head_dim)
    v = value_cache.index_select(0, idx).view(bs, cap, num_kv_heads, head_dim)
    # [bs, num_kv_heads, cap, head_dim]
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)

    num_heads = query.shape[1]
    # [bs, num_heads, 1, head_dim]
    q = query[:bs].to(k.dtype).unsqueeze(2)
    rep = num_heads // num_kv_heads
    if rep > 1:
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

    scale = layer.impl.scale
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [bs, num_heads, 1, cap]
    # Padding mask: scratchpad slot >= valid seq_len -> -inf (masked out). The
    # compaction lays sinks then trailing pages contiguously with padding only
    # at the tail, so a simple (col >= seq_len) test is exact.
    invalid = ctx.col_arange[:, :cap] >= ctx.seq_lens[:bs].unsqueeze(1)  # [bs, cap]
    scores = scores.masked_fill(invalid.unsqueeze(1).unsqueeze(1), float("-inf"))
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(v.dtype)
    out = torch.matmul(probs, v)  # [bs, num_heads, 1, head_dim]
    # Padded (num_reqs..num_tokens) rows have no valid window -> zero them so
    # their (discarded) logits stay finite; real rows get the attention output.
    if output.shape[0] > bs:
        output[bs:].zero_()
    output[:bs] = out.squeeze(2).to(output.dtype)
