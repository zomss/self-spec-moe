#!/usr/bin/env python3
"""W1 correct-routing core: comm-free local routing (skip-cold) -- reference + tests.

This is the routing the vLLM draft path must apply when the local-routing flag is set:
mask the gate probabilities to the per-device RESIDENT expert set, top-k over the survivors,
renormalize. Tokens whose true top-k include non-resident experts fall back to their top
RESIDENT experts (skip-cold). At resident = all experts this is identical to full routing.

Reference (matches the harness local_forward used across Phases 25/27/28). The vLLM
integration applies the same masking inside the GPU routing path; this file pins the
semantics and is unit-testable with no model.

Run: python local_routing_ref.py   (executes the asserts)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def full_route(router_logits: torch.Tensor, top_k: int, norm_topk_prob: bool = True,
               scoring: str = "softmax") -> tuple[torch.Tensor, torch.Tensor]:
    """Standard full-EP routing: top-k over all experts."""
    probs = (F.softmax(router_logits, dim=-1, dtype=torch.float)
             if scoring == "softmax" else router_logits.sigmoid().float())
    w, ids = torch.topk(probs, top_k, dim=-1)
    if norm_topk_prob:
        w = w / w.sum(dim=-1, keepdim=True)
    return ids, w.to(router_logits.dtype)


def local_route(router_logits: torch.Tensor, resident_mask: torch.Tensor, top_k: int,
                norm_topk_prob: bool = True, scoring: str = "softmax",
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """Comm-free local routing with skip-cold.

    Args:
        router_logits: [n_tokens, E] gate logits.
        resident_mask: [E] bool, True for experts resident on THIS device.
        top_k: experts per token.
    Returns (topk_ids [n, top_k] -- all in the resident set, topk_weights [n, top_k]).
    """
    assert resident_mask.dtype == torch.bool and resident_mask.any()
    probs = (F.softmax(router_logits, dim=-1, dtype=torch.float)
             if scoring == "softmax" else router_logits.sigmoid().float())
    # skip-cold: zero out non-resident experts before top-k.
    masked = probs.masked_fill(~resident_mask, 0.0)
    w, ids = torch.topk(masked, top_k, dim=-1)
    if norm_topk_prob:
        denom = w.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float).tiny)
        w = w / denom
    return ids, w.to(router_logits.dtype)


def _tests():
    torch.manual_seed(0)
    n, E, k = 16, 64, 6
    logits = torch.randn(n, E)

    # 1) resident = all experts -> local == full routing (sanity).
    all_mask = torch.ones(E, dtype=torch.bool)
    fi, fw = full_route(logits, k)
    li, lw = local_route(logits, all_mask, k)
    assert torch.equal(fi, li), "resident=all must match full top-k ids"
    assert torch.allclose(fw.float(), lw.float(), atol=1e-6), "weights must match"

    # 2) subset resident -> all selected ids are resident, weights renormed to 1.
    res = torch.zeros(E, dtype=torch.bool); res[:16] = True   # first 16 experts resident
    li, lw = local_route(logits, res, k)
    assert res[li].all(), "all selected experts must be resident (skip-cold)"
    assert torch.allclose(lw.float().sum(-1), torch.ones(n), atol=1e-5), "renorm to 1"

    # 3) skip-cold actually differs from full when the true top-k leak off-resident.
    fi, _ = full_route(logits, k)
    leaked = (~res[fi]).any(dim=-1)            # tokens whose full top-k include cold experts
    assert leaked.any(), "test setup should have some cold-leaking tokens"
    # for a leaked token, the local ids differ from the full ids
    assert not torch.equal(fi[leaked], li[leaked]), "skip-cold must reroute leaked tokens"

    # 4) no-renorm path keeps the masked-prob magnitude (drop-cold, weight lost).
    li2, lw2 = local_route(logits, res, k, norm_topk_prob=False)
    assert (lw2.float().sum(-1) <= 1.0 + 1e-5).all(), "no-renorm weights <= 1"

    # 5) sigmoid scoring (DeepSeek-V3 style) runs and stays resident.
    li3, _ = local_route(logits, res, k, scoring="sigmoid")
    assert res[li3].all()

    print("local_routing_ref: all tests passed "
          f"(n={n}, E={E}, k={k}; cold-leaking tokens={int(leaked.sum())}/{n})")


if __name__ == "__main__":
    _tests()
