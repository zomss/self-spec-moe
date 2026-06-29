#!/usr/bin/env python3
"""Tree-drafting building block: h(b) = P(verify's next token in draft's top-b).

Accept length of a tree is governed by, at each on-path node, the chance that verify's
greedy token is among the draft's top-b proposed children. For b=1 this is the chain
acceptance beta; b>1 is the branching gain. Measured along the verify-greedy (accepted)
path: at depth j, context = prompt + verify_greedy[:j], the draft (local routing, C=0.5E)
proposes top-8; record whether verify_greedy[j] is in the top-b for b=1..8, per depth.

This is the exact quantity that sets per-level accept prob on the accepted path (the draft
node that proposes depth-j candidates is conditioned on the accepted prefix verify[:j]).
Off-path tree nodes cost verify compute (node count) but not accept length -- handled
separately in the tree optimizer.
"""

from __future__ import annotations

import argparse
import json
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "The future of artificial intelligence is",
    "Write a Python function to compute the nth Fibonacci number.",
    "Explain mixture of experts models in simple terms.",
    "Solve step by step: if x + 3 = 7, what is x?",
    "Describe how a transformer attention layer works.",
    "List three benefits of expert parallelism.",
    "Summarize the theory of relativity in two sentences.",
    "What are the trade-offs of speculative decoding?",
]
BMAX = 8


def local_forward(module, allowed):
    def forward(self, hidden_states):
        hs = hidden_states.reshape(-1, self.hidden_dim)
        rl = F.linear(hs, self.weight)
        probs = F.softmax(rl, dim=-1, dtype=torch.float)
        masked = probs.masked_fill(~allowed, 0.0)
        v, i = torch.topk(masked, self.top_k, dim=-1)
        if self.norm_topk_prob:
            v = v / v.sum(dim=-1, keepdim=True)
        return rl, v.to(rl.dtype), i
    return types.MethodType(forward, module)


@contextmanager
def local_routing(gates, masks):
    saved = []
    try:
        for g, mk in zip(gates, masks):
            saved.append((g, g.forward))
            g.forward = local_forward(g, mk)
        yield
    finally:
        for g, o in saved:
            g.forward = o


def build_masks(model, ids, E, C, device):
    with torch.inference_mode():
        out = model(input_ids=ids, use_cache=False, output_router_logits=True)
    masks = []
    for rl in out.router_logits:
        mass = F.softmax(rl.float(), dim=-1).sum(0)
        mk = torch.zeros(E, dtype=torch.bool, device=device)
        mk[torch.topk(mass, C).indices] = True
        masks.append(mk)
    return masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--cache-frac", type=float, default=0.5)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--local-files-only", action="store_true")
    a = ap.parse_args()
    dev = a.device

    cfg = AutoConfig.from_pretrained(a.model, local_files_only=a.local_files_only)
    E = cfg.num_experts
    C = int(round(a.cache_frac * E))
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=a.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        local_files_only=a.local_files_only,
    ).to(dev).eval()
    gates = [m for m in model.modules() if type(m).__name__ == "Qwen3MoeTopKRouter"]
    D = a.depth
    print(f"[cfg] E={E} C={C}(0.5E) depth={D} prompts={len(PROMPTS)}")

    # hits[j][b-1] = #times verify_greedy[j] in draft top-b ; cnt[j] = samples at depth j
    hits = np.zeros((D, BMAX)); cnt = np.zeros(D)
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids.to(dev)
        masks = build_masks(model, ids, E, C, dev)
        # verify-greedy reference path (full model)
        ref, cur = [], ids
        for _ in range(D):
            with torch.inference_mode():
                t = int(model(input_ids=cur, use_cache=False).logits[0, -1].argmax())
            ref.append(t)
            cur = torch.cat([cur, torch.tensor([[t]], device=dev)], dim=1)
        # at each depth, draft top-b membership of verify token
        for j in range(D):
            ctx = torch.cat([ids, torch.tensor([ref[:j]], device=dev)], dim=1) if j else ids
            with local_routing(gates, masks), torch.inference_mode():
                dlog = model(input_ids=ctx, use_cache=False).logits[0, -1]
            dtop = torch.topk(dlog, BMAX).indices.tolist()
            cnt[j] += 1
            for b in range(1, BMAX + 1):
                if ref[j] in dtop[:b]:
                    hits[j][b - 1] += 1

    h_jb = hits / cnt[:, None]            # per-depth hit rate
    h_b = hits.sum(0) / cnt.sum()         # pooled over depth
    print("\npooled h(b) = P(verify-next in draft top-b):")
    for b in range(BMAX):
        print(f"  b={b+1}: {h_b[b]:.3f}" + ("  <- beta (chain)" if b == 0 else ""))
    print("\nper-depth h_j(b) (rows=depth, cols=b1..b8):")
    for j in range(D):
        print("  d%-2d " % j + " ".join(f"{h_jb[j][b]:.2f}" for b in range(BMAX)))

    a.output_json.parent.mkdir(parents=True, exist_ok=True)
    a.output_json.write_text(json.dumps({
        "model": a.model, "C": C, "depth": D, "bmax": BMAX,
        "h_pooled": [round(x, 4) for x in h_b.tolist()],
        "h_per_depth": [[round(x, 4) for x in row] for row in h_jb.tolist()],
    }, indent=2))
    print(f"[saved] {a.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
