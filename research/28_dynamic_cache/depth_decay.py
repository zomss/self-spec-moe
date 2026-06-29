#!/usr/bin/env python3
"""Depth>0 decay: does the dynamic-cache memory win survive k>1 / trees?

Within a cycle only the FIRST draft token predicts from a committed token (experts known
to verify -> last-token cache -> beta~1). Deeper draft tokens predict from UNVERIFIED
draft tokens whose experts aren't known, so the resident cache must already cover them.
Measures beta_j = overlap(verify, verify-KV draft) at draft depth j along the greedy
spine, for cache policies fixed for the cycle (built from committed usage only).

Policies: static (prompt top-32), last-token (committed[-1]'s experts, C=8), EMA C=16/32,
oracle (the prediction-shaping token's FULL experts -- realizable only at j=0, upper bound).
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
]
PRE = 8       # committed greedy tokens before the cycle
K = 6         # draft depth
DECAY = 0.9


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
def local_routing(gates, cache_sets, E, dev):
    saved = []
    try:
        for g, cs in zip(gates, cache_sets):
            mk = torch.zeros(E, dtype=torch.bool, device=dev); mk[list(cs)] = True
            saved.append((g, g.forward)); g.forward = local_forward(g, mk)
        yield
    finally:
        for g, o in saved:
            g.forward = o


def overlap(lp, lq):
    p = F.softmax(lp.float(), dim=-1); q = F.softmax(lq.float(), dim=-1)
    return float(torch.minimum(p, q).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--local-files-only", action="store_true")
    a = ap.parse_args()
    dev = a.device

    cfg = AutoConfig.from_pretrained(a.model, local_files_only=a.local_files_only)
    E = cfg.num_experts; TOPK = cfg.num_experts_per_tok
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=a.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        local_files_only=a.local_files_only).to(dev).eval()
    gates = [m for m in model.modules() if type(m).__name__ == "Qwen3MoeTopKRouter"]
    Lg = len(gates)
    pols = ["static32", "lasttok8", "ema16", "ema32", "oracle"]
    beta = {p: [[] for _ in range(K)] for p in pols}
    print(f"[cfg] E={E} top_k={TOPK} layers={Lg} PRE={PRE} K={K}")

    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids.to(dev)
        L0 = ids.shape[1]
        # greedy: PRE committed + K spine
        seq = ids
        for _ in range(PRE + K):
            with torch.inference_mode():
                t = int(model(input_ids=seq, use_cache=False).logits[0, -1].argmax())
            seq = torch.cat([seq, torch.tensor([[t]], device=dev)], 1)
        Lc = L0 + PRE                       # committed length
        # full forward over committed+spine -> verify logits + per-layer usage everywhere
        with torch.inference_mode():
            out = model(input_ids=seq, use_cache=False, output_router_logits=True)
        vlog = out.logits[0]
        used = [torch.topk(rl, TOPK, dim=-1).indices for rl in out.router_logits]  # [pos,TOPK]
        prompt_top32 = [torch.topk(F.softmax(rl[:L0].float(), -1).sum(0), 32).indices.tolist()
                        for rl in out.router_logits]
        ema = [np.zeros(E) for _ in range(Lg)]
        for s in range(Lc):                 # EMA over committed only
            for l in range(Lg):
                ema[l] *= DECAY
                for e in used[l][s].tolist():
                    ema[l][e] += 1.0

        def caches(pol, j):
            if pol == "static32":
                return [set(prompt_top32[l]) for l in range(Lg)]
            if pol == "lasttok8":
                return [set(used[l][Lc - 1][:8].tolist()) for l in range(Lg)]
            if pol == "ema16":
                return [set(np.argsort(-ema[l])[:16].tolist()) for l in range(Lg)]
            if pol == "ema32":
                return [set(np.argsort(-ema[l])[:32].tolist()) for l in range(Lg)]
            if pol == "oracle":             # prediction-shaping token's FULL experts
                return [set(used[l][Lc + j - 1].tolist()) for l in range(Lg)]

        for j in range(K):
            vdist = vlog[Lc + j - 1]        # verify next-dist at this depth
            # active tokens processed locally: [committed[-1]] + spine[:j]
            active = seq[:, Lc - 1: Lc + j]  # positions Lc-1 .. Lc+j-1
            for pol in pols:
                cs = caches(pol, j)
                with torch.inference_mode():
                    past = model(input_ids=seq[:, : Lc - 1], use_cache=True).past_key_values
                with local_routing(gates, cs, E, dev), torch.inference_mode():
                    dlog = model(input_ids=active, past_key_values=past,
                                 use_cache=True).logits[0, -1]
                beta[pol][j].append(overlap(vdist, dlog))

    print(f"\ndepth {'static32':>9} {'lasttok8':>9} {'ema16':>7} {'ema32':>7} {'oracle':>7}")
    res = {p: [] for p in pols}
    for j in range(K):
        row = []
        for pol in pols:
            m = float(np.mean(beta[pol][j])); res[pol].append(round(m, 3)); row.append(m)
        print(f"  d{j}  {row[0]:>9.3f} {row[1]:>9.3f} {row[2]:>7.3f} {row[3]:>7.3f} {row[4]:>7.3f}")
    a.output_json.parent.mkdir(parents=True, exist_ok=True)
    a.output_json.write_text(json.dumps({"PRE": PRE, "K": K, "results": res}, indent=2))
    print(f"[saved] {a.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
