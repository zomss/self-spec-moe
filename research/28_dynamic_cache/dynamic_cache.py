#!/usr/bin/env python3
"""Memory axis: verify-warmed dynamic cache vs static top-C (with verify-KV draft).

With the verify-context-KV draft (now default) the context is full-weight, so only the
PREDICTING token's MoE is local -> the draft cache only needs that token's experts. The
predicting token is shaped by the LAST COMMITTED token's experts, which verify already
knows. So a verify-warmed cache (track recently-used experts) should hit high beta at a
much smaller C than the static prompt-top-C frontier (Phase 26).

Measures depth-0 acceptance beta = overlap(verify next-dist, draft next-dist) along a
real greedy continuation, for cache policies {static, verify-warmed (EMA), oracle (last
committed token's experts)} x cache size C. Draft = verify-KV (full context KV + local
routing for the last token only).
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
CS = [4, 8, 16, 32, 64]
DECAY = 0.9
WARMUP = 4


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
    ap.add_argument("--depth", type=int, default=20)
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
    D = a.depth
    print(f"[cfg] E={E} top_k={TOPK} layers={Lg} depth={D} policies=static/verify-warmed/oracle")

    # acc[policy][C] = list of beta
    acc = {p: {c: [] for c in CS} for p in ("static", "vwarm", "oracle")}
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids.to(dev)
        L0 = ids.shape[1]
        # greedy continuation
        ref, cur = [], ids
        for _ in range(D):
            with torch.inference_mode():
                t = int(model(input_ids=cur, use_cache=False).logits[0, -1].argmax())
            ref.append(t); cur = torch.cat([cur, torch.tensor([[t]], device=dev)], 1)
        full = cur  # prompt + ref
        # per-layer used experts at each position + per-layer prompt mass + verify logits
        with torch.inference_mode():
            out = model(input_ids=full, use_cache=False, output_router_logits=True)
        vlogits = out.logits[0]                       # [L0+D, vocab]
        used = []                                     # used[l] = [seqpos] -> set of experts
        prompt_top = []                               # static cache per layer (max C)
        for rl in out.router_logits:                  # [L0+D, E]
            topk = torch.topk(rl, TOPK, dim=-1).indices  # [L0+D, TOPK]
            used.append(topk)
            mass = F.softmax(rl[:L0].float(), dim=-1).sum(0)
            prompt_top.append(torch.topk(mass, max(CS)).indices.tolist())

        ema = [np.zeros(E) for _ in range(Lg)]
        for t in range(1, D):
            seqpos = L0 + t - 1                        # last committed token position
            # update EMA with last committed token's experts (known to verify)
            for l in range(Lg):
                ema[l] *= DECAY
                for e in used[l][seqpos].tolist():
                    ema[l][e] += 1.0
            if t < WARMUP:
                continue
            ctx = full[:, : L0 + t]                    # predict ref[t] from prompt+ref[:t]
            vlog = vlogits[seqpos]                     # verify next-token dist
            # verify context KV (full weight) over ctx[:-1]
            for C in CS:
                caches = {
                    "static": [set(prompt_top[l][:C]) for l in range(Lg)],
                    "vwarm": [set(np.argsort(-ema[l])[:C].tolist()) for l in range(Lg)],
                    "oracle": [set(used[l][seqpos][:C].tolist()) for l in range(Lg)],
                }
                for pol, cs in caches.items():
                    with torch.inference_mode():
                        past = model(input_ids=ctx[:, :-1], use_cache=True).past_key_values
                    with local_routing(gates, cs, E, dev), torch.inference_mode():
                        dlog = model(input_ids=ctx[:, -1:], past_key_values=past,
                                     use_cache=True).logits[0, -1]
                    acc[pol][C].append(overlap(vlog, dlog))

    print(f"\n{'C':>4} {'static':>8} {'vwarm':>8} {'oracle':>8}  (beta=overlap, verify-KV draft)")
    res = {}
    for C in CS:
        s = np.mean(acc['static'][C]); v = np.mean(acc['vwarm'][C]); o = np.mean(acc['oracle'][C])
        res[C] = {"static": round(float(s), 3), "vwarm": round(float(v), 3),
                  "oracle": round(float(o), 3)}
        print(f"{C:>4} {s:>8.3f} {v:>8.3f} {o:>8.3f}")
    a.output_json.parent.mkdir(parents=True, exist_ok=True)
    a.output_json.write_text(json.dumps(
        {"model": a.model, "E": E, "top_k": TOPK, "decay": DECAY, "results": res}, indent=2))
    print(f"[saved] {a.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
