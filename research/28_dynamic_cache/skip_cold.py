#!/usr/bin/env python3
"""Skip-cold beta: real acceptance with a FIXED globally-hot top-C cache (high-batch).

At serving (high) batch the resident cache must be batch-independent: the globally-hot
top-C experts. A token routing to cold (non-resident) experts drafts with its top resident
experts instead (router masked to resident -> top_k among resident -> renorm) = skip-cold.
Verify corrects any divergence (lossless, rejection-sampling theorem), so this only sets
beta. Measures beta = overlap(verify, verify-KV draft) with the global-hot cache vs C, and
the raw coverage (frac of a token's top_k that is resident) to validate the proxy.
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
    "Summarize the causes of the French Revolution.",
    "What is the difference between TCP and UDP?",
]
GEN = 36
STRIDE = 4
CS = [8, 16, 32, 64]


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
            saved.append((g, g.forward)); g.forward = local_forward(g, mk)
        yield
    finally:
        for g, o in saved:
            g.forward = o


def overlap(lp, lq):
    p = F.softmax(lp.float(), -1); q = F.softmax(lq.float(), -1)
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
    print(f"[cfg] E={E} top_k={TOPK} layers={Lg}")

    # collect global expert frequency over generated traffic
    freq = [np.zeros(E) for _ in range(Lg)]
    seqs = []
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids.to(dev)
        seq = ids
        for _ in range(GEN):
            with torch.inference_mode():
                t = int(model(input_ids=seq, use_cache=False).logits[0, -1].argmax())
            seq = torch.cat([seq, torch.tensor([[t]], device=dev)], 1)
        seqs.append((ids.shape[1], seq))
        with torch.inference_mode():
            out = model(input_ids=seq, use_cache=False, output_router_logits=True)
        for l, rl in enumerate(out.router_logits):
            for e in torch.topk(rl, TOPK, -1).indices.reshape(-1).tolist():
                freq[l][e] += 1
    ghot = {C: [set(np.argsort(-freq[l])[:C].tolist()) for l in range(Lg)] for C in CS}

    beta = {C: [] for C in CS}; cover = {C: [] for C in CS}
    for L0, seq in seqs:
        Ltot = seq.shape[1]
        with torch.inference_mode():
            out = model(input_ids=seq, use_cache=False, output_router_logits=True)
        vlog = out.logits[0]
        used = [torch.topk(rl, TOPK, -1).indices for rl in out.router_logits]
        for t in range(L0, Ltot, STRIDE):           # request snapshot at position t-1 predicts t
            ctx = seq[:, :t]
            vdist = vlog[t - 1]
            for C in CS:
                cs = ghot[C]
                with torch.inference_mode():
                    past = model(input_ids=ctx[:, :-1], use_cache=True).past_key_values
                masks = [torch.zeros(E, dtype=torch.bool, device=dev) for _ in range(Lg)]
                for l in range(Lg):
                    masks[l][list(cs[l])] = True
                with local_routing(gates, masks), torch.inference_mode():
                    dlog = model(input_ids=ctx[:, -1:], past_key_values=past,
                                 use_cache=True).logits[0, -1]
                beta[C].append(overlap(vdist, dlog))
                # coverage: frac of last committed token's top_k that is resident
                tk = set(used[l_ := 0][t - 1].tolist())  # placeholder; compute mean below
                cov = np.mean([len(set(used[l][t - 1].tolist()) & cs[l]) / TOPK
                               for l in range(Lg)])
                cover[C].append(cov)

    print(f"\n{'C':>4} {'%E':>4} {'GB':>5} {'coverage':>9} {'skip-cold beta':>14}")
    res = {}
    for C in CS:
        b = float(np.mean(beta[C])); c = float(np.mean(cover[C]))
        res[C] = {"coverage": round(c, 3), "beta": round(b, 3),
                  "gb_fp4": round(C / E * 16.3, 1)}
        print(f"{C:>4} {100*C//E:>3}% {C/E*16.3:>5.1f} {c:>9.3f} {b:>14.3f}")
    a.output_json.parent.mkdir(parents=True, exist_ok=True)
    a.output_json.write_text(json.dumps({"E": E, "top_k": TOPK, "results":
        {str(k): v for k, v in res.items()}}, indent=2))
    print(f"[saved] {a.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
