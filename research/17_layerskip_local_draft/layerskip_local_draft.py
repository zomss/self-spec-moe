#!/usr/bin/env python3
"""Layer-skip + local-routing draft: acceptance vs cost tradeoff.

The draft skips a band of middle decoder layers (identity passthrough) AND routes
the surviving MoE layers to a local top-M expert set. Both make the draft cheaper
(skip -> compute+memory; local -> expert weight read) but lower acceptance.

Cost (from Phase 16 pinned numbers, phi_moe~0.7, r_moe(M) measured):
  T_draft/T_full = ((L-S)/L) * ((1-phi_moe) + r_moe(M)*phi_moe)
Acceptance is measured directly (one-step proxy). Throughput gain combines them.
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
    "Summarize why distributed inference needs communication.",
    "Describe how a mixture-of-experts layer routes tokens.",
    "Hi! Can you recommend a good book for a long flight?",
    "I'm feeling stressed about work. Any advice?",
    "What's a fun fact about the ocean?",
    "Write a Python function to compute the nth Fibonacci number.",
    "Implement binary search over a sorted list in Python.",
    "Explain what a race condition is and how to avoid it.",
    "Solve step by step: if x + 3 = 7, what is x?",
    "What is the derivative of x^2 + 3x with respect to x?",
    "A train travels 60 km in 45 minutes. What is its speed in km/h?",
    "Explain the theory of relativity in one paragraph.",
    "List the steps to bake sourdough bread.",
    "Compare TCP and UDP for video streaming.",
    "Write a haiku about autumn leaves.",
]

PHI_MOE = 0.70  # phase 16, pinned
# r_moe(M/E): MoE-FFN time ratio vs full, from phase 16 kernel sweep (B~256)
R_MOE = {1.0: 1.0, 0.75: 0.77, 0.5: 0.55, 0.25: 0.50}


def parse_int_list(raw):
    return [int(v) for v in raw.split(",") if v.strip()]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--skip-counts", type=parse_int_list, default=[0, 6, 12, 16, 24])
    p.add_argument("--local-fracs", default="1.0,0.5")  # M/E values
    p.add_argument("--num-spec", type=parse_int_list, default=[2, 4, 8])
    p.add_argument("--sample-count", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-json", type=Path, required=True)
    return p.parse_args()


def restrict_logits(rl, allowed, k):
    mn = torch.finfo(rl.dtype).min
    m = rl.masked_fill(~allowed, mn)
    n = int(allowed.sum())
    if k < n:
        thr = m.topk(k, dim=-1).values[..., -1:]
        m = m.masked_fill(m < thr, mn)
    return m


def local_router_forward(module, allowed, draft_top_k):
    def forward(self, hidden_states):
        rl = F.linear(hidden_states, self.weight)
        m = restrict_logits(rl, allowed, draft_top_k)
        probs = F.softmax(m, dim=-1, dtype=torch.float)
        v, i = torch.topk(probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            v = v / v.sum(dim=-1, keepdim=True)
        return rl, v.to(rl.dtype), i

    return types.MethodType(forward, module)


def identity_layer_forward(module):
    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states

    return types.MethodType(forward, module)


@contextmanager
def patched(layers, skip_set, masks, draft_top_k):
    saved = []
    try:
        for i, layer in enumerate(layers):
            if i in skip_set:
                saved.append((layer, layer.forward))
                layer.forward = identity_layer_forward(layer)
            elif hasattr(layer.mlp, "gate") and masks is not None:
                gate = layer.mlp.gate
                saved.append((gate, gate.forward))
                gate.forward = local_router_forward(gate, masks[i], draft_top_k)
        yield
    finally:
        for mod, orig in saved:
            mod.forward = orig


def middle_skip_set(L, S):
    if S <= 0:
        return set()
    start = (L - S) // 2
    return set(range(start, start + S))


def acceptance(p_logits, q_logits, n, gen):
    p = torch.softmax(p_logits, -1)
    q = torch.softmax(q_logits, -1)
    overlap = float(torch.minimum(p, q).sum())
    d = torch.multinomial(q, n, replacement=True, generator=gen)
    acc = torch.minimum(torch.ones_like(p[d]), p[d] / q[d].clamp_min(1e-30))
    draws = torch.rand(n, generator=gen)
    return overlap, float((draws < acc).float().mean()), float(p.argmax() == q.argmax())


def e_tokens(beta, k):
    return sum(beta**i for i in range(k + 1))


def main():
    a = parse_args()
    torch.manual_seed(a.seed)
    cfg = AutoConfig.from_pretrained(a.model)
    E = getattr(cfg, "num_experts")
    top_k = getattr(cfg, "num_experts_per_tok")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(a.device).eval()
    layers = model.model.layers
    L = len(layers)
    print(f"[cfg] E={E} top_k={top_k} L={L}")

    # full-routing target dist + per-layer mass for the local mask
    full_p, layer_mass = [], None
    for prompt in PROMPTS:
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=64).to(a.device)
        with torch.inference_mode():
            out = model(**enc, use_cache=False, logits_to_keep=1, output_router_logits=True)
        full_p.append(out.logits[:, -1, :].float().squeeze(0).cpu())
        probs = torch.stack([F.softmax(rl.float(), -1).sum(0) for rl in out.router_logits])  # [L,E]
        layer_mass = probs.cpu() if layer_mass is None else layer_mass + probs.cpu()

    def masks_for(M):
        if M >= E:
            return None  # full routing
        ms = []
        for layer in range(L):
            top = torch.topk(layer_mass[layer], M).indices
            mk = torch.zeros(E, dtype=torch.bool, device=a.device)
            mk[top] = True
            ms.append(mk)
        return ms

    fracs = [float(x) for x in a.local_fracs.split(",")]
    rows = []
    gen = torch.Generator().manual_seed(a.seed + 3)
    for frac in fracs:
        M = int(round(frac * E))
        masks = masks_for(M)
        r_moe = R_MOE.get(round(frac, 2), 0.5 if frac < 0.5 else frac)
        for S in a.skip_counts:
            skip = middle_skip_set(L, S)
            accs = []
            with patched(layers, skip, masks, top_k):
                for prompt, p in zip(PROMPTS, full_p):
                    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=64).to(a.device)
                    with torch.inference_mode():
                        q = model(**enc, use_cache=False, logits_to_keep=1).logits[:, -1, :].float().squeeze(0).cpu()
                    accs.append(acceptance(p, q, a.sample_count, gen))
            sampled = float(np.mean([x[1] for x in accs]))
            top1 = float(np.mean([x[2] for x in accs]))
            t_ratio = ((L - S) / L) * ((1 - PHI_MOE) + r_moe * PHI_MOE)
            row = {
                "local_frac": frac, "M": M, "skip": S, "skip_frac": round(S / L, 3),
                "sampled_acc": round(sampled, 4), "top1": round(top1, 4),
                "T_draft_over_T_full": round(t_ratio, 4),
            }
            # throughput gain vs k (verify creep ~1.15x at memory-bound)
            for k in a.num_spec:
                g = e_tokens(sampled, k) / (k * t_ratio + 1.15)
                row[f"gain_k{k}"] = round(g, 3)
            rows.append(row)
            gains = "  ".join(f"k{k}={row[f'gain_k{k}']}" for k in a.num_spec)
            print(f"frac={frac} M={M} skip={S}({row['skip_frac']}): acc={sampled:.3f} "
                  f"top1={top1:.3f} Tdraft/Tfull={t_ratio:.3f} | {gains}")

    a.output_json.parent.mkdir(parents=True, exist_ok=True)
    a.output_json.write_text(json.dumps({"model": a.model, "E": E, "L": L, "phi_moe": PHI_MOE, "rows": rows}, indent=2))
    # best gain
    best = max(rows, key=lambda r: max(r[f"gain_k{k}"] for k in a.num_spec))
    print(f"\nBEST: frac={best['local_frac']} skip={best['skip']} acc={best['sampled_acc']} "
          f"-> max gain {max(best[f'gain_k{k}'] for k in a.num_spec)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
