#!/usr/bin/env python3
"""Lossless-scheme acceptance: FP8(+local) draft vs the bf16 target.

For a lossless method the verify is bf16; the draft can be FP8 to be cheaper. This
measures whether an FP8 (per-tensor fake-quant) draft -- optionally with local
routing -- keeps acceptance against the exact bf16 full-routing target.

Configs (acceptance vs bf16-full target p):
  bf16 + local        (reference: local error only, ~Phase 13)
  FP8  + full         (pure FP8 rounding error, no local)
  FP8  + local        (combined)
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


@contextmanager
def local_routing(layers, masks, draft_top_k):
    saved = []
    try:
        for i, layer in enumerate(layers):
            if hasattr(layer.mlp, "gate") and masks is not None:
                g = layer.mlp.gate
                saved.append((g, g.forward))
                g.forward = local_router_forward(g, masks[i], draft_top_k)
        yield
    finally:
        for g, orig in saved:
            g.forward = orig


def fake_fp8(w):
    """Per-tensor E4M3 fake quantization (round to FP8 grid, back to bf16)."""
    amax = w.abs().amax().clamp_min(1e-8)
    scale = amax / 448.0
    q = torch.clamp(w / scale, -448.0, 448.0).to(torch.float8_e4m3fn).to(w.dtype)
    return q * scale


def accept(p_logits, q_logits, n, gen):
    p = torch.softmax(p_logits, -1)
    q = torch.softmax(q_logits, -1)
    overlap = float(torch.minimum(p, q).sum())
    d = torch.multinomial(q, n, replacement=True, generator=gen)
    acc = torch.minimum(torch.ones_like(p[d]), p[d] / q[d].clamp_min(1e-30))
    draws = torch.rand(n, generator=gen)
    return overlap, float((draws < acc).float().mean()), float(p.argmax() == q.argmax())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--local-frac", type=float, default=0.5)
    ap.add_argument("--sample-count", type=int, default=512)
    ap.add_argument("--output-json", type=Path, required=True)
    a = ap.parse_args()

    cfg = AutoConfig.from_pretrained(a.model)
    E = cfg.num_experts
    top_k = cfg.num_experts_per_tok
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(a.device).eval()
    layers = model.model.layers
    L = len(layers)
    M = int(round(a.local_frac * E))
    print(f"[cfg] E={E} top_k={top_k} L={L} M={M}")

    # bf16 target p + per-layer mass for local mask
    full_p, layer_mass = [], None
    for prompt in PROMPTS:
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=64).to(a.device)
        with torch.inference_mode():
            out = model(**enc, use_cache=False, logits_to_keep=1, output_router_logits=True)
        full_p.append(out.logits[:, -1, :].float().squeeze(0).cpu())
        probs = torch.stack([F.softmax(rl.float(), -1).sum(0) for rl in out.router_logits])
        layer_mass = probs.cpu() if layer_mass is None else layer_mass + probs.cpu()
    masks = []
    for layer in range(L):
        top = torch.topk(layer_mass[layer], M).indices
        mk = torch.zeros(E, dtype=torch.bool, device=a.device)
        mk[top] = True
        masks.append(mk)

    gen = torch.Generator().manual_seed(0)

    def measure(use_local):
        accs = []
        ctx = local_routing(layers, masks, top_k) if use_local else _null()
        with ctx:
            for prompt, p in zip(PROMPTS, full_p):
                enc = tok(prompt, return_tensors="pt", truncation=True, max_length=64).to(a.device)
                with torch.inference_mode():
                    q = model(**enc, use_cache=False, logits_to_keep=1).logits[:, -1, :].float().squeeze(0).cpu()
                accs.append(accept(p, q, a.sample_count, gen))
        return float(np.mean([x[1] for x in accs])), float(np.mean([x[2] for x in accs]))

    @contextmanager
    def _null():
        yield

    results = {}
    # bf16 + local (reference)
    s, t1 = measure(True)
    results["bf16_local"] = {"sampled": round(s, 4), "top1": round(t1, 4)}
    print(f"bf16+local: acc={s:.4f} top1={t1:.4f}")

    # fake-quant experts to FP8 (in place), then FP8 configs
    nq = 0
    for layer in layers:
        if hasattr(layer.mlp, "experts"):
            for p in layer.mlp.experts.parameters():
                if p.dim() >= 2:  # weight tensors only
                    p.data = fake_fp8(p.data)
                    nq += 1
    print(f"[fp8] fake-quantized {nq} expert weight tensors to E4M3")

    s, t1 = measure(False)
    results["fp8_full"] = {"sampled": round(s, 4), "top1": round(t1, 4)}
    print(f"FP8+full : acc={s:.4f} top1={t1:.4f}  (pure FP8 error)")
    s, t1 = measure(True)
    results["fp8_local"] = {"sampled": round(s, 4), "top1": round(t1, 4)}
    print(f"FP8+local: acc={s:.4f} top1={t1:.4f}  (combined)")

    a.output_json.parent.mkdir(parents=True, exist_ok=True)
    a.output_json.write_text(json.dumps({"model": a.model, "E": E, "M": M, "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
