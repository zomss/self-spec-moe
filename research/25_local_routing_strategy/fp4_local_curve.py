#!/usr/bin/env python3
"""Phase 25 Exp 2: local-routing acceptance vs cache size C, bf16 vs NVFP4 weights.

Maps how local-routing draft acceptance depends on the local cache size C (per-layer
top-C routed experts by request gate mass), with and without NVFP4 weight quant, vs
the exact bf16 full-routing target. Confirms local routing interpolates from
coverage-limited (small C) to the quant-limited ceiling (C=E -> NVFP4 weight-only
~0.92, Phase 22; bf16 -> ~1.0). Qwen3-30B-A3B (E=128, top_k 8).
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
FP4_MAGS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
ROUTER = {"Qwen3MoeTopKRouter", "Qwen2MoeTopKRouter"}


def fp4_qdq(w, group_size, grid, mids):
    last = w.shape[-1]
    g = group_size if last % group_size == 0 else last
    wf = w.float().reshape(-1, g)
    absmax = wf.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    scale = (absmax / 6.0).to(torch.float8_e4m3fn).float().clamp_min(1e-8)  # NVFP4 e4m3
    sign = torch.sign(wf)
    ax = (wf / scale).abs().clamp(max=6.0)
    q = sign * grid[torch.bucketize(ax, mids)]
    return (q * scale).reshape(w.shape).to(w.dtype)


def local_forward(module, allowed):
    """Match the real router (softmax over ALL, top-k, optional norm) but restrict
    the SELECTION to the allowed local set. At allowed=all -> original router."""
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


def accept(p_logits, q_logits, n, gen):
    p = torch.softmax(p_logits, -1)
    q = torch.softmax(q_logits, -1)
    overlap = float(torch.minimum(p, q).sum())
    d = torch.multinomial(q, n, replacement=True, generator=gen)
    acc = torch.minimum(torch.ones_like(p[d]), p[d] / q[d].clamp_min(1e-30))
    return overlap, float((torch.rand(n, generator=gen) < acc).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sample-count", type=int, default=512)
    ap.add_argument("--pos-from", type=int, default=4)
    ap.add_argument("--cache-fracs", default="0.0625,0.125,0.25,0.5,0.75,1.0")
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--local-files-only", action="store_true")
    a = ap.parse_args()
    dev = a.device

    cfg = AutoConfig.from_pretrained(a.model, local_files_only=a.local_files_only)
    E, top_k = cfg.num_experts, cfg.num_experts_per_tok
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=a.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        local_files_only=a.local_files_only,
    ).to(dev).eval()
    gates = [m for m in model.modules() if type(m).__name__ in ROUTER]
    print(f"[cfg] {a.model} E={E} top_k={top_k} layers={len(gates)}")
    grid = torch.tensor(FP4_MAGS, device=dev)
    mids = (grid[1:] + grid[:-1]) / 2.0

    # bf16 full-routing target p + per-layer top-C masks (by request gate mass)
    enc_cache, target, mass_acc = [], [], None
    for prompt in PROMPTS:
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=64).to(dev)
        enc_cache.append(enc)
        with torch.inference_mode():
            out = model(**enc, use_cache=False, output_router_logits=True)
        lg = out.logits[0].float()
        T = lg.shape[0]
        pos = list(range(a.pos_from, T))
        target.append((pos, lg[pos].cpu()))
        m = torch.stack([F.softmax(rl.float(), -1).sum(0) for rl in out.router_logits])
        mass_acc = m.cpu() if mass_acc is None else mass_acc + m.cpu()
    fracs = [float(x) for x in a.cache_fracs.split(",")]
    gen = torch.Generator().manual_seed(0)

    def masks_for(C):
        out = []
        for layer in range(len(gates)):
            mk = torch.zeros(E, dtype=torch.bool, device=dev)
            mk[torch.topk(mass_acc[layer], C).indices] = True
            out.append(mk)
        return out

    def measure(masks):
        ov, sa = [], []
        ctx = local_routing(gates, masks) if masks else _null()
        with ctx:
            for enc, (pos, p_cpu) in zip(enc_cache, target):
                with torch.inference_mode():
                    q = model(**enc, use_cache=False).logits[0].float().cpu()
                for j, t in enumerate(pos):
                    o, s = accept(p_cpu[j], q[t], a.sample_count, gen)
                    ov.append(o); sa.append(s)
        return round(float(np.mean(ov)), 4), round(float(np.mean(sa)), 4)

    @contextmanager
    def _null():
        yield

    results = {"bf16": [], "nvfp4": []}
    # bf16 mode: original weights, sweep C
    for frac in fracs:
        C = max(1, int(round(frac * E)))
        ov, sa = measure(masks_for(C))
        results["bf16"].append({"frac": frac, "C": C, "sampled": sa, "expected": ov})
        print(f"[bf16] C={C} (f={frac}): sampled={sa}")
    # nvfp4 mode: quantize experts once, sweep C
    nq = 0
    for layer in model.model.layers:
        if hasattr(layer.mlp, "experts"):
            for p in layer.mlp.experts.parameters():
                if p.dim() >= 2:
                    p.data = fp4_qdq(p.data, 16, grid, mids); nq += 1
    print(f"[nvfp4] quantized {nq} expert tensors")
    for frac in fracs:
        C = max(1, int(round(frac * E)))
        ov, sa = measure(masks_for(C))
        results["nvfp4"].append({"frac": frac, "C": C, "sampled": sa, "expected": ov})
        print(f"[nvfp4] C={C} (f={frac}): sampled={sa}")

    a.output_json.parent.mkdir(parents=True, exist_ok=True)
    a.output_json.write_text(json.dumps(
        {"model": a.model, "E": E, "top_k": top_k, "results": results}, indent=2))
    print(f"[saved] {a.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
