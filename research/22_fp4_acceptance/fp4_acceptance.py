#!/usr/bin/env python3
"""FP4 expert-draft acceptance vs the bf16 target (Phase 18 method, FP4 schemes).

Phase 18 measured a per-tensor FP8 expert draft -> sampled acceptance 0.954 vs bf16.
The per-node-replication direction (Phase 21 / intra-node EP draft) only closes the
memory math at FP4, so the decisive question is whether an FP4-expert draft still
accepts. This fake-quantizes (quant->dequant) the expert weights weight-only (W4A16;
attention/router/embeddings stay bf16) under several FP4 schemes and measures
rejection-sampling acceptance against the exact bf16 full-routing target.

Schemes:
  fp8_pertensor : per-tensor E4M3 (Phase 18 sanity anchor; expect ~0.954)
  mxfp4_g32     : FP4 E2M1, block 32, power-of-2 (E8M0) shared scale  (MXFP4)
  nvfp4_g16     : FP4 E2M1, block 16, E4M3 (FP8) block scale          (NVFP4)
  int4_g128     : symmetric INT4, group 128 (AWQ/GPTQ-style baseline)

Acceptance is evaluated at many teacher-forced positions per prompt for statistics.
"""

from __future__ import annotations

import argparse
import json
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

# E2M1 representable magnitudes (FP4); max = 6.0
FP4_MAGS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def round_to_fp4(ax, grid, mids):
    idx = torch.bucketize(ax, mids)
    return grid[idx]


def fp4_qdq(w, group_size, scale_mode, grid, mids):
    last = w.shape[-1]
    g = group_size if last % group_size == 0 else last
    wf = w.float().reshape(-1, g)
    absmax = wf.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    scale = absmax / 6.0
    if scale_mode == "pow2":  # MXFP4: E8M0 power-of-2 scale
        scale = torch.pow(2.0, torch.round(torch.log2(scale)))
    elif scale_mode == "e4m3":  # NVFP4: FP8 block scale
        scale = scale.to(torch.float8_e4m3fn).float().clamp_min(1e-8)
    sign = torch.sign(wf)
    ax = (wf / scale).abs().clamp(max=6.0)
    q = sign * round_to_fp4(ax, grid, mids)
    return (q * scale).reshape(w.shape).to(w.dtype)


def int4_qdq(w, group_size):
    last = w.shape[-1]
    g = group_size if last % group_size == 0 else last
    wf = w.float().reshape(-1, g)
    absmax = wf.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    scale = absmax / 7.0
    q = torch.clamp(torch.round(wf / scale), -8.0, 7.0)
    return (q * scale).reshape(w.shape).to(w.dtype)


def fp8_pertensor(w):
    amax = w.abs().amax().clamp_min(1e-8)
    scale = amax / 448.0
    q = torch.clamp(w.float() / scale, -448.0, 448.0).to(torch.float8_e4m3fn).float()
    return (q * scale).to(w.dtype)


def quantize_scheme(w, scheme, grid, mids):
    if scheme == "fp8_pertensor":
        return fp8_pertensor(w)
    if scheme == "mxfp4_g32":
        return fp4_qdq(w, 32, "pow2", grid, mids)
    if scheme == "nvfp4_g16":
        return fp4_qdq(w, 16, "e4m3", grid, mids)
    if scheme == "int4_g128":
        return int4_qdq(w, 128)
    raise ValueError(scheme)


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
    ap.add_argument("--sample-count", type=int, default=512)
    ap.add_argument("--pos-from", type=int, default=4, help="eval positions >= this")
    ap.add_argument("--schemes", default="fp8_pertensor,mxfp4_g32,nvfp4_g16,int4_g128")
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--local-files-only", action="store_true")
    a = ap.parse_args()

    cfg = AutoConfig.from_pretrained(a.model, local_files_only=a.local_files_only)
    E, top_k = cfg.num_experts, cfg.num_experts_per_tok
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=a.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        local_files_only=a.local_files_only,
    ).to(a.device).eval()
    layers = model.model.layers
    print(f"[cfg] {a.model} E={E} top_k={top_k} layers={len(layers)}")

    grid = torch.tensor(FP4_MAGS, device=a.device)
    mids = (grid[1:] + grid[:-1]) / 2.0

    # enumerate expert weight tensors (dim>=2), snapshot originals to CPU
    expert_w = []
    for layer in layers:
        if hasattr(layer.mlp, "experts"):
            for p in layer.mlp.experts.parameters():
                if p.dim() >= 2:
                    expert_w.append(p)
    orig_cpu = [p.data.detach().to("cpu", copy=True) for p in expert_w]
    n_params = sum(p.numel() for p in expert_w)
    print(f"[snap] {len(expert_w)} expert weight tensors, {n_params/1e9:.2f}B params")

    # bf16 target p: all eval positions per prompt (kept on CPU float)
    enc_cache, target = [], []
    for prompt in PROMPTS:
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=64).to(a.device)
        enc_cache.append(enc)
        with torch.inference_mode():
            lg = model(**enc, use_cache=False).logits[0].float()  # [T, V]
        T = lg.shape[0]
        pos = list(range(a.pos_from, T))
        target.append((pos, lg[pos].cpu()))
    n_pos = sum(len(p) for p, _ in target)
    print(f"[target] bf16 logits cached at {n_pos} positions")

    gen = torch.Generator().manual_seed(0)

    def measure_current():
        ov, sa, t1 = [], [], []
        for enc, (pos, p_cpu) in zip(enc_cache, target):
            with torch.inference_mode():
                q_lg = model(**enc, use_cache=False).logits[0].float()
            q_lg = q_lg.cpu()
            for j, t in enumerate(pos):
                o, s, m = accept(p_cpu[j], q_lg[t], a.sample_count, gen)
                ov.append(o); sa.append(s); t1.append(m)
        return (round(float(np.mean(ov)), 4), round(float(np.mean(sa)), 4),
                round(float(np.mean(t1)), 4))

    results = {}
    for scheme in a.schemes.split(","):
        scheme = scheme.strip()
        # restore originals, then fake-quant in place
        with torch.inference_mode():
            for p, o in zip(expert_w, orig_cpu):
                p.data.copy_(o.to(a.device))
                p.data = quantize_scheme(p.data, scheme, grid, mids)
        ov, sa, t1 = measure_current()
        results[scheme] = {"expected": ov, "sampled": sa, "top1": t1}
        print(f"{scheme:>14}: sampled_acc={sa:.4f}  expected={ov:.4f}  top1={t1:.4f}")

    a.output_json.parent.mkdir(parents=True, exist_ok=True)
    a.output_json.write_text(json.dumps(
        {"model": a.model, "E": E, "top_k": top_k, "n_positions": n_pos,
         "sample_count": a.sample_count, "results": results}, indent=2))
    print(f"[saved] {a.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
