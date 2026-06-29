#!/usr/bin/env python3
"""Batched memory: the per-device resident set must cover the UNION of concurrent requests.

Experts are SHARED weights per device, not per request. So with B requests on a device the
resident cache must cover the union of their hot experts. This measures:
  (1) union(B) = distinct experts needed by B random requests (per layer) vs B.
  (2) global-hot coverage: a FIXED resident top-C (batch-independent) -- what fraction of a
      request's top_k experts it covers (= skip-cold beta proxy; misses are corrected by
      verify, lossless).
Each position in real generations is treated as an independent request snapshot.
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
    "Write a Python function to compute the nth Fibonacci number.",
    "Explain mixture of experts models in simple terms.",
    "Solve step by step: if x + 3 = 7, what is x?",
    "Describe how a transformer attention layer works.",
    "List three benefits of expert parallelism.",
    "Summarize the causes of the French Revolution.",
    "What is the difference between TCP and UDP?",
]
GEN = 40
BS = [1, 2, 4, 8, 16, 32, 64, 128]
CS = [8, 16, 32, 64]


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

    # collect per-layer top_k experts at every generated position (= request snapshots)
    used = None  # used[l] -> [n_pos, TOPK]
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids.to(dev)
        seq = ids
        for _ in range(GEN):
            with torch.inference_mode():
                t = int(model(input_ids=seq, use_cache=False).logits[0, -1].argmax())
            seq = torch.cat([seq, torch.tensor([[t]], device=dev)], 1)
        with torch.inference_mode():
            out = model(input_ids=seq, use_cache=False, output_router_logits=True)
        per = [torch.topk(rl, TOPK, -1).indices.cpu().numpy() for rl in out.router_logits]
        if used is None:
            used = [[] for _ in per]
        for l, u in enumerate(per):
            used[l].append(u)
    used = [np.concatenate(u, 0) for u in used]   # [n_pos, TOPK] per layer
    Lg = len(used); npos = used[0].shape[0]
    print(f"[cfg] E={E} top_k={TOPK} layers={Lg} request-snapshots={npos}")

    rng = np.random.RandomState(0)
    # (1) union(B)
    union = {}
    for B in BS:
        if B > npos:
            continue
        vals = []
        for _ in range(40):
            for l in range(0, Lg, 6):  # subsample layers for speed
                idx = rng.choice(npos, B, replace=False)
                vals.append(len(np.unique(used[l][idx])))
        union[B] = float(np.mean(vals))
    # (2) global-hot top-C coverage (fraction of a request's top_k in global top-C)
    cov = {}
    for C in CS:
        fr = []
        for l in range(Lg):
            freq = np.bincount(used[l].reshape(-1), minlength=E)
            hot = set(np.argsort(-freq)[:C].tolist())
            for r in used[l]:
                fr.append(len(set(r.tolist()) & hot) / TOPK)
        cov[C] = float(np.mean(fr))

    print(f"\nunion(B) = distinct experts needed by B requests (per layer, E={E}):")
    for B in union:
        print(f"  B={B:>4}: {union[B]:>6.1f} experts ({100*union[B]/E:.0f}% of E)")
    print(f"\nglobal-hot top-C coverage (frac of a request's top_k resident; skip-cold beta proxy):")
    for C in CS:
        print(f"  C={C:>3} ({100*C/E:.0f}% of E, {C/E*16.3:.1f} GB FP4): coverage {cov[C]:.3f}")

    a.output_json.parent.mkdir(parents=True, exist_ok=True)
    a.output_json.write_text(json.dumps(
        {"E": E, "top_k": TOPK, "n_pos": npos,
         "union": {str(k): round(v, 1) for k, v in union.items()},
         "coverage": {str(k): round(v, 3) for k, v in cov.items()}}, indent=2))
    print(f"[saved] {a.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
