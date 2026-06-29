#!/usr/bin/env python3
"""Cross-model check: routing skew (union growth + global-hot coverage) generalizes?

Validates the Phase 28 batched-memory enablers on other architectures:
  - union(B): distinct routed experts B requests need (per layer).
  - global-hot top-C coverage: fixed batch-independent cache coverage (skip-cold beta proxy).
For shared-expert models the shared experts are ALWAYS resident (free) -- noted separately.
Routed-expert selection captured via forward hooks on the gate/router modules (works
across architectures that don't expose output_router_logits).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
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
GEN = 48
BS = [1, 2, 4, 8, 16, 32, 64, 128]


def cfg_get(c, names, default=None):
    for n in names:
        if hasattr(c, n) and getattr(c, n) is not None:
            return getattr(c, n)
    return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    a = ap.parse_args()
    dev = "cuda:0"

    c = AutoConfig.from_pretrained(a.model, local_files_only=True)
    E = cfg_get(c, ["num_experts", "n_routed_experts", "num_local_experts"])
    TOPK = cfg_get(c, ["num_experts_per_tok", "experts_per_token"])
    SHARED = cfg_get(c, ["n_shared_experts"], 0) or 0
    CS = [max(1, round(f * E)) for f in (0.0625, 0.125, 0.25, 0.5)]
    print(f"[cfg] {a.model} [{c.model_type}] E={E} top_k={TOPK} shared={SHARED}")

    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    kw = dict(dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True)
    if c.model_type == "gpt_oss":  # dequantize MXFP4 -> bf16
        from transformers import Mxfp4Config
        kw["quantization_config"] = Mxfp4Config(dequantize=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw).to(dev).eval()

    store = {}
    hooks = []

    # Path A: MoE block has a `.gate` Linear (out=E) called functionally (DeepSeek) ->
    # pre-hook the block, recompute top_k from its input + gate weight (greedy softmax
    # top_k == top_k of logits). Path B: router IS a module (Qwen-like) -> forward-hook it.
    moe = [(n, m, getattr(m, "gate", None)) for n, m in model.named_modules()
           if isinstance(getattr(m, "gate", None), torch.nn.Linear)
           and getattr(m, "gate").out_features == E]

    if moe:
        print(f"  extraction: gate-weight recompute on {len(moe)} MoE blocks")

        def mkpre(name, gate):
            def pre(mod, args):
                h = args[0]
                lg = torch.nn.functional.linear(
                    h.reshape(-1, h.shape[-1]).float(), gate.weight.float())
                store[name] = torch.topk(lg, TOPK, -1).indices.cpu().numpy()
            return pre
        for n, m, g in moe:
            hooks.append(m.register_forward_pre_hook(mkpre(n, g)))
    else:
        print("  extraction: forward-hook on gate/router modules")

        def mk(name):
            def f(mod, inp, out):
                outs = out if isinstance(out, (tuple, list)) else (out,)
                for o in outs:
                    if torch.is_tensor(o) and not o.dtype.is_floating_point and o.shape[-1] == TOPK:
                        store[name] = o.detach().reshape(-1, TOPK).cpu().numpy(); return
                for o in outs:
                    if torch.is_tensor(o) and o.dtype.is_floating_point and o.dim() >= 2 and o.shape[-1] == E:
                        store[name] = torch.topk(o.detach().reshape(-1, E), TOPK, -1).indices.cpu().numpy(); return
            return f
        for name, mod in model.named_modules():
            last = name.split(".")[-1].lower()
            if last == "gate" or "router" in last:
                hooks.append(mod.register_forward_hook(mk(name)))

    used = None; order = None
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids.to(dev)
        with torch.inference_mode():
            seq = model.generate(ids, max_new_tokens=GEN, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        store.clear()
        with torch.inference_mode():
            model(input_ids=seq, use_cache=False)
        if order is None:
            order = [n for n in store]; used = [[] for _ in order]
        for l, n in enumerate(order):
            if n in store:
                used[l].append(store[n])
    used = [np.concatenate(u, 0) for u in used if u]
    Lg = len(used); npos = used[0].shape[0]
    print(f"  hooked gates={len(order)} MoE layers w/ data={Lg} snapshots={npos}")

    rng = np.random.RandomState(0)
    union = {}
    for B in BS:
        if B > npos:
            continue
        union[B] = float(np.mean([len(np.unique(used[l][rng.choice(npos, B, replace=False)]))
                                  for _ in range(40) for l in range(0, Lg, max(1, Lg // 8))]))
    cov = {}
    for C in CS:
        fr = []
        for l in range(Lg):
            freq = np.bincount(used[l].reshape(-1), minlength=E)
            hot = set(np.argsort(-freq)[:C].tolist())
            fr += [len(set(r.tolist()) & hot) / TOPK for r in used[l]]
        cov[C] = float(np.mean(fr))

    print(f"\nunion(B) distinct routed experts (E={E}):")
    for B in union:
        print(f"  B={B:>4}: {union[B]:>6.1f} ({100*union[B]/E:.0f}% of E)")
    print(f"\nglobal-hot top-C coverage (routed; shared {SHARED} always free):")
    for C in CS:
        print(f"  C={C:>3} ({100*C/E:.0f}% E): {cov[C]:.3f}")

    a.output_json.parent.mkdir(parents=True, exist_ok=True)
    a.output_json.write_text(json.dumps({
        "model": a.model, "type": c.model_type, "E": E, "top_k": TOPK, "shared": SHARED,
        "union": {str(k): round(v, 1) for k, v in union.items()},
        "coverage": {str(k): round(v, 3) for k, v in cov.items()}}, indent=2))
    print(f"[saved] {a.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
