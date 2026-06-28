#!/usr/bin/env python3
"""Phase 25 follow-up: shared-expert anchor on DeepSeek-V2-Lite (native impl).

Generalization of Exp 1 (Qwen1.5-MoE, 45% shared mass) to a model with a NATIVELY
smaller shared fraction: DeepSeek-V2-Lite (64 routed top_k 6 + 2 shared,
norm_topk_prob=False, first_k_dense_replace=1). Uses the native transformers
deepseek_v2 impl (trust_remote_code=False -- the bundled custom code is incompatible
with transformers 5.12). Same protocol: local-routing acceptance vs the bf16
full-routing target, sweep routed cache C, with_shared vs without_shared (ablated in
both), + measured shared mass fraction. One block patch does local routing
(restrict top-k selection), shared ablation, and mass recording.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.deepseek_v2 import modeling_deepseek_v2 as dv2

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
_ST = {"masks": None, "ablate": False, "measure": False, "record": False,
       "rec": {}, "snorm": [], "rnorm": []}


def patched_forward(self, hidden_states):
    residuals = hidden_states
    orig_shape = hidden_states.shape
    rl = F.linear(hidden_states.type(torch.float32), self.gate.weight.type(torch.float32))
    scores = rl.view(-1, rl.shape[-1]).softmax(dim=-1, dtype=torch.float32)
    idx = self._moe_idx
    if _ST["record"]:
        _ST["rec"][idx] = _ST["rec"].get(idx, 0) + scores.sum(0).detach()
    if _ST["masks"] is not None:
        scores = scores.masked_fill(~_ST["masks"][idx], 0.0)
    tw, ti = torch.topk(scores, self.top_k, dim=-1, sorted=False)
    tw = tw * self.routed_scaling_factor
    hs = hidden_states.view(-1, hidden_states.shape[-1])
    y = self.experts(hs, ti, tw).view(*orig_shape)
    sh = self.shared_experts(residuals)
    if _ST["measure"]:
        _ST["snorm"].append(float(sh.float().norm()))
        _ST["rnorm"].append(float(y.float().norm()))
    if not _ST["ablate"]:
        y = y + sh
    return y


_FP4 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def nvfp4_qdq(w, grid, mids, group=16):
    last = w.shape[-1]
    g = group if last % group == 0 else last
    wf = w.float().reshape(-1, g)
    absmax = wf.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    scale = (absmax / 6.0).to(torch.float8_e4m3fn).float().clamp_min(1e-8)
    sign = torch.sign(wf)
    ax = (wf / scale).abs().clamp(max=6.0)
    q = sign * grid[torch.bucketize(ax, mids)]
    return (q * scale).reshape(w.shape).to(w.dtype)


def quantize_experts(blocks, dev):
    grid = torch.tensor(_FP4, device=dev)
    mids = (grid[1:] + grid[:-1]) / 2.0
    n = 0
    with torch.inference_mode():
        for b in blocks:
            mods = [b.experts]
            if hasattr(b, "shared_experts"):
                mods.append(b.shared_experts)
            for mod in mods:
                for p in mod.parameters():
                    if p.dim() >= 2:
                        p.data = nvfp4_qdq(p.data, grid, mids)
                        n += 1
    return n


def accept(p_logits, q_logits, n, gen):
    p = torch.softmax(p_logits, -1)
    q = torch.softmax(q_logits, -1)
    overlap = float(torch.minimum(p, q).sum())
    d = torch.multinomial(q, n, replacement=True, generator=gen)
    acc = torch.minimum(torch.ones_like(p[d]), p[d] / q[d].clamp_min(1e-30))
    return overlap, float((torch.rand(n, generator=gen) < acc).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V2-Lite")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sample-count", type=int, default=512)
    ap.add_argument("--pos-from", type=int, default=4)
    ap.add_argument("--cache-fracs", default="0.0625,0.125,0.25,0.5,0.75,1.0")
    ap.add_argument("--quant", choices=["none", "nvfp4"], default="none")
    ap.add_argument("--output-json", type=Path, required=True)
    a = ap.parse_args()
    dev = a.device

    cfg = AutoConfig.from_pretrained(a.model, local_files_only=True)
    E = cfg.n_routed_experts
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True,
    ).to(dev).eval()
    blocks = [m for m in model.modules() if type(m).__name__ == "DeepseekV2Moe"]
    for i, b in enumerate(blocks):
        b._moe_idx = i
    type(blocks[0]).forward = patched_forward
    print(f"[cfg] {a.model} routed_E={E} top_k={cfg.num_experts_per_tok} "
          f"n_shared={cfg.n_shared_experts} moe_blocks={len(blocks)}")

    enc_cache = [tok(p, return_tensors="pt", truncation=True, max_length=64).to(dev)
                 for p in PROMPTS]

    # per-block per-expert mass (record mode, full routing)
    _ST["record"] = True
    _ST["masks"] = None
    for enc in enc_cache:
        with torch.inference_mode():
            model(**enc, use_cache=False)
    _ST["record"] = False
    mass = _ST["rec"]
    fracs = [float(x) for x in a.cache_fracs.split(",")]
    gen = torch.Generator().manual_seed(0)

    def masks_for(C):
        return [torch.zeros(E, dtype=torch.bool, device=dev).index_fill_(
            0, torch.topk(mass[i], C).indices, True) for i in range(len(blocks))]

    # shared mass fraction
    _ST["measure"] = True
    with torch.inference_mode():
        model(**enc_cache[0], use_cache=False)
    _ST["measure"] = False
    sn, rn = np.array(_ST["snorm"]), np.array(_ST["rnorm"])
    shared_frac = float(np.mean(sn / (sn + rn + 1e-9)))
    print(f"[shared mass fraction] ~{shared_frac:.3f}")

    # bf16 full-routing targets for both modes, computed BEFORE any quant
    targets = {}
    for mode in ["with_shared", "without_shared"]:
        _ST["ablate"] = (mode == "without_shared")
        _ST["masks"] = None
        tg = []
        for enc in enc_cache:
            with torch.inference_mode():
                lg = model(**enc, use_cache=False).logits[0].float()
            pos = list(range(a.pos_from, lg.shape[0]))
            tg.append((pos, lg[pos].cpu()))
        targets[mode] = tg
    _ST["ablate"] = False

    if a.quant == "nvfp4":
        nq = quantize_experts(blocks, dev)
        print(f"[quant] NVFP4 fake-quantized {nq} expert tensors (routed+shared)")

    results = {}
    for mode in ["with_shared", "without_shared"]:
        _ST["ablate"] = (mode == "without_shared")
        rows = []
        for frac in fracs:
            C = max(1, int(round(frac * E)))
            _ST["masks"] = masks_for(C)
            sa = []
            for enc, (pos, p_cpu) in zip(enc_cache, targets[mode]):
                with torch.inference_mode():
                    q = model(**enc, use_cache=False).logits[0].float().cpu()
                for j, t in enumerate(pos):
                    sa.append(accept(p_cpu[j], q[t], a.sample_count, gen)[1])
            _ST["masks"] = None
            rows.append({"frac": frac, "C": C, "sampled": round(float(np.mean(sa)), 4)})
            print(f"[{mode}] C={C} (f={frac}): sampled={rows[-1]['sampled']}")
        _ST["ablate"] = False
        results[mode] = rows

    a.output_json.parent.mkdir(parents=True, exist_ok=True)
    a.output_json.write_text(json.dumps(
        {"model": a.model, "routed_E": E, "n_shared": cfg.n_shared_experts,
         "quant": a.quant, "shared_mass_fraction": round(shared_frac, 4),
         "results": results}, indent=2))
    print(f"[saved] {a.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
