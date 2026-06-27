#!/usr/bin/env python3
"""Phase 25 Exp 1: does a shared-expert anchor break the local-routing coverage cap?

The one local-routing lever never tested (Phase 08 scope): an always-on shared expert
gives always-local mass independent of replication. On Qwen1.5-MoE-A2.7B (60 routed
top_k 4, + a large sigmoid-gated shared expert), measure local-routing acceptance vs
the bf16 full-routing target, sweeping the routed cache C, in two modes on the SAME
model/prompts:
  with_shared    : draft = shared + local-routed ; verify = shared + full-routed
  without_shared : draft = local-routed only      ; verify = full-routed only (shared ablated in both)
If with_shared >> without_shared at small C, the anchor breaks the cap. Also report the
shared mass fraction (why).
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
from transformers.models.qwen2_moe import modeling_qwen2_moe as q2

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
_ST = {"ablate": False, "measure": False, "snorm": [], "rnorm": []}


def patched_block_forward(self, hidden_states):
    bsz, seqlen, hdim = hidden_states.shape
    hs = hidden_states.view(-1, hdim)
    shared = self.shared_expert(hs)
    _, rw, sel = self.gate(hs)
    routed = self.experts(hs, sel, rw)
    shared = F.sigmoid(self.shared_expert_gate(hs)) * shared
    if _ST["measure"]:
        _ST["snorm"].append(float(shared.float().norm()))
        _ST["rnorm"].append(float(routed.float().norm()))
    out = routed if _ST["ablate"] else routed + shared
    return out.reshape(bsz, seqlen, hdim)


def local_forward(module, allowed):
    """Match the real router exactly (softmax over ALL experts, then top-k, then
    optional norm) but restrict the SELECTION to the allowed local set. At
    allowed=all this reduces to the original router (-> beta=1 at full coverage),
    independent of norm_topk_prob."""
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
    ap.add_argument("--model", default="Qwen/Qwen1.5-MoE-A2.7B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sample-count", type=int, default=512)
    ap.add_argument("--pos-from", type=int, default=4)
    ap.add_argument("--cache-fracs", default="0.0667,0.133,0.25,0.5,0.75,1.0")
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
    q2.Qwen2MoeSparseMoeBlock.forward = patched_block_forward
    gates = [m for m in model.modules() if type(m).__name__ == "Qwen2MoeTopKRouter"]
    print(f"[cfg] {a.model} routed_E={E} top_k={top_k} layers={len(gates)}")

    # encode + per-layer top-C masks (by routed gate mass; mode-independent)
    enc_cache, mass_acc = [], None
    for prompt in PROMPTS:
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=64).to(dev)
        enc_cache.append(enc)
        with torch.inference_mode():
            out = model(**enc, use_cache=False, output_router_logits=True)
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

    # shared mass fraction (one pass, measure on)
    _ST["measure"] = True
    with torch.inference_mode():
        model(**enc_cache[0], use_cache=False)
    _ST["measure"] = False
    sn, rn = np.array(_ST["snorm"]), np.array(_ST["rnorm"])
    shared_frac = float(np.mean(sn / (sn + rn + 1e-9)))
    print(f"[shared mass fraction] ~{shared_frac:.3f} of MoE-output norm")

    results = {}
    for mode in ["with_shared", "without_shared"]:
        _ST["ablate"] = (mode == "without_shared")
        # target p (full routing) for this mode
        target = []
        for enc in enc_cache:
            with torch.inference_mode():
                lg = model(**enc, use_cache=False).logits[0].float()
            pos = list(range(a.pos_from, lg.shape[0]))
            target.append((pos, lg[pos].cpu()))
        rows = []
        for frac in fracs:
            C = max(1, int(round(frac * E)))
            ov, sa = [], []
            with local_routing(gates, masks_for(C)):
                for enc, (pos, p_cpu) in zip(enc_cache, target):
                    with torch.inference_mode():
                        q = model(**enc, use_cache=False).logits[0].float().cpu()
                    for j, t in enumerate(pos):
                        o, s = accept(p_cpu[j], q[t], a.sample_count, gen)
                        ov.append(o); sa.append(s)
            rows.append({"frac": frac, "C": C,
                         "sampled": round(float(np.mean(sa)), 4),
                         "expected": round(float(np.mean(ov)), 4)})
            print(f"[{mode}] C={C} (f={frac}): sampled={rows[-1]['sampled']}")
        _ST["ablate"] = False
        results[mode] = rows

    a.output_json.parent.mkdir(parents=True, exist_ok=True)
    a.output_json.write_text(json.dumps(
        {"model": a.model, "routed_E": E, "top_k": top_k,
         "shared_mass_fraction": round(shared_frac, 4), "results": results}, indent=2))
    print(f"[saved] {a.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
