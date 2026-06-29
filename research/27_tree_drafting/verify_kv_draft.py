#!/usr/bin/env python3
"""Measure the acceptance upside of a verify-context-KV draft.

Conservative draft (what we measured so far): the draft recomputes the whole context
through LOCAL-routing weights -> degraded context. Verify-context-KV draft: the context
K,V come from the VERIFY (full bf16) forward (already cached in a real system), and only
the predicting token runs local routing -> still comm-free, but accurate context.

Measures h(b)=P(verify-next in draft top-b) BOTH ways on the same positions along the
verify-greedy path, so the delta is the clean upside.
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
BMAX = 4


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


def build_masks(model, ids, E, C, dev):
    with torch.inference_mode():
        out = model(input_ids=ids, use_cache=False, output_router_logits=True)
    masks = []
    for rl in out.router_logits:
        mass = F.softmax(rl.float(), dim=-1).sum(0)
        mk = torch.zeros(E, dtype=torch.bool, device=dev); mk[torch.topk(mass, C).indices] = True
        masks.append(mk)
    return masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--cache-frac", type=float, default=0.5)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--local-files-only", action="store_true")
    a = ap.parse_args()
    dev = a.device

    cfg = AutoConfig.from_pretrained(a.model, local_files_only=a.local_files_only)
    E = cfg.num_experts; C = int(round(a.cache_frac * E))
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=a.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        local_files_only=a.local_files_only).to(dev).eval()
    gates = [m for m in model.modules() if type(m).__name__ == "Qwen3MoeTopKRouter"]
    D = a.depth
    print(f"[cfg] E={E} C={C}(0.5E) depth={D}")

    hit_cons = np.zeros(BMAX); hit_vkv = np.zeros(BMAX); n = 0
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids.to(dev)
        masks = build_masks(model, ids, E, C, dev)
        # verify-greedy reference
        ref, cur = [], ids
        for _ in range(D):
            with torch.inference_mode():
                t = int(model(input_ids=cur, use_cache=False).logits[0, -1].argmax())
            ref.append(t); cur = torch.cat([cur, torch.tensor([[t]], device=dev)], dim=1)

        for j in range(D):
            ctx = torch.cat([ids, torch.tensor([ref[:j]], device=dev)], 1) if j else ids
            # conservative: full local forward over ctx
            with local_routing(gates, masks), torch.inference_mode():
                lc = model(input_ids=ctx, use_cache=False).logits[0, -1]
            # verify-KV: full-weight KV for ctx[:-1], then local forward of last token
            with torch.inference_mode():
                past = model(input_ids=ctx[:, :-1], use_cache=True).past_key_values
            with local_routing(gates, masks), torch.inference_mode():
                lv = model(input_ids=ctx[:, -1:], past_key_values=past,
                           use_cache=True).logits[0, -1]
            tc = torch.topk(lc, BMAX).indices.tolist()
            tv = torch.topk(lv, BMAX).indices.tolist()
            n += 1
            for b in range(1, BMAX + 1):
                hit_cons[b - 1] += int(ref[j] in tc[:b])
                hit_vkv[b - 1] += int(ref[j] in tv[:b])

    hc = hit_cons / n; hv = hit_vkv / n
    print(f"\n{'b':>3} {'conservative':>12} {'verify-KV':>10} {'delta':>7}")
    for b in range(BMAX):
        print(f"{b+1:>3} {hc[b]:>12.3f} {hv[b]:>10.3f} {hv[b]-hc[b]:>+7.3f}")
    a.output_json.parent.mkdir(parents=True, exist_ok=True)
    a.output_json.write_text(json.dumps({
        "C": C, "depth": D, "n": n,
        "h_conservative": [round(x, 4) for x in hc.tolist()],
        "h_verify_kv": [round(x, 4) for x in hv.tolist()]}, indent=2))
    print(f"[saved] {a.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
