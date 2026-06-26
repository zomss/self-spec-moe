#!/usr/bin/env python3
"""Stage B1: lockstep self-speculative MoE cycle -- losslessness + real acceptance.

Validates the two things Stage A's composed speedup assumed: (1) the draft/verify
cycle is LOSSLESS (exact match to bf16 greedy), and (2) the real multi-token
accepted-length is consistent with the one-step beta -> T(k,beta) model. Single GPU,
HF transformers, recompute-from-scratch (correctness over speed); the comm-bound
throughput itself is Stage A / Stage B2.

Draft = comm-light **local routing** (per-layer router masked to a per-request top-C
expert cache, Phase 18 style); verify = full-routing bf16. Greedy decoding, so the
rejection rule reduces to: accept the draft prefix where draft-greedy == verify-greedy,
then append verify's token -> output is exactly bf16 greedy.
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
    "List three benefits of expert parallelism.",
    "Describe how a transformer attention layer works.",
]


def restrict_logits(rl, allowed, k):
    mn = torch.finfo(rl.dtype).min
    m = rl.masked_fill(~allowed, mn)
    n = int(allowed.sum())
    if k < n:
        thr = m.topk(k, dim=-1).values[..., -1:]
        m = m.masked_fill(m < thr, mn)
    return m


def local_router_forward(module, allowed):
    def forward(self, hidden_states):
        rl = F.linear(hidden_states, self.weight)
        m = restrict_logits(rl, allowed, self.top_k)
        probs = F.softmax(m, dim=-1, dtype=torch.float)
        v, i = torch.topk(probs, self.top_k, dim=-1)
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
            g.forward = local_router_forward(g, mk)
        yield
    finally:
        for g, orig in saved:
            g.forward = orig


def build_masks(model, gates, ids, E, C, device):
    """Per-request static top-C-by-gate-mass mask per layer."""
    with torch.inference_mode():
        out = model(input_ids=ids, use_cache=False, output_router_logits=True)
    masks = []
    for rl in out.router_logits:
        mass = F.softmax(rl.float(), dim=-1).sum(0)
        top = torch.topk(mass, C).indices
        mk = torch.zeros(E, dtype=torch.bool, device=device)
        mk[top] = True
        masks.append(mk)
    return masks


def greedy_next(model, ids):
    with torch.inference_mode():
        return int(model(input_ids=ids, use_cache=False).logits[0, -1].argmax())


def reference_greedy(model, ids, n):
    out = []
    cur = ids
    for _ in range(n):
        t = greedy_next(model, cur)
        out.append(t)
        cur = torch.cat([cur, torch.tensor([[t]], device=ids.device)], dim=1)
    return out


def spec_greedy(model, gates, masks, ids0, n, k):
    """Lockstep local-draft / full-verify greedy. Returns (tokens, accepted_lens)."""
    ids = ids0
    produced, acc_lens = [], []
    while len(produced) < n:
        L = ids.shape[1]
        # draft k tokens with local routing (recompute each step)
        drafts = []
        with local_routing(gates, masks):
            cur = ids
            for _ in range(k):
                t = greedy_next(model, cur)
                drafts.append(t)
                cur = torch.cat([cur, torch.tensor([[t]], device=ids.device)], dim=1)
        # verify: one full-routing forward over ids + drafts
        cat = torch.cat(
            [ids, torch.tensor([drafts], device=ids.device)], dim=1
        )
        with torch.inference_mode():
            vlog = model(input_ids=cat, use_cache=False).logits[0]
        vtok = [int(vlog[L - 1 + j].argmax()) for j in range(k + 1)]
        # accept prefix where draft == verify-greedy
        m = 0
        while m < k and drafts[m] == vtok[m]:
            m += 1
        accepted = drafts[:m] + [vtok[m]]  # m accepted + 1 bonus (verify token)
        acc_lens.append(m)
        produced.extend(accepted)
        ids = torch.cat(
            [ids, torch.tensor([accepted], device=ids.device)], dim=1
        )
    return produced[:n], acc_lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-tokens", type=int, default=96)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--cache-fracs", default="0.25,0.5")
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--local-files-only", action="store_true")
    a = ap.parse_args()

    cfg = AutoConfig.from_pretrained(a.model, local_files_only=a.local_files_only)
    E = cfg.num_experts
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=a.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        local_files_only=a.local_files_only,
    ).to(a.device).eval()
    gates = [m for m in model.modules() if type(m).__name__ == "Qwen3MoeTopKRouter"]
    print(f"[cfg] E={E} top_k={cfg.num_experts_per_tok} gates={len(gates)}")

    fracs = [float(x) for x in a.cache_fracs.split(",")]
    results = {}
    for frac in fracs:
        C = int(round(frac * E))
        loss_ok, all_m, n_cyc = True, [], 0
        for prompt in PROMPTS:
            ids = tok(prompt, return_tensors="pt").input_ids.to(a.device)
            masks = build_masks(model, gates, ids, E, C, a.device)
            ref = reference_greedy(model, ids, a.n_tokens)
            spec, acc = spec_greedy(model, gates, masks, ids, a.n_tokens, a.k)
            match = ref == spec
            loss_ok = loss_ok and match
            all_m.extend(acc)
            n_cyc += len(acc)
            print(f"  C={C} '{prompt[:30]}...': lossless={match} "
                  f"mean_accept={np.mean(acc):.2f}/{a.k} cycles={len(acc)}")
        mean_m = float(np.mean(all_m))
        # implied per-token beta from mean accepted m over k draws:
        # E[m] = sum_{i=1..k} beta^i ; solve numerically
        betas = np.linspace(0.01, 0.999, 999)
        em = np.array([sum(b ** i for i in range(1, a.k + 1)) for b in betas])
        beta_hat = float(betas[np.argmin(np.abs(em - mean_m))])
        results[f"C={C}"] = {
            "cache_frac": frac, "C": C, "lossless": loss_ok,
            "mean_accepted": round(mean_m, 3),
            "tokens_per_cycle": round(mean_m + 1, 3),
            "implied_beta": round(beta_hat, 3),
            "n_cycles": n_cyc,
        }
        print(f"C={C} (frac {frac}): LOSSLESS={loss_ok} mean_accept={mean_m:.3f}/{a.k}"
              f" -> tokens/cycle={mean_m + 1:.3f}, implied beta~{beta_hat:.3f}")

    a.output_json.parent.mkdir(parents=True, exist_ok=True)
    a.output_json.write_text(json.dumps(
        {"model": a.model, "E": E, "k": a.k, "n_tokens": a.n_tokens,
         "results": results}, indent=2))
    print(f"[saved] {a.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
