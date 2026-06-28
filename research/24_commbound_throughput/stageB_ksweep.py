#!/usr/bin/env python3
"""Stage B1 k-sweep: REAL multi-token accepted-length vs k (replaces geometric model).

The k-curve speedup used the geometric tokens/cycle T(k,beta)=1+beta(1-beta^k)/(1-beta),
which assumes a CONSTANT per-position acceptance beta. This measures the real lockstep
accepted-length at k=2..12 (local-routing draft, C=0.5E, bf16 verify, greedy) and the
per-position conditional acceptance a_j = P(match at depth j | reached j). If a_j is
flat the geometric model holds; if it declines (draft drift), geometric overstates the
large-k gain. One model load; ref greedy computed once per prompt (k-independent).
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
            saved.append((g, g.forward))
            g.forward = local_forward(g, mk)
        yield
    finally:
        for g, o in saved:
            g.forward = o


def build_masks(model, gates, ids, E, C, device):
    with torch.inference_mode():
        out = model(input_ids=ids, use_cache=False, output_router_logits=True)
    masks = []
    for rl in out.router_logits:
        mass = F.softmax(rl.float(), dim=-1).sum(0)
        mk = torch.zeros(E, dtype=torch.bool, device=device)
        mk[torch.topk(mass, C).indices] = True
        masks.append(mk)
    return masks


def greedy_next(model, ids):
    with torch.inference_mode():
        return int(model(input_ids=ids, use_cache=False).logits[0, -1].argmax())


def reference_greedy(model, ids, n):
    out, cur = [], ids
    for _ in range(n):
        t = greedy_next(model, cur)
        out.append(t)
        cur = torch.cat([cur, torch.tensor([[t]], device=ids.device)], dim=1)
    return out


def spec_greedy(model, gates, masks, ids0, n, k):
    ids = ids0
    produced, acc_lens = [], []
    while len(produced) < n:
        L = ids.shape[1]
        drafts = []
        with local_routing(gates, masks):
            cur = ids
            for _ in range(k):
                t = greedy_next(model, cur)
                drafts.append(t)
                cur = torch.cat([cur, torch.tensor([[t]], device=ids.device)], dim=1)
        cat = torch.cat([ids, torch.tensor([drafts], device=ids.device)], dim=1)
        with torch.inference_mode():
            vlog = model(input_ids=cat, use_cache=False).logits[0]
        vtok = [int(vlog[L - 1 + j].argmax()) for j in range(k + 1)]
        m = 0
        while m < k and drafts[m] == vtok[m]:
            m += 1
        accepted = drafts[:m] + [vtok[m]]
        acc_lens.append(m)
        produced.extend(accepted)
        ids = torch.cat([ids, torch.tensor([accepted], device=ids.device)], dim=1)
    return produced[:n], acc_lens


def geom_T(k, b):
    return 1 + b * (1 - b ** k) / (1 - b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-tokens", type=int, default=64)
    ap.add_argument("--cache-frac", type=float, default=0.5)
    ap.add_argument("--k-list", default="2,4,6,8,12")
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--local-files-only", action="store_true")
    a = ap.parse_args()
    dev = a.device

    cfg = AutoConfig.from_pretrained(a.model, local_files_only=a.local_files_only)
    E = cfg.num_experts
    C = int(round(a.cache_frac * E))
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=a.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        local_files_only=a.local_files_only,
    ).to(dev).eval()
    gates = [m for m in model.modules() if type(m).__name__ == "Qwen3MoeTopKRouter"]
    ks = [int(x) for x in a.k_list.split(",")]
    print(f"[cfg] E={E} C={C}(0.5E) gates={len(gates)} k_list={ks}")

    prep = []
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids.to(dev)
        masks = build_masks(model, gates, ids, E, C, dev)
        ref = reference_greedy(model, ids, a.n_tokens)
        prep.append((ids, masks, ref))

    results = {}
    for k in ks:
        all_m, lossless = [], True
        for ids, masks, ref in prep:
            spec, acc = spec_greedy(model, gates, masks, ids, a.n_tokens, k)
            lossless = lossless and (spec == ref)
            all_m.extend(acc)
        all_m = np.array(all_m)
        mean_acc = float(all_m.mean())
        # per-position conditional acceptance a_j = P(m>j | m>=j)
        a_j = []
        for j in range(k):
            denom = int((all_m >= j).sum())
            num = int((all_m > j).sum())
            a_j.append(round(num / denom, 4) if denom else None)
        # implied constant beta from mean accepted (solve geom sum)
        bs = np.linspace(0.01, 0.999, 999)
        em = np.array([sum(b ** i for i in range(1, k + 1)) for b in bs])
        beta_hat = float(bs[np.argmin(np.abs(em - mean_acc))])
        results[k] = {
            "mean_accepted": round(mean_acc, 3),
            "tokens_per_cycle_real": round(mean_acc + 1, 3),
            "implied_beta": round(beta_hat, 3),
            "n_cycles": int(all_m.size),
            "lossless": lossless,
            "per_pos_accept": a_j,
        }
        print(f"k={k:>2}: mean_acc={mean_acc:.3f} tok/cyc={mean_acc+1:.3f} "
              f"implied_beta={beta_hat:.3f} | per-pos a_j={a_j}")

    # geometric vs real, using beta from k=4 (or smallest k>=4)
    bref = results[min(ks, key=lambda k: abs(k - 4))]["implied_beta"]
    print(f"\n--- real tokens/cycle vs geometric (beta={bref} from k~4) ---")
    for k in ks:
        real = results[k]["tokens_per_cycle_real"]
        geo = geom_T(k, bref)
        print(f"k={k:>2}: real={real:.2f}  geometric={geo:.2f}  "
              f"real/geo={real/geo:.3f}")

    a.output_json.parent.mkdir(parents=True, exist_ok=True)
    a.output_json.write_text(json.dumps(
        {"model": a.model, "E": E, "C": C, "n_tokens": a.n_tokens,
         "beta_ref_k4": bref, "results": {str(k): v for k, v in results.items()}},
        indent=2))
    print(f"[saved] {a.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
