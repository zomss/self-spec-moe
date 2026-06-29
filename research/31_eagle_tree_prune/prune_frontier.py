#!/usr/bin/env python3
"""Stage A (31a): does free confidence-pruning of a wide local-routing tree recover
accept length vs the oracle and vs a directly-drafted small tree?

Draft a WIDE local-routing tree, full-EP tree-attention verify it (Phase 27 machinery
extended to record per-node draft-prob P_conf and verify-prob P_exact + the accepted
path). Prune to a node budget N by path-probability under each signal; the surviving
prefix of the verify-accepted path IS the pruned tree's accept length (pruning is lossless
-> only shortens). Compare:
  - P_conf (free, draft's own confidence) -- the 31a candidate pruner.
  - P_exact (full-EP probs) -- oracle / upper bound.
  - DIRECT small trees (chain k, full d2b2) drafted at the SAME context -- the decisive
    baseline: if pruned-wide(N) ~= direct-small(N), pruning adds nothing (just draft small).
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
WIDE = [2, 2, 2, 2]                 # wide tree to prune (depth 4, 30 nodes)
BUDGETS = [1, 2, 3, 4, 6, 8, 12, 16]
DIRECT = {"chain_k2": [1, 1], "chain_k3": [1, 1, 1], "chain_k4": [1, 1, 1, 1],
          "full_d2b2": [2, 2]}


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


def greedy(model, ids):
    with torch.inference_mode():
        return int(model(input_ids=ids, use_cache=False).logits[0, -1].argmax())


def build_tree(model, gates, masks, ids, schedule, dev):
    nodes = []  # token, parent, depth, dp (draft prob of this token given parent)
    frontier = [(-1, ids)]
    for depth, b in enumerate(schedule):
        nf = []
        for pidx, ctx in frontier:
            with local_routing(gates, masks), torch.inference_mode():
                lg = model(input_ids=ctx, use_cache=False).logits[0, -1]
            probs = F.softmax(lg.float(), -1)
            for t in torch.topk(lg, b).indices.tolist():
                nidx = len(nodes)
                nodes.append({"token": t, "parent": pidx, "depth": depth, "dp": float(probs[t])})
                nf.append((nidx, torch.cat([ctx, torch.tensor([[t]], device=dev)], 1)))
        frontier = nf
    return nodes


def verify_tree(model, ids, nodes, dev):
    """Full-EP tree-attention verify. Returns accepted path (node idx list) and sets vp."""
    L = ids.shape[1]; N = len(nodes)
    seq = ids[0].tolist() + [n["token"] for n in nodes]
    pos = list(range(L)) + [L + n["depth"] for n in nodes]
    T = L + N; neg = torch.finfo(torch.bfloat16).min
    m = torch.full((T, T), neg, device=dev, dtype=torch.bfloat16)
    for i in range(L):
        m[i, : i + 1] = 0
    for k, n in enumerate(nodes):
        p = L + k; m[p, :L] = 0; cur = k
        while cur != -1:
            m[p, L + cur] = 0; cur = nodes[cur]["parent"]
    with torch.inference_mode():
        lg = model(input_ids=torch.tensor([seq], device=dev),
                   position_ids=torch.tensor([pos], device=dev),
                   attention_mask=m[None, None], use_cache=False).logits[0]
    kids = {-1: []}
    for k, n in enumerate(nodes):
        kids.setdefault(n["parent"], []).append(k); kids.setdefault(k, [])
        ppos = L - 1 if n["parent"] == -1 else L + n["parent"]
        n["vp"] = float(F.softmax(lg[ppos].float(), -1)[n["token"]])
    path, cur_pos, cur_node = [], L - 1, -1
    while True:
        v = int(lg[cur_pos].argmax())
        match = [k for k in kids[cur_node] if nodes[k]["token"] == v]
        if match:
            path.append(match[0]); cur_node = match[0]; cur_pos = L + match[0]
        else:
            break
    return path


def path_prob(nodes, key):
    pp = [0.0] * len(nodes)
    for k, n in enumerate(nodes):
        pp[k] = n[key] * (1.0 if n["parent"] == -1 else pp[n["parent"]])
    return pp


def surviving(nodes, path, key, N):
    pp = path_prob(nodes, key)
    kept = set(sorted(range(len(nodes)), key=lambda k: -pp[k])[:N])
    m = 0
    for a in path:
        if a in kept:
            m += 1
        else:
            break
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-tokens", type=int, default=40)
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
        attn_implementation="eager", local_files_only=a.local_files_only).to(dev).eval()
    gates = [m for m in model.modules() if type(m).__name__ == "Qwen3MoeTopKRouter"]
    Nfull = sum(int(np.prod(WIDE[:i + 1])) for i in range(len(WIDE)))
    print(f"[cfg] E={E} C={C}(0.5E) wide={WIDE} ({Nfull} nodes) budgets={BUDGETS}")

    Lfull, conf, exact = [], {N: [] for N in BUDGETS}, {N: [] for N in BUDGETS}
    direct = {k: [] for k in DIRECT}
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids.to(dev)
        masks = build_masks(model, ids, E, C, dev)
        cur, produced = ids, 0
        while produced < a.n_tokens:
            nodes = build_tree(model, gates, masks, cur, WIDE, dev)
            path = verify_tree(model, cur, nodes, dev)
            Lf = len(path); Lfull.append(Lf)
            for N in BUDGETS:
                conf[N].append(surviving(nodes, path, "dp", N))
                exact[N].append(surviving(nodes, path, "vp", N))
            # direct small trees at the SAME context
            for k, sch in DIRECT.items():
                dn = build_tree(model, gates, masks, cur, sch, dev)
                direct[k].append(len(verify_tree(model, cur, dn, dev)))
            # advance using the wide tree's accepted path + bonus
            acc = [nodes[i]["token"] for i in path]
            bonus = greedy(model, torch.cat([cur, torch.tensor([acc], device=dev)], 1)) if acc \
                else greedy(model, cur)
            acc = acc + [bonus]
            cur = torch.cat([cur, torch.tensor([acc], device=dev)], 1); produced += len(acc)

    Lf = float(np.mean(Lfull))
    print(f"\nL_full(wide {WIDE}) = {Lf:.3f} accept (max {len(WIDE)})\n")
    print(f"{'N':>3} {'P_conf(free)':>13} {'P_exact(oracle)':>16} {'recall_conf':>12}")
    res = {"wide": WIDE, "Nfull": Nfull, "L_full": round(Lf, 3), "budgets": {}}
    for N in BUDGETS:
        c = float(np.mean(conf[N])); x = float(np.mean(exact[N]))
        res["budgets"][N] = {"conf": round(c, 3), "exact": round(x, 3),
                             "recall_conf": round(c / Lf, 3), "recall_exact": round(x / Lf, 3)}
        print(f"{N:>3} {c:>13.3f} {x:>16.3f} {c/Lf:>12.3f}")
    print(f"\ndirect small trees (same context, accept length):")
    res["direct"] = {}
    for k, sch in DIRECT.items():
        d = float(np.mean(direct[k])); res["direct"][k] = {"nodes": sum(int(np.prod(sch[:i+1])) for i in range(len(sch))), "accept": round(d, 3)}
        print(f"  {k:>10} (N={res['direct'][k]['nodes']:>2}): {d:.3f}")
    print(f"\nHEADLINE: at equal node budget, pruned-wide(P_conf) vs direct-small:")
    for k in DIRECT:
        Nn = res["direct"][k]["nodes"]
        if Nn in res["budgets"]:
            pc = res["budgets"][Nn]["conf"]; ds = res["direct"][k]["accept"]
            print(f"  N={Nn:>2}: pruned-conf {pc:.2f} vs {k} {ds:.2f}  "
                  f"({'PRUNE WINS' if pc > ds + 0.03 else 'tie/direct' if pc < ds - 0.03 else 'tie'})")
    a.output_json.parent.mkdir(parents=True, exist_ok=True)
    a.output_json.write_text(json.dumps({"model": a.model, **res}, indent=2))
    print(f"[saved] {a.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
