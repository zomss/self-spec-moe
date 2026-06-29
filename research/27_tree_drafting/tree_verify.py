#!/usr/bin/env python3
"""Tree-attention validation: REAL accept length of concrete tree structures.

The B1-analog for trees. Builds a draft token tree (local-routing draft), verifies it in
ONE forward with a proper tree attention mask (each node attends to context + ancestors)
and shared position ids per depth, then follows verify's greedy through the tree to get
the real accept length. Compares to the h(b)+geometric prediction from tree_optimize.

Schedules (branching per depth): [1,1]=chain k2, [2,2]=full d2b2, [2,2,2]=full d3b2.
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
SCHEDULES = {"chain_k2": [1, 1], "full_d2b2": [2, 2],
             "chain_k3": [1, 1, 1], "full_d3b2": [2, 2, 2]}


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


def build_masks(model, ids, E, C, dev):
    with torch.inference_mode():
        out = model(input_ids=ids, use_cache=False, output_router_logits=True)
    masks = []
    for rl in out.router_logits:
        mass = F.softmax(rl.float(), dim=-1).sum(0)
        mk = torch.zeros(E, dtype=torch.bool, device=dev)
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


def build_tree(model, gates, masks, ids, schedule):
    """Full expansion: each frontier node -> draft top-b children. Returns node list."""
    nodes = []  # {token, parent, depth}
    frontier = [(-1, ids)]  # (parent_node_idx, context_ids)
    for depth, b in enumerate(schedule):
        nf = []
        for pidx, ctx in frontier:
            with local_routing(gates, masks), torch.inference_mode():
                dlog = model(input_ids=ctx, use_cache=False).logits[0, -1]
            for t in torch.topk(dlog, b).indices.tolist():
                nidx = len(nodes)
                nodes.append({"token": t, "parent": pidx, "depth": depth})
                nf.append((nidx, torch.cat([ctx, torch.tensor([[t]], device=ids.device)], 1)))
        frontier = nf
    return nodes


def verify_tree(model, ids, nodes):
    L = ids.shape[1]
    N = len(nodes)
    dev = ids.device
    seq = ids[0].tolist() + [n["token"] for n in nodes]
    pos = list(range(L)) + [L + n["depth"] for n in nodes]
    T = L + N
    neg = torch.finfo(torch.bfloat16).min
    m = torch.full((T, T), neg, device=dev, dtype=torch.bfloat16)
    for i in range(L):           # context: causal
        m[i, : i + 1] = 0
    for k, n in enumerate(nodes):  # node attends context + ancestors + self
        p = L + k
        m[p, :L] = 0
        cur = k
        while cur != -1:
            m[p, L + cur] = 0
            cur = nodes[cur]["parent"]
    with torch.inference_mode():
        out = model(input_ids=torch.tensor([seq], device=dev),
                    position_ids=torch.tensor([pos], device=dev),
                    attention_mask=m[None, None], use_cache=False)
    logits = out.logits[0]
    # follow verify greedy through the tree
    kids = {-1: []}
    for k, n in enumerate(nodes):
        kids.setdefault(n["parent"], []).append(k)
        kids.setdefault(k, [])
    accepted, cur_pos, cur_node = [], L - 1, -1
    while True:
        v = int(logits[cur_pos].argmax())
        match = [k for k in kids[cur_node] if nodes[k]["token"] == v]
        accepted.append(v)
        if match:
            cur_node = match[0]; cur_pos = L + cur_node
        else:
            break
    return accepted, len(accepted) - 1  # (tokens incl bonus, accept_len)


def geom(rate, d):
    return rate * (1 - rate ** d) / (1 - rate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-tokens", type=int, default=48)
    ap.add_argument("--cache-frac", type=float, default=0.5)
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
        attn_implementation="eager", local_files_only=a.local_files_only,
    ).to(dev).eval()
    gates = [m for m in model.modules() if type(m).__name__ == "Qwen3MoeTopKRouter"]
    H = {1: 0.875, 2: 0.979}  # measured h(b) (tree_hitrate)
    print(f"[cfg] E={E} C={C}(0.5E) eager-attn schedules={list(SCHEDULES)}")

    prep = []
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids.to(dev)
        prep.append((ids, build_masks(model, ids, E, C, dev),
                     reference_greedy(model, ids, a.n_tokens)))

    results = {}
    for name, sched in SCHEDULES.items():
        b = sched[0]; d = len(sched)
        all_acc, lossless_hits, total_tok = [], 0, 0
        for ids, masks, ref in prep:
            cur = ids; produced = []
            while len(produced) < a.n_tokens:
                nodes = build_tree(model, gates, masks, cur, sched)
                acc_seq, acc_len = verify_tree(model, cur, nodes)
                all_acc.append(acc_len)
                # losslessness: produced tokens should match the bf16 greedy ref
                for j, t in enumerate(acc_seq):
                    if len(produced) + j < len(ref):
                        lossless_hits += int(t == ref[len(produced) + j]); total_tok += 1
                produced.extend(acc_seq)
                cur = torch.cat([cur, torch.tensor([acc_seq], device=dev)], dim=1)
        mean_acc = float(np.mean(all_acc))
        pred = geom(H[b], d)
        results[name] = {"schedule": sched, "nodes": sum(b ** (j + 1) for j in range(d)),
                         "mean_accept_real": round(mean_acc, 3),
                         "pred_geom": round(pred, 3),
                         "real_over_pred": round(mean_acc / pred, 3),
                         "greedy_match": round(lossless_hits / max(total_tok, 1), 3),
                         "cycles": len(all_acc)}
        print(f"{name:>10} sched={sched} nodes={results[name]['nodes']:>2}: "
              f"real_accept={mean_acc:.3f} pred={pred:.3f} "
              f"(real/pred={results[name]['real_over_pred']}) "
              f"greedy_match={results[name]['greedy_match']}")

    print("\n--- tree vs chain (real) ---")
    print(f"  chain_k2 {results['chain_k2']['mean_accept_real']} -> "
          f"full_d2b2 {results['full_d2b2']['mean_accept_real']} "
          f"(+{100*(results['full_d2b2']['mean_accept_real']/results['chain_k2']['mean_accept_real']-1):.0f}%)")
    a.output_json.parent.mkdir(parents=True, exist_ok=True)
    a.output_json.write_text(json.dumps({"model": a.model, "C": C, "results": results}, indent=2))
    print(f"[saved] {a.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
