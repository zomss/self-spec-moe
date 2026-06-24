#!/usr/bin/env python3
"""Affinity placement + request-to-device scheduling + gated local draft.

Semantic-Parallelism-style co-scheduling on no-shared-expert MoE models:
1. Build per-layer co-activation clustering of experts into `num_groups` balanced
   groups (collocate co-activated experts -> a "device"/node holds a cluster).
2. Assign each request to the group that maximizes its local coverage (oracle
   request-to-device scheduling, best case).
3. Measure the per-step local-coverage distribution -> the *draftable subset*
   (high-coverage steps), and the actual local-draft acceptance of that subset
   vs all steps.

Contiguous placement is the no-affinity baseline. One-step acceptance proxy.
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

ROUTER_CLASSES = {"Qwen3MoeTopKRouter", "GptOssTopKRouter"}


def parse_int_list(raw: str) -> list[int]:
    return [int(v) for v in raw.split(",") if v.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--gen-tokens", type=int, default=128)
    p.add_argument("--num-groups", type=parse_int_list, default=[2, 4])
    p.add_argument("--accept-groups", type=parse_int_list, default=[2, 4])
    p.add_argument("--accept-positions", type=int, default=6)
    p.add_argument("--sample-count", type=int, default=512)
    p.add_argument("--draftable-threshold", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Router patch (per-layer masks) -- shared with phase 10
# --------------------------------------------------------------------------- #
def restrict_logits(router_logits, allowed, draft_top_k):
    min_val = torch.finfo(router_logits.dtype).min
    masked = router_logits.masked_fill(~allowed, min_val)
    n = int(allowed.sum())
    eff = min(draft_top_k, n)
    if 0 < eff < n:
        thresh = masked.topk(eff, dim=-1).values[..., -1:]
        masked = masked.masked_fill(masked < thresh, min_val)
    return masked


def make_qwen3_forward(module, allowed, k):
    def forward(self, hidden_states):
        rl = F.linear(hidden_states, self.weight)
        m = restrict_logits(rl, allowed, k)
        probs = F.softmax(m, dim=-1, dtype=torch.float)
        v, i = torch.topk(probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            v = v / v.sum(dim=-1, keepdim=True)
        return rl, v.to(rl.dtype), i

    return types.MethodType(forward, module)


def make_gpt_oss_forward(module, allowed, k):
    def forward(self, hidden_states):
        rl = F.linear(hidden_states, self.weight, self.bias)
        m = restrict_logits(rl, allowed, k)
        v, i = torch.topk(m, self.top_k, dim=-1)
        s = F.softmax(v, dim=-1, dtype=v.dtype)
        return rl, s, i

    return types.MethodType(forward, module)


BUILDERS = {"Qwen3MoeTopKRouter": make_qwen3_forward, "GptOssTopKRouter": make_gpt_oss_forward}


@contextmanager
def patched(routers, masks, k):
    saved = []
    try:
        for mod, mask in zip(routers, masks):
            saved.append((mod, mod.forward))
            mod.forward = BUILDERS[type(mod).__name__](mod, mask, k)
        yield
    finally:
        for mod, orig in saved:
            mod.forward = orig


def accept_metrics(p_logits, q_logits, n, gen):
    p = torch.softmax(p_logits, dim=-1)
    q = torch.softmax(q_logits, dim=-1)
    overlap = float(torch.minimum(p, q).sum())
    d = torch.multinomial(q, n, replacement=True, generator=gen)
    acc = torch.minimum(torch.ones_like(p[d]), p[d] / q[d].clamp_min(1e-30))
    draws = torch.rand(n, generator=gen)
    return overlap, float((draws < acc).float().mean())


# --------------------------------------------------------------------------- #
# Affinity clustering (balanced greedy, per layer)
# --------------------------------------------------------------------------- #
def coactivation(topk, num_experts):
    """topk: [T, k] -> co-activation matrix [E, E] (diag = popularity)."""
    onehot = np.zeros((topk.shape[0], num_experts), dtype=np.float64)
    rows = np.repeat(np.arange(topk.shape[0]), topk.shape[1])
    onehot[rows, topk.reshape(-1)] = 1.0
    return onehot.T @ onehot


def affinity_clusters(coact, num_groups, num_experts):
    """Balanced greedy: seed spread-out, then assign by max co-activation."""
    cap = num_experts // num_groups
    pop = np.diag(coact).copy()
    C = coact.copy()
    np.fill_diagonal(C, 0.0)
    # seeds: most popular, then farthest (least co-activated) from existing seeds
    seeds = [int(np.argmax(pop))]
    while len(seeds) < num_groups:
        # score each expert by min co-activation to existing seeds (want low)
        score = np.array([min(C[e, s] for s in seeds) for e in range(num_experts)])
        score[seeds] = np.inf
        # prefer popular but dissimilar: rank by (low coact, high pop)
        cand = np.argsort(score - 1e-6 * pop)  # low coact first, pop tiebreak
        for c in cand:
            if c not in seeds:
                seeds.append(int(c))
                break
    groups = [[s] for s in seeds]
    sizes = [1] * num_groups
    assigned = set(seeds)
    order = np.argsort(-pop)
    for e in order.tolist():
        if e in assigned:
            continue
        best_g, best_score = -1, -np.inf
        for g in range(num_groups):
            if sizes[g] >= cap:
                continue
            s = sum(C[e, m] for m in groups[g])
            if s > best_score:
                best_score, best_g = s, g
        if best_g < 0:  # all full (remainder) -> least full
            best_g = int(np.argmin(sizes))
        groups[best_g].append(e)
        sizes[best_g] += 1
        assigned.add(e)
    assign = np.empty(num_experts, dtype=np.int64)
    for g, members in enumerate(groups):
        for m in members:
            assign[m] = g
    return assign  # [E] -> group id


def contiguous_clusters(num_groups, num_experts):
    per = num_experts // num_groups
    assign = np.empty(num_experts, dtype=np.int64)
    for g in range(num_groups):
        lo = g * per
        hi = num_experts if g == num_groups - 1 else (g + 1) * per
        assign[lo:hi] = g
    return assign


def request_coverage(req_topk, layer_assign, group, k):
    """Mean fraction of top-k local to `group` across layers/positions, per pos."""
    # req_topk: list over layers of [T, k]; layer_assign: list over layers of [E]
    L = len(req_topk)
    T = req_topk[0].shape[0]
    per_pos = np.zeros(T)
    for layer in range(L):
        local = layer_assign[layer][req_topk[layer]] == group  # [T, k] bool
        per_pos += local.mean(axis=1)
    return per_pos / L  # [T] coverage per position (averaged over layers)


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    cfg = AutoConfig.from_pretrained(args.model)
    mt = getattr(cfg, "model_type", "")
    E = getattr(cfg, "num_experts", None) or getattr(cfg, "num_local_experts", None)
    k = getattr(cfg, "num_experts_per_tok")
    print(f"[config] {mt} E={E} top_k={k}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    load_kwargs = dict(dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)
    if mt == "gpt_oss":
        from transformers import Mxfp4Config

        load_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs).to(args.device).eval()
    routers = [m for m in model.modules() if type(m).__name__ in ROUTER_CLASSES]
    L = len(routers)

    # ---- decode traces ----
    seqs = []
    for prompt in PROMPTS:
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=64).to(args.device)
        pl = int(enc["input_ids"].shape[1])
        with torch.inference_mode():
            gen = model.generate(
                **enc, max_new_tokens=args.gen_tokens, do_sample=True,
                temperature=0.7, top_p=0.9, pad_token_id=tok.pad_token_id,
            )
        ids = gen[0]
        with torch.inference_mode():
            out = model(input_ids=ids.unsqueeze(0), use_cache=False, output_router_logits=True)
        topk = [rl.float().topk(k, dim=-1).indices.cpu().numpy() for rl in out.router_logits]
        seqs.append({"ids": ids.cpu(), "pl": pl, "topk": topk, "T": int(ids.shape[0])})
    print(f"[trace] {len(seqs)} requests, L={L}")

    # population co-activation per layer
    coact = []
    for layer in range(L):
        allt = np.concatenate([s["topk"][layer][s["pl"]:] for s in seqs], axis=0)
        coact.append(coactivation(allt, E))

    results = {"model": args.model, "num_experts": E, "top_k": k, "by_config": {}}
    gen_rng = torch.Generator(device="cpu")
    gen_rng.manual_seed(args.seed + 5)

    for placement in ["affinity", "contiguous"]:
        for G in args.num_groups:
            if placement == "affinity":
                layer_assign = [affinity_clusters(coact[l], G, E) for l in range(L)]
            else:
                layer_assign = [contiguous_clusters(G, E) for _ in range(L)]

            # request-to-group assignment (oracle: best coverage) + per-step coverage
            all_cov = []
            req_group = []
            for s in seqs:
                pl, T = s["pl"], s["T"]
                covs = []
                for g in range(G):
                    pp = request_coverage([t[pl:] for t in s["topk"]], [la for la in layer_assign], g, k)
                    covs.append(pp.mean())
                g_best = int(np.argmax(covs))
                req_group.append(g_best)
                pp = request_coverage([t[pl:] for t in s["topk"]], layer_assign, g_best, k)
                all_cov.append(pp)
            step_cov = np.concatenate(all_cov)
            lar = float(step_cov.mean())
            draftable = float((step_cov >= args.draftable_threshold).mean())
            fully_local = float((step_cov >= 0.999).mean())
            cfg_key = f"{placement}_G{G}"
            results["by_config"][cfg_key] = {
                "placement": placement, "num_groups": G,
                "LAR_mean": round(lar, 4),
                "cov_p50": round(float(np.percentile(step_cov, 50)), 4),
                "cov_p90": round(float(np.percentile(step_cov, 90)), 4),
                "draftable_frac_ge_thr": round(draftable, 4),
                "fully_local_frac": round(fully_local, 4),
            }
            print(f"[{cfg_key}] LAR={lar:.3f} p90cov={np.percentile(step_cov,90):.3f} "
                  f"draftable(>= {args.draftable_threshold})={draftable:.3f} fully_local={fully_local:.3f}")

            # ---- acceptance: masked forward for assigned group, bin by step coverage ----
            if placement == "affinity" and G in args.accept_groups:
                acc_rows = []
                for s, g_best in zip(seqs, req_group):
                    pl, T = s["pl"], s["T"]
                    hi = T - 2
                    if hi <= pl:
                        continue
                    vpos = sorted(set(np.linspace(pl, hi, args.accept_positions).astype(int).tolist()))
                    masks = []
                    for layer in range(L):
                        mk = torch.zeros(E, dtype=torch.bool, device=args.device)
                        mk[np.where(layer_assign[layer] == g_best)[0]] = True
                        masks.append(mk)
                    for t in vpos:
                        ids = s["ids"][: t + 1].unsqueeze(0).to(args.device)
                        with torch.inference_mode():
                            p = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[:, -1, :].float().squeeze(0).cpu()
                        with patched(routers, masks, k):
                            with torch.inference_mode():
                                q = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[:, -1, :].float().squeeze(0).cpu()
                        overlap, sampled = accept_metrics(p, q, args.sample_count, gen_rng)
                        # step coverage at t
                        cov_t = float(np.mean([
                            (layer_assign[layer][s["topk"][layer][t]] == g_best).mean() for layer in range(L)
                        ]))
                        acc_rows.append({"cov": cov_t, "overlap": overlap, "sampled": sampled})
                draft_sel = [r for r in acc_rows if r["cov"] >= args.draftable_threshold]
                results["by_config"][cfg_key]["acceptance"] = {
                    "all_steps_mean_sampled": round(float(np.mean([r["sampled"] for r in acc_rows])), 4),
                    "all_steps_n": len(acc_rows),
                    "draftable_mean_sampled": round(float(np.mean([r["sampled"] for r in draft_sel])), 4) if draft_sel else None,
                    "draftable_n": len(draft_sel),
                    "draftable_mean_cov": round(float(np.mean([r["cov"] for r in draft_sel])), 4) if draft_sel else None,
                }
                a = results["by_config"][cfg_key]["acceptance"]
                print(f"   acceptance: all={a['all_steps_mean_sampled']} (n={a['all_steps_n']}) | "
                      f"draftable={a['draftable_mean_sampled']} (n={a['draftable_n']})")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2))
    print("\n=== summary ===")
    print(json.dumps(results["by_config"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
