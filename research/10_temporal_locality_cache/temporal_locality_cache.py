#!/usr/bin/env python3
"""Temporal routing locality: a verify-warmed LRU draft cache vs a static cache.

Decode a real continuation per prompt, read per-layer per-position true top-k
experts, then measure how well a per-layer LRU expert cache of capacity C (warmed
by verified routing) covers the next position's experts -- compared to a static
top-C-by-mass cache of the same size. Optionally validate with masked-forward
sampled acceptance. Coverage is a one-step proxy; multi-token drafting would be
lower.
"""

from __future__ import annotations

import argparse
import csv
import json
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

PROMPT_BANK: dict[str, list[str]] = {
    "general": [
        "The future of artificial intelligence is",
        "Summarize why distributed inference needs communication.",
        "List three benefits of expert parallelism.",
        "Describe how a mixture-of-experts layer routes tokens.",
    ],
    "chat": [
        "Hi! Can you recommend a good book for a long flight?",
        "I'm feeling stressed about work. Any advice?",
        "What's a fun fact about the ocean?",
        "Explain mixture of experts models in simple terms.",
    ],
    "code": [
        "Write a Python function to compute the nth Fibonacci number.",
        "Implement binary search over a sorted list in Python.",
        "Explain what a race condition is and how to avoid it.",
        "Write a SQL query to find the second highest salary.",
    ],
    "math": [
        "Solve step by step: if x + 3 = 7, what is x?",
        "What is the derivative of x^2 + 3x with respect to x?",
        "Compute the greatest common divisor of 48 and 36, showing work.",
        "A train travels 60 km in 45 minutes. What is its speed in km/h?",
    ],
}

ROUTER_CLASSES = {"Qwen3MoeTopKRouter", "GptOssTopKRouter"}


def parse_int_list(raw: str) -> list[int]:
    vals = [int(v) for v in raw.split(",") if v.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("at least one value required")
    return vals


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--output-json", type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16"
    )
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--max-prompts-per-bucket", type=int, default=4)
    p.add_argument("--gen-tokens", type=int, default=192)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--cache-sizes", type=parse_int_list, required=True)
    p.add_argument("--validate-cache-sizes", type=parse_int_list, default=None)
    p.add_argument("--validate-prompts", type=int, default=6)
    p.add_argument("--validate-positions", type=int, default=5)
    p.add_argument("--sample-count", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--local-files-only", action="store_true")
    return p.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
        name
    ]


def load_prompts(max_per_bucket: int) -> list[tuple[str, str]]:
    return [
        (bucket, prompt)
        for bucket, items in PROMPT_BANK.items()
        for prompt in items[:max_per_bucket]
    ]


# --------------------------------------------------------------------------- #
# Router patching (per-layer masks) -- shared with phase 07/09
# --------------------------------------------------------------------------- #
def restrict_logits(router_logits, allowed, draft_top_k):
    min_val = torch.finfo(router_logits.dtype).min
    masked = router_logits.masked_fill(~allowed, min_val)
    num_allowed = int(allowed.sum())
    eff = min(draft_top_k, num_allowed)
    if 0 < eff < num_allowed:
        thresh = masked.topk(eff, dim=-1).values[..., -1:]
        masked = masked.masked_fill(masked < thresh, min_val)
    return masked


def make_qwen3_forward(module, allowed, draft_top_k):
    def forward(self, hidden_states):
        router_logits = F.linear(hidden_states, self.weight)
        masked = restrict_logits(router_logits, allowed, draft_top_k)
        probs = F.softmax(masked, dim=-1, dtype=torch.float)
        top_value, indices = torch.topk(probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            top_value = top_value / top_value.sum(dim=-1, keepdim=True)
        return router_logits, top_value.to(router_logits.dtype), indices

    return types.MethodType(forward, module)


def make_gpt_oss_forward(module, allowed, draft_top_k):
    def forward(self, hidden_states):
        router_logits = F.linear(hidden_states, self.weight, self.bias)
        masked = restrict_logits(router_logits, allowed, draft_top_k)
        top_value, indices = torch.topk(masked, self.top_k, dim=-1)
        scores = F.softmax(top_value, dim=-1, dtype=top_value.dtype)
        return router_logits, scores, indices

    return types.MethodType(forward, module)


BUILDERS = {"Qwen3MoeTopKRouter": make_qwen3_forward, "GptOssTopKRouter": make_gpt_oss_forward}


@contextmanager
def patched_per_layer(routers, masks, draft_top_k):
    saved = []
    try:
        for module, mask in zip(routers, masks):
            saved.append((module, module.forward))
            module.forward = BUILDERS[type(module).__name__](module, mask, draft_top_k)
        yield
    finally:
        for module, original in saved:
            module.forward = original


def acceptance_metrics(p_logits, q_logits, sample_count, generator):
    p = torch.softmax(p_logits, dim=-1)
    q = torch.softmax(q_logits, dim=-1)
    overlap = float(torch.minimum(p, q).sum())
    draft = torch.multinomial(q, sample_count, replacement=True, generator=generator)
    accept = torch.minimum(torch.ones_like(p[draft]), p[draft] / q[draft].clamp_min(1e-30))
    draws = torch.rand(sample_count, generator=generator)
    sampled = float((draws < accept).float().mean())
    top1 = float(p.argmax() == q.argmax())
    return {"expected_acceptance": overlap, "sampled_acceptance": sampled, "top1": top1}


# --------------------------------------------------------------------------- #
# LRU cache simulation over a single layer's per-position top-k
# --------------------------------------------------------------------------- #
def lru_coverage(topk_seq, decode_start, capacity, snapshot_positions=None):
    """topk_seq: [T, k] int array. Coverage of pos t vs cache built from < t."""
    cache, cset = [], set()
    covs, alls, occ = [], [], []
    snaps = {}
    for t in range(topk_seq.shape[0]):
        experts = topk_seq[t].tolist()
        if t >= decode_start:
            if snapshot_positions is not None and t in snapshot_positions:
                snaps[t] = set(cset)
            inter = sum(1 for e in experts if e in cset)
            covs.append(inter / len(experts))
            alls.append(float(inter == len(experts)))
            occ.append(len(cset))
        for e in experts:
            if e in cset:
                cache.remove(e)
            cache.insert(0, e)
            cset.add(e)
        while len(cache) > capacity:
            cset.discard(cache.pop())
    return covs, alls, occ, snaps


def static_coverage(topk_seq, decode_start, mass, capacity):
    keep = set(np.argsort(-mass)[:capacity].tolist())
    covs, alls = [], []
    for t in range(decode_start, topk_seq.shape[0]):
        experts = topk_seq[t].tolist()
        inter = sum(1 for e in experts if e in keep)
        covs.append(inter / len(experts))
        alls.append(float(inter == len(experts)))
    return covs, alls


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    config = AutoConfig.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code, local_files_only=args.local_files_only
    )
    model_type = getattr(config, "model_type", "")
    num_experts = getattr(config, "num_experts", None) or getattr(config, "num_local_experts", None)
    default_top_k = getattr(config, "num_experts_per_tok")
    print(f"[config] model_type={model_type} num_experts={num_experts} top_k={default_top_k}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(
        dtype=dtype_from_name(args.dtype),
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        low_cpu_mem_usage=True,
    )
    if model_type == "gpt_oss":
        try:
            from transformers import Mxfp4Config

            load_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
            print("[load] Mxfp4Config(dequantize=True)")
        except Exception as exc:  # noqa: BLE001
            print(f"[load] Mxfp4Config unavailable ({exc})")

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.to(args.device)
    model.eval()
    routers = [m for m in model.modules() if type(m).__name__ in ROUTER_CLASSES]
    print(f"[setup] routers={len(routers)}")

    prompts = load_prompts(args.max_prompts_per_bucket)

    # ---- decode sequences + extract per-layer per-position top-k ----
    sequences = []
    for bucket, prompt in prompts:
        enc = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=args.max_length
        ).to(args.device)
        prompt_len = int(enc["input_ids"].shape[1])
        with torch.inference_mode():
            gen = model.generate(
                **enc,
                max_new_tokens=args.gen_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
            )
        full_ids = gen[0]
        with torch.inference_mode():
            out = model(
                input_ids=full_ids.unsqueeze(0), use_cache=False, output_router_logits=True
            )
        topk = [
            rl.float().topk(default_top_k, dim=-1).indices.cpu().numpy().astype(np.int32)
            for rl in out.router_logits
        ]
        mass = [F.softmax(rl.float(), dim=-1).sum(0).cpu().numpy() for rl in out.router_logits]
        sequences.append(
            dict(
                bucket=bucket,
                full_ids=full_ids.cpu(),
                prompt_len=prompt_len,
                topk=topk,
                mass=mass,
                T=int(full_ids.shape[0]),
            )
        )
        print(f"  decoded {bucket}: prompt_len={prompt_len} total={sequences[-1]['T']}")

    num_layers = len(sequences[0]["topk"])

    # ---- trace coverage: dynamic LRU vs static top-C ----
    rows = []
    for c in args.cache_sizes:
        dyn_cov, dyn_all, dyn_occ, stat_cov, stat_all = [], [], [], [], []
        for seq in sequences:
            ds = seq["prompt_len"]
            for layer in range(num_layers):
                tk = seq["topk"][layer]
                covs, alls, occ, _ = lru_coverage(tk, ds, c)
                dyn_cov += covs
                dyn_all += alls
                dyn_occ += occ
                scov, sall = static_coverage(tk, ds, seq["mass"][layer], c)
                stat_cov += scov
                stat_all += sall
        rows.append(
            {
                "cache_size_C": c,
                "C_over_E": round(c / num_experts, 3),
                "dyn_coverage": round(float(np.mean(dyn_cov)), 4),
                "dyn_all_local": round(float(np.mean(dyn_all)), 4),
                "dyn_cache_fill": round(float(np.mean(dyn_occ)), 2),
                "static_coverage": round(float(np.mean(stat_cov)), 4),
                "static_all_local": round(float(np.mean(stat_all)), 4),
            }
        )
        print(
            f"C={c:>4} (C/E={rows[-1]['C_over_E']}): dyn_cov={rows[-1]['dyn_coverage']} "
            f"dyn_all={rows[-1]['dyn_all_local']} | static_cov={rows[-1]['static_coverage']} "
            f"static_all={rows[-1]['static_all_local']}"
        )

    # ---- optional acceptance validation ----
    accept_rows = []
    if args.validate_cache_sizes:
        gen_rng = torch.Generator(device="cpu")
        gen_rng.manual_seed(args.seed + 7)
        vseqs = sequences[: args.validate_prompts]
        for seq in vseqs:
            T, pl = seq["T"], seq["prompt_len"]
            hi = T - 2
            if hi <= pl:
                continue
            vpos = sorted(
                set(np.linspace(pl, hi, args.validate_positions).astype(int).tolist())
            )
            # full p_t at each validation position
            p_by_t = {}
            for t in vpos:
                ids = seq["full_ids"][: t + 1].unsqueeze(0).to(args.device)
                with torch.inference_mode():
                    o = model(input_ids=ids, use_cache=False, logits_to_keep=1)
                p_by_t[t] = o.logits[:, -1, :].float().squeeze(0).cpu()
            for c in args.validate_cache_sizes:
                # per-layer LRU snapshot of cache-before-t at each vpos
                snaps = [
                    lru_coverage(seq["topk"][layer], pl, c, snapshot_positions=set(vpos))[3]
                    for layer in range(num_layers)
                ]
                for t in vpos:
                    masks = []
                    for layer in range(num_layers):
                        s = snaps[layer].get(t, set())
                        mask = torch.zeros(num_experts, dtype=torch.bool, device=args.device)
                        if s:
                            mask[list(s)] = True
                        else:
                            mask[:] = True
                        masks.append(mask)
                    ids = seq["full_ids"][: t + 1].unsqueeze(0).to(args.device)
                    with patched_per_layer(routers, masks, default_top_k):
                        with torch.inference_mode():
                            o = model(input_ids=ids, use_cache=False, logits_to_keep=1)
                    q = o.logits[:, -1, :].float().squeeze(0).cpu()
                    m = acceptance_metrics(p_by_t[t], q, args.sample_count, gen_rng)
                    accept_rows.append({"cache_size_C": c, "bucket": seq["bucket"], **m})
        # aggregate acceptance by C
        accept_summary = []
        for c in args.validate_cache_sizes:
            sel = [r for r in accept_rows if r["cache_size_C"] == c]
            if sel:
                accept_summary.append(
                    {
                        "cache_size_C": c,
                        "n": len(sel),
                        "sampled_acceptance": round(
                            float(np.mean([r["sampled_acceptance"] for r in sel])), 4
                        ),
                        "expected_acceptance": round(
                            float(np.mean([r["expected_acceptance"] for r in sel])), 4
                        ),
                        "top1_match": round(float(np.mean([r["top1"] for r in sel])), 4),
                    }
                )
        print("\n=== dynamic-cache acceptance validation ===")
        for r in accept_summary:
            print(
                f"C={r['cache_size_C']:>4}: sampled_acc={r['sampled_acceptance']} "
                f"expected={r['expected_acceptance']} top1={r['top1_match']} (n={r['n']})"
            )
    else:
        accept_summary = []

    # ---- write outputs ----
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.output_json:
        with args.output_json.open("w") as f:
            json.dump(
                {
                    "model": args.model,
                    "model_type": model_type,
                    "num_experts": num_experts,
                    "default_top_k": default_top_k,
                    "num_layers": num_layers,
                    "gen_tokens": args.gen_tokens,
                    "num_sequences": len(sequences),
                    "coverage_rows": rows,
                    "acceptance_validation": accept_summary,
                },
                f,
                indent=2,
            )

    print("\n=== coverage vs cache size (dynamic LRU vs static) ===")
    for r in rows:
        print(
            f"C={r['cache_size_C']:>4} C/E={r['C_over_E']:<5} "
            f"dyn_cov={r['dyn_coverage']:<6} static_cov={r['static_coverage']:<6} "
            f"dyn_all_local={r['dyn_all_local']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
