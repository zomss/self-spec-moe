#!/usr/bin/env python3
"""Measure one-token local-draft acceptance on recent MoE checkpoints.

Supports Qwen3-MoE (`Qwen3MoeTopKRouter`) and GPT-OSS (`GptOssTopKRouter`).
For each placement / EP rank / draft_top_k, routers are masked to device-local
experts and the resulting next-token distribution is compared against the exact
full-routing distribution. This is a one-token acceptance proxy, not multi-token
runtime drafting, so measured acceptance is an optimistic upper bound.
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

# Domain-bucketed prompts. Routing locality is domain-dependent, so we report
# acceptance per bucket as well as overall.
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
        "def quicksort(arr):",
        "Explain what a race condition is and how to avoid it.",
        "Write a SQL query to find the second highest salary.",
    ],
    "math": [
        "Solve: if x + 3 = 7, what is x?",
        "What is the derivative of x^2 + 3x with respect to x?",
        "Compute the greatest common divisor of 48 and 36.",
        "A train travels 60 km in 45 minutes. What is its speed in km/h?",
    ],
}


def parse_int_list(raw_value: str) -> list[int]:
    values = [int(v) for v in raw_value.split(",") if v.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-prompts-per-bucket", type=int, default=4)
    parser.add_argument("--ep-sizes", type=parse_int_list, default=[2, 4, 8])
    parser.add_argument(
        "--draft-top-k-values", type=parse_int_list, default=[1, 2, 4, 8]
    )
    parser.add_argument("--hot-expert-count", type=int, default=8)
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def load_prompts(max_per_bucket: int) -> list[tuple[str, str]]:
    prompts: list[tuple[str, str]] = []
    for bucket, items in PROMPT_BANK.items():
        for prompt in items[:max_per_bucket]:
            prompts.append((bucket, prompt))
    return prompts


# --------------------------------------------------------------------------- #
# Placement
# --------------------------------------------------------------------------- #
def contiguous_placement(num_experts: int, ep_size: int) -> list[set[int]]:
    per_rank = num_experts // ep_size
    local: list[set[int]] = []
    for rank in range(ep_size):
        start = rank * per_rank
        end = num_experts if rank == ep_size - 1 else (rank + 1) * per_rank
        local.append(set(range(start, end)))
    return local


def random_placement(num_experts: int, ep_size: int, seed: int) -> list[set[int]]:
    rng = np.random.default_rng(seed)
    experts = np.arange(num_experts)
    rng.shuffle(experts)
    local: list[set[int]] = [set() for _ in range(ep_size)]
    for idx, expert in enumerate(experts.tolist()):
        local[idx % ep_size].add(expert)
    return local


def build_placements(
    num_experts: int, ep_size: int, hot: list[int], seed: int
) -> dict[str, list[set[int]]]:
    contiguous = contiguous_placement(num_experts, ep_size)
    random = random_placement(num_experts, ep_size, seed)
    return {
        "contiguous": contiguous,
        "random": random,
        "contiguous_hot_replicated": [s | set(hot) for s in contiguous],
        "random_hot_replicated": [s | set(hot) for s in random],
    }


# --------------------------------------------------------------------------- #
# Router patching
# --------------------------------------------------------------------------- #
ROUTER_CLASSES = {"Qwen3MoeTopKRouter", "GptOssTopKRouter"}


def restrict_logits(
    router_logits: torch.Tensor, allowed: torch.Tensor, draft_top_k: int
) -> torch.Tensor:
    """Mask to local experts, then keep only the top draft_top_k local experts."""
    min_val = torch.finfo(router_logits.dtype).min
    masked = router_logits.masked_fill(~allowed, min_val)
    num_allowed = int(allowed.sum())
    eff = min(draft_top_k, num_allowed)
    if eff < num_allowed:
        thresh = masked.topk(eff, dim=-1).values[..., -1:]
        masked = masked.masked_fill(masked < thresh, min_val)
    return masked


def make_qwen3_forward(module, allowed: torch.Tensor, draft_top_k: int):
    def forward(self, hidden_states):
        router_logits = F.linear(hidden_states, self.weight)
        masked = restrict_logits(router_logits, allowed, draft_top_k)
        router_probs = F.softmax(masked, dim=-1, dtype=torch.float)
        top_value, indices = torch.topk(router_probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            top_value = top_value / top_value.sum(dim=-1, keepdim=True)
        top_value = top_value.to(router_logits.dtype)
        return router_logits, top_value, indices

    return types.MethodType(forward, module)


def make_gpt_oss_forward(module, allowed: torch.Tensor, draft_top_k: int):
    def forward(self, hidden_states):
        router_logits = F.linear(hidden_states, self.weight, self.bias)
        masked = restrict_logits(router_logits, allowed, draft_top_k)
        top_value, indices = torch.topk(masked, self.top_k, dim=-1)
        scores = F.softmax(top_value, dim=-1, dtype=top_value.dtype)
        return router_logits, scores, indices

    return types.MethodType(forward, module)


@contextmanager
def patched_routers(model, *, allowed: set[int], draft_top_k: int, device, num_experts):
    mask = torch.zeros(num_experts, dtype=torch.bool, device=device)
    mask[list(allowed)] = True
    builders = {
        "Qwen3MoeTopKRouter": make_qwen3_forward,
        "GptOssTopKRouter": make_gpt_oss_forward,
    }
    patched = []
    try:
        for module in model.modules():
            name = type(module).__name__
            if name in builders:
                patched.append((module, module.forward))
                module.forward = builders[name](module, mask, draft_top_k)
        if not patched:
            raise ValueError(
                f"No router modules in {ROUTER_CLASSES} found in this model"
            )
        yield
    finally:
        for module, original in patched:
            module.forward = original


# --------------------------------------------------------------------------- #
# Forward helpers
# --------------------------------------------------------------------------- #
def full_forward(model, tokenizer, prompt: str, args):
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
    ).to(args.device)
    with torch.inference_mode():
        out = model(
            **enc,
            use_cache=False,
            logits_to_keep=1,
            output_router_logits=True,
        )
    p_logits = out.logits[:, -1, :].float().squeeze(0).cpu()
    # router_logits: tuple over layers, each [num_tokens, num_experts]
    layer_probs = [
        F.softmax(rl.float(), dim=-1).cpu() for rl in out.router_logits
    ]
    router_probs = torch.stack(layer_probs, dim=0)  # [L, T, E]
    return p_logits, router_probs


def draft_forward(model, tokenizer, prompt: str, args):
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
    ).to(args.device)
    with torch.inference_mode():
        out = model(**enc, use_cache=False, logits_to_keep=1)
    return out.logits[:, -1, :].float().squeeze(0).cpu()


def coverage_metrics(
    router_probs: torch.Tensor, allowed: set[int], default_top_k: int
) -> dict[str, float]:
    """gamma and top-k overlap from full-router probabilities [L, T, E]."""
    e = router_probs.shape[-1]
    mask = torch.zeros(e, dtype=torch.bool)
    mask[list(allowed)] = True
    gamma = router_probs[..., mask].sum(dim=-1).mean()
    true_topk = router_probs.topk(default_top_k, dim=-1).indices  # [L, T, k]
    local_hit = mask[true_topk].float().mean()
    return {"gamma": float(gamma), "topk_overlap": float(local_hit)}


def acceptance_metrics(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    sample_count: int,
    generator: torch.Generator,
) -> dict[str, float]:
    p = torch.softmax(p_logits, dim=-1)
    q = torch.softmax(q_logits, dim=-1)
    overlap = float(torch.minimum(p, q).sum())
    draft_tokens = torch.multinomial(
        q, sample_count, replacement=True, generator=generator
    )
    p_vals = p[draft_tokens]
    q_vals = q[draft_tokens].clamp_min(1e-30)
    accept = torch.minimum(torch.ones_like(p_vals), p_vals / q_vals)
    draws = torch.rand(sample_count, generator=generator)
    sampled = float((draws < accept).float().mean())
    p_log = torch.log_softmax(p_logits, dim=-1)
    q_log = torch.log_softmax(q_logits, dim=-1)
    kl = float((p * (p_log - q_log)).sum())
    top1 = float(p.argmax() == q.argmax())
    return {
        "expected_acceptance": overlap,
        "sampled_acceptance": sampled,
        "kl_pq": kl,
        "top1_match": top1,
    }


def mean_over(rows: list[dict], key: str) -> float:
    return float(np.mean([r[key] for r in rows])) if rows else float("nan")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()
    config = AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    model_type = getattr(config, "model_type", "")
    num_experts = getattr(config, "num_experts", None) or getattr(
        config, "num_local_experts", None
    )
    default_top_k = getattr(config, "num_experts_per_tok", None)
    if num_experts is None or default_top_k is None:
        raise ValueError(
            "Could not read num_experts / num_experts_per_tok from config"
        )
    print(
        f"[config] model_type={model_type} num_experts={num_experts} "
        f"default_top_k={default_top_k}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )

    load_kwargs = dict(
        dtype=dtype_from_name(args.dtype),
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        low_cpu_mem_usage=True,
    )
    if model_type == "gpt_oss":
        # Dequantize the MXFP4 checkpoint to bf16 to avoid triton-kernel deps.
        try:
            from transformers import Mxfp4Config

            load_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
            print("[load] using Mxfp4Config(dequantize=True)")
        except Exception as exc:  # noqa: BLE001
            print(f"[load] Mxfp4Config unavailable ({exc}); loading as-is")

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.to(args.device)
    model.eval()

    prompts = load_prompts(args.max_prompts_per_bucket)
    draft_top_ks = sorted({k for k in args.draft_top_k_values if k <= default_top_k})
    print(f"[setup] prompts={len(prompts)} draft_top_k={draft_top_ks}")

    # Pass 1: full forward per prompt -> target dist + router probs + hot experts.
    full_p: list[torch.Tensor] = []
    full_router: list[torch.Tensor] = []
    expert_mass = torch.zeros(num_experts, dtype=torch.float64)
    for bucket, prompt in prompts:
        p_logits, router_probs = full_forward(model, tokenizer, prompt, args)
        full_p.append(p_logits)
        full_router.append(router_probs)
        expert_mass += router_probs.mean(dim=(0, 1)).double()
    hot = torch.argsort(expert_mass, descending=True)[: args.hot_expert_count]
    hot = hot.tolist()
    print(f"[hot] top hot experts: {hot}")

    rows: list[dict] = []
    for ep_size in args.ep_sizes:
        placements = build_placements(num_experts, ep_size, hot, args.seed)
        for placement_name, ranks in placements.items():
            for rank, allowed in enumerate(ranks):
                cov = [
                    coverage_metrics(rp, allowed, default_top_k)
                    for rp in full_router
                ]
                gamma = mean_over(cov, "gamma")
                topk_overlap = mean_over(cov, "topk_overlap")
                for draft_top_k in draft_top_ks:
                    gen = torch.Generator(device="cpu")
                    gen.manual_seed(
                        args.seed + 1000 * ep_size + 100 * rank + draft_top_k
                    )
                    with patched_routers(
                        model,
                        allowed=allowed,
                        draft_top_k=draft_top_k,
                        device=args.device,
                        num_experts=num_experts,
                    ):
                        per_prompt = []
                        for (bucket, prompt), p_logits in zip(prompts, full_p):
                            q_logits = draft_forward(model, tokenizer, prompt, args)
                            m = acceptance_metrics(
                                p_logits, q_logits, args.sample_count, gen
                            )
                            m["bucket"] = bucket
                            per_prompt.append(m)
                    by_bucket = {}
                    for b in PROMPT_BANK:
                        sel = [r for r in per_prompt if r["bucket"] == b]
                        if sel:
                            by_bucket[b] = mean_over(sel, "sampled_acceptance")
                    row = {
                        "ep_size": ep_size,
                        "placement": placement_name,
                        "rank": rank,
                        "local_experts": len(allowed),
                        "draft_top_k": draft_top_k,
                        "gamma": round(gamma, 4),
                        "topk_overlap": round(topk_overlap, 4),
                        "expected_acceptance": round(
                            mean_over(per_prompt, "expected_acceptance"), 4
                        ),
                        "sampled_acceptance": round(
                            mean_over(per_prompt, "sampled_acceptance"), 4
                        ),
                        "kl_pq": round(mean_over(per_prompt, "kl_pq"), 4),
                        "top1_match": round(mean_over(per_prompt, "top1_match"), 4),
                        **{
                            f"acc_{b}": round(v, 4) for b, v in by_bucket.items()
                        },
                    }
                    rows.append(row)
                    print(
                        f"ep{ep_size} {placement_name} r{rank} "
                        f"k{draft_top_k}: gamma={gamma:.3f} "
                        f"overlap={topk_overlap:.3f} "
                        f"sampled_acc={row['sampled_acceptance']:.3f}"
                    )

    rows.sort(key=lambda r: r["sampled_acceptance"], reverse=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r})
    # keep a stable, readable column order
    lead = [
        "ep_size",
        "placement",
        "rank",
        "local_experts",
        "draft_top_k",
        "gamma",
        "topk_overlap",
        "expected_acceptance",
        "sampled_acceptance",
        "kl_pq",
        "top1_match",
    ]
    fieldnames = lead + [c for c in fieldnames if c not in lead]
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as f:
            json.dump(
                {
                    "model": args.model,
                    "model_type": model_type,
                    "num_experts": num_experts,
                    "default_top_k": default_top_k,
                    "hot_experts": hot,
                    "num_prompts": len(prompts),
                    "best_sampled_acceptance": rows[0]["sampled_acceptance"]
                    if rows
                    else None,
                    "rows": rows,
                },
                f,
                indent=2,
            )

    print("\n=== top 5 by sampled acceptance ===")
    for r in rows[:5]:
        print(
            f"ep{r['ep_size']} {r['placement']} r{r['rank']} k{r['draft_top_k']}: "
            f"sampled={r['sampled_acceptance']} gamma={r['gamma']} "
            f"topk_overlap={r['topk_overlap']} top1={r['top1_match']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
