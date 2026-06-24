#!/usr/bin/env python3
"""Rebalancing ceiling: local-draft acceptance vs replicated draft-cache budget M.

A per-device draft-expert cache of M experts is replicated on every device and
chosen per layer as the top-M experts by aggregated gate mass (the mass-optimal
fixed set). Because the cache is identical on every device it is EP-invariant, so
acceptance vs M bounds what any rebalancing/replication scheme can achieve at any
EP size. Verification stays exact; only the draft routing is masked.

One-token acceptance proxy (same as Phase 05/07): optimistic upper bound.
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
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--max-prompts-per-bucket", type=int, default=4)
    p.add_argument("--budgets", type=parse_int_list, default=None)
    p.add_argument("--sample-count", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--local-files-only", action="store_true")
    return p.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def load_prompts(max_per_bucket: int) -> list[tuple[str, str]]:
    return [
        (bucket, prompt)
        for bucket, items in PROMPT_BANK.items()
        for prompt in items[:max_per_bucket]
    ]


# --------------------------------------------------------------------------- #
# Router patching (per-layer masks)
# --------------------------------------------------------------------------- #
def restrict_logits(
    router_logits: torch.Tensor, allowed: torch.Tensor, draft_top_k: int
) -> torch.Tensor:
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
        probs = F.softmax(masked, dim=-1, dtype=torch.float)
        top_value, indices = torch.topk(probs, self.top_k, dim=-1)
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


BUILDERS = {
    "Qwen3MoeTopKRouter": make_qwen3_forward,
    "GptOssTopKRouter": make_gpt_oss_forward,
}


@contextmanager
def patched_per_layer(routers, masks, draft_top_k):
    """masks[i] is the allowed bool mask for routers[i] (layer i)."""
    saved = []
    try:
        for module, mask in zip(routers, masks):
            saved.append((module, module.forward))
            module.forward = BUILDERS[type(module).__name__](module, mask, draft_top_k)
        yield
    finally:
        for module, original in saved:
            module.forward = original


# --------------------------------------------------------------------------- #
# Forward + metrics
# --------------------------------------------------------------------------- #
def encode(tokenizer, prompt, args):
    return tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=args.max_length
    ).to(args.device)


def full_forward(model, tokenizer, prompt, args):
    enc = encode(tokenizer, prompt, args)
    with torch.inference_mode():
        out = model(
            **enc, use_cache=False, logits_to_keep=1, output_router_logits=True
        )
    p_logits = out.logits[:, -1, :].float().squeeze(0).cpu()
    layer_probs = [F.softmax(rl.float(), dim=-1).cpu() for rl in out.router_logits]
    return p_logits, torch.stack(layer_probs, dim=0)  # [L, T, E]


def draft_forward(model, tokenizer, prompt, args):
    enc = encode(tokenizer, prompt, args)
    with torch.inference_mode():
        out = model(**enc, use_cache=False, logits_to_keep=1)
    return out.logits[:, -1, :].float().squeeze(0).cpu()


def acceptance_metrics(p_logits, q_logits, sample_count, generator):
    p = torch.softmax(p_logits, dim=-1)
    q = torch.softmax(q_logits, dim=-1)
    overlap = float(torch.minimum(p, q).sum())
    draft = torch.multinomial(q, sample_count, replacement=True, generator=generator)
    accept = torch.minimum(
        torch.ones_like(p[draft]), p[draft] / q[draft].clamp_min(1e-30)
    )
    draws = torch.rand(sample_count, generator=generator)
    sampled = float((draws < accept).float().mean())
    top1 = float(p.argmax() == q.argmax())
    return {"expected_acceptance": overlap, "sampled_acceptance": sampled, "top1": top1}


def mean_over(rows, key):
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
    default_top_k = getattr(config, "num_experts_per_tok")
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
        try:
            from transformers import Mxfp4Config

            load_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
            print("[load] using Mxfp4Config(dequantize=True)")
        except Exception as exc:  # noqa: BLE001
            print(f"[load] Mxfp4Config unavailable ({exc}); loading as-is")

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.to(args.device)
    model.eval()

    routers = [m for m in model.modules() if type(m).__name__ in ROUTER_CLASSES]
    print(f"[setup] found {len(routers)} router modules")

    prompts = load_prompts(args.max_prompts_per_bucket)

    # Pass 1: full forward -> target dist + per-layer router probs.
    full_p, full_router = [], []
    for _, prompt in prompts:
        p_logits, router_probs = full_forward(model, tokenizer, prompt, args)
        full_p.append(p_logits)
        full_router.append(router_probs)

    num_layers = full_router[0].shape[0]
    assert len(routers) == num_layers, (
        f"router count {len(routers)} != layer count {num_layers}"
    )

    # Aggregate per-layer expert mass across all prompt tokens: [L, E].
    layer_mass = torch.zeros(num_layers, num_experts, dtype=torch.float64)
    for rp in full_router:
        layer_mass += rp.sum(dim=1).double()

    budgets = args.budgets or sorted(
        {
            b
            for b in [
                default_top_k,
                num_experts // 8,
                num_experts // 4,
                num_experts // 2,
                3 * num_experts // 4,
                num_experts,
            ]
            if default_top_k <= b <= num_experts
        }
    )
    print(f"[setup] budgets={budgets} prompts={len(prompts)}")

    rows = []
    for m in budgets:
        # Per-layer top-M experts by mass -> allowed masks.
        masks = []
        for layer in range(num_layers):
            top = torch.topk(layer_mass[layer], m).indices
            mask = torch.zeros(num_experts, dtype=torch.bool, device=args.device)
            mask[top] = True
            masks.append(mask)
        # Coverage metrics from traces (no forward needed).
        gammas, overlaps = [], []
        cpu_masks = [mask.cpu() for mask in masks]
        for rp in full_router:  # [L, T, E]
            for layer in range(num_layers):
                lm = cpu_masks[layer]
                probs = rp[layer]  # [T, E]
                gammas.append(float(probs[:, lm].sum(dim=-1).mean()))
                true_topk = probs.topk(default_top_k, dim=-1).indices  # [T, k]
                overlaps.append(float(lm[true_topk].float().mean()))
        gamma = float(np.mean(gammas))
        topk_overlap = float(np.mean(overlaps))

        gen = torch.Generator(device="cpu")
        gen.manual_seed(args.seed + m)
        per_prompt = []
        with patched_per_layer(routers, masks, default_top_k):
            for (bucket, prompt), p_logits in zip(prompts, full_p):
                q_logits = draft_forward(model, tokenizer, prompt, args)
                metric = acceptance_metrics(
                    p_logits, q_logits, args.sample_count, gen
                )
                metric["bucket"] = bucket
                per_prompt.append(metric)

        by_bucket = {
            f"acc_{b}": round(
                mean_over(
                    [r for r in per_prompt if r["bucket"] == b], "sampled_acceptance"
                ),
                4,
            )
            for b in PROMPT_BANK
            if any(r["bucket"] == b for r in per_prompt)
        }
        row = {
            "budget_M": m,
            "M_over_E": round(m / num_experts, 3),
            "gamma": round(gamma, 4),
            "topk_overlap": round(topk_overlap, 4),
            "expected_acceptance": round(
                mean_over(per_prompt, "expected_acceptance"), 4
            ),
            "sampled_acceptance": round(
                mean_over(per_prompt, "sampled_acceptance"), 4
            ),
            "top1_match": round(mean_over(per_prompt, "top1"), 4),
            **by_bucket,
        }
        rows.append(row)
        print(
            f"M={m:>4} (M/E={row['M_over_E']}): gamma={gamma:.3f} "
            f"overlap={topk_overlap:.3f} sampled_acc={row['sampled_acceptance']:.3f} "
            f"top1={row['top1_match']:.3f}"
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    lead = [
        "budget_M",
        "M_over_E",
        "gamma",
        "topk_overlap",
        "expected_acceptance",
        "sampled_acceptance",
        "top1_match",
    ]
    fieldnames = lead + [c for c in sorted({k for r in rows for k in r}) if c not in lead]
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
                    "num_layers": num_layers,
                    "num_prompts": len(prompts),
                    "rows": rows,
                },
                f,
                indent=2,
            )

    print("\n=== acceptance vs budget ===")
    for r in rows:
        print(
            f"M={r['budget_M']:>4} M/E={r['M_over_E']:<5} "
            f"sampled={r['sampled_acceptance']:<6} gamma={r['gamma']:<6} "
            f"top1={r['top1_match']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
