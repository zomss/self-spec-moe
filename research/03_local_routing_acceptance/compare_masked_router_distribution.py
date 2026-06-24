#!/usr/bin/env python3
"""Compare full-router and local-masked-router next-token distributions."""

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
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--router-logits", type=Path)
    parser.add_argument("--prompts-jsonl", type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-prompts", type=int, default=8)
    parser.add_argument("--ep-size", type=int, default=2)
    parser.add_argument("--hot-expert-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def load_prompts(path: Path | None, max_prompts: int) -> list[str]:
    if path is None:
        prompts = [
            "The future of artificial intelligence is",
            "Explain mixture of experts models in simple terms.",
            "Write a short Python function to add two numbers.",
            "What is the capital of France?",
            "Solve: if x + 3 = 7, what is x?",
            "Summarize why distributed inference needs communication.",
            "List three benefits of expert parallelism.",
            "Translate 'hello world' to Korean.",
        ]
        return prompts[:max_prompts]

    prompts = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            prompts.append(record["prompt"] if isinstance(record, dict) else str(record))
            if len(prompts) >= max_prompts:
                break
    return prompts


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def contiguous_placement(num_experts: int, ep_size: int) -> list[set[int]]:
    experts_per_rank = num_experts // ep_size
    local_experts = []
    for rank in range(ep_size):
        start = rank * experts_per_rank
        end = (rank + 1) * experts_per_rank
        if rank == ep_size - 1:
            end = num_experts
        local_experts.append(set(range(start, end)))
    return local_experts


def random_placement(num_experts: int, ep_size: int, seed: int) -> list[set[int]]:
    rng = np.random.default_rng(seed)
    experts = np.arange(num_experts)
    rng.shuffle(experts)
    local_experts = [set() for _ in range(ep_size)]
    for idx, expert in enumerate(experts.tolist()):
        local_experts[idx % ep_size].add(expert)
    return local_experts


def top_hot_experts(router_logits: Path | None, count: int, num_experts: int) -> list[int]:
    if count <= 0:
        return []
    if router_logits is None:
        return list(range(min(count, num_experts)))
    data = np.load(router_logits)
    logits = data["router_logits"]
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = probs / probs.sum(axis=-1, keepdims=True)
    scores = probs.mean(axis=(0, 1))
    return np.argsort(-scores)[:count].tolist()


def hot_replicated_placement(
    base: list[set[int]],
    hot_experts: list[int],
) -> list[set[int]]:
    return [set(experts).union(hot_experts) for experts in base]


def build_placements(
    *,
    num_experts: int,
    ep_size: int,
    hot_experts: list[int],
    seed: int,
) -> dict[str, list[set[int]]]:
    contiguous = contiguous_placement(num_experts, ep_size)
    return {
        "contiguous": contiguous,
        "random": random_placement(num_experts, ep_size, seed),
        "contiguous_hot_replicated": hot_replicated_placement(contiguous, hot_experts),
    }


def is_qwen_topk_router(module: torch.nn.Module) -> bool:
    return all(hasattr(module, attr) for attr in ("weight", "hidden_dim", "top_k"))


@contextmanager
def masked_qwen_routers(model: torch.nn.Module, local_experts: set[int]):
    patched = []

    def make_forward(module):
        def forward(self, hidden_states):
            hidden_states = hidden_states.reshape(-1, self.hidden_dim)
            router_logits = F.linear(hidden_states, self.weight)
            mask = torch.ones(router_logits.shape[-1], dtype=torch.bool, device=router_logits.device)
            mask[list(local_experts)] = False
            router_logits = router_logits.masked_fill(
                mask,
                torch.finfo(router_logits.dtype).min,
            )
            router_probs = torch.nn.functional.softmax(
                router_logits,
                dtype=torch.float,
                dim=-1,
            )
            router_top_value, router_indices = torch.topk(
                router_probs,
                self.top_k,
                dim=-1,
            )
            if self.norm_topk_prob:
                router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
            router_top_value = router_top_value.to(router_logits.dtype)
            return router_logits, router_top_value, router_indices

        return types.MethodType(forward, module)

    try:
        for module in model.modules():
            if is_qwen_topk_router(module):
                patched.append((module, module.forward))
                module.forward = make_forward(module)
        if not patched:
            raise ValueError("No Qwen-style top-k router modules were found")
        yield
    finally:
        for module, original_forward in patched:
            module.forward = original_forward


def next_token_logits(model, tokenizer, prompts: list[str], args) -> torch.Tensor:
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_length,
    ).to(args.device)
    with torch.inference_mode():
        outputs = model(**encoded, use_cache=False, logits_to_keep=1)
    return outputs.logits[:, -1, :].float().cpu()


def distribution_metrics(full_logits: torch.Tensor, draft_logits: torch.Tensor) -> dict[str, float]:
    p = torch.softmax(full_logits, dim=-1)
    q = torch.softmax(draft_logits, dim=-1)
    overlap = torch.minimum(p, q).sum(dim=-1)
    p_log = torch.log_softmax(full_logits, dim=-1)
    q_log = torch.log_softmax(draft_logits, dim=-1)
    kl_pq = (p * (p_log - q_log)).sum(dim=-1)
    top1_match = (p.argmax(dim=-1) == q.argmax(dim=-1)).float()
    target_prob_draft_top1 = p.gather(1, q.argmax(dim=-1, keepdim=True)).squeeze(1)
    return {
        "acceptance_overlap_mean": float(overlap.mean()),
        "acceptance_overlap_min": float(overlap.min()),
        "kl_pq_mean": float(kl_pq.mean()),
        "top1_match_rate": float(top1_match.mean()),
        "target_prob_draft_top1_mean": float(target_prob_draft_top1.mean()),
    }


def main() -> int:
    args = parse_args()
    prompts = load_prompts(args.prompts_jsonl, args.max_prompts)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_from_name(args.dtype),
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        low_cpu_mem_usage=True,
    )
    model.to(args.device)
    model.eval()

    num_experts = getattr(model.config, "num_experts", None)
    if num_experts is None:
        raise ValueError("This proxy currently expects config.num_experts")
    hot_experts = top_hot_experts(args.router_logits, args.hot_expert_count, num_experts)
    placements = build_placements(
        num_experts=num_experts,
        ep_size=args.ep_size,
        hot_experts=hot_experts,
        seed=args.seed,
    )

    full_logits = next_token_logits(model, tokenizer, prompts, args)
    rows = []
    for placement_name, local_experts_by_rank in placements.items():
        for rank, local_experts in enumerate(local_experts_by_rank):
            with masked_qwen_routers(model, local_experts):
                draft_logits = next_token_logits(model, tokenizer, prompts, args)
            metrics = distribution_metrics(full_logits, draft_logits)
            rows.append(
                {
                    "placement": placement_name,
                    "rank": rank,
                    "local_experts": len(local_experts),
                    **metrics,
                }
            )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as f:
            json.dump(
                {
                    "model": args.model,
                    "num_prompts": len(prompts),
                    "ep_size": args.ep_size,
                    "hot_experts": hot_experts,
                    "rows": rows,
                },
                f,
                indent=2,
            )
    print(json.dumps({"rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
