#!/usr/bin/env python3
"""Simulate local-expert routing coverage for Phase 03."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Placement:
    name: str
    local_experts_by_rank: list[set[int]]


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def topk_indices(values: np.ndarray, top_k: int) -> np.ndarray:
    if top_k <= 0:
        return np.empty(values.shape[:-1] + (0,), dtype=np.int64)
    indices = np.argpartition(values, -top_k, axis=-1)[..., -top_k:]
    scores = np.take_along_axis(values, indices, axis=-1)
    order = np.argsort(-scores, axis=-1)
    return np.take_along_axis(indices, order, axis=-1)


def contiguous_placement(num_experts: int, ep_size: int) -> Placement:
    experts_per_rank = num_experts // ep_size
    local_experts = []
    for rank in range(ep_size):
        start = rank * experts_per_rank
        end = (rank + 1) * experts_per_rank
        if rank == ep_size - 1:
            end = num_experts
        local_experts.append(set(range(start, end)))
    return Placement("contiguous", local_experts)


def round_robin_placement(num_experts: int, ep_size: int) -> Placement:
    local_experts = [set() for _ in range(ep_size)]
    for expert in range(num_experts):
        local_experts[expert % ep_size].add(expert)
    return Placement("round_robin", local_experts)


def random_placement(num_experts: int, ep_size: int, seed: int) -> Placement:
    rng = np.random.default_rng(seed)
    experts = np.arange(num_experts)
    rng.shuffle(experts)
    local_experts = [set() for _ in range(ep_size)]
    for idx, expert in enumerate(experts.tolist()):
        local_experts[idx % ep_size].add(expert)
    return Placement("random", local_experts)


def hot_replicated_placement(
    base: Placement,
    hot_experts: list[int],
) -> Placement:
    local_experts = [set(experts) for experts in base.local_experts_by_rank]
    for experts in local_experts:
        experts.update(hot_experts)
    return Placement(f"{base.name}_hot_replicated", local_experts)


def hot_grouped_placement(
    num_experts: int,
    ep_size: int,
    hot_experts: list[int],
) -> Placement:
    local_experts = [set() for _ in range(ep_size)]
    for expert in hot_experts:
        local_experts[0].add(expert)

    rank = 0
    for expert in range(num_experts):
        if expert in local_experts[0]:
            continue
        while len(local_experts[rank]) >= int(np.ceil(num_experts / ep_size)):
            rank = min(rank + 1, ep_size - 1)
        local_experts[rank].add(expert)
    return Placement("hot_grouped", local_experts)


def synthetic_router_logits(
    *,
    num_layers: int,
    num_tokens: int,
    num_experts: int,
    hot_experts: list[int],
    hot_bias: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(num_layers, num_tokens, num_experts)).astype(np.float32)
    if hot_experts:
        logits[:, :, hot_experts] += hot_bias
    return logits


def load_logits(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        data = np.load(path)
        if "router_logits" not in data:
            raise ValueError("NPZ input must contain 'router_logits'")
        return data["router_logits"]
    raise ValueError("Only .npy and .npz router logits are supported")


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


def evaluate_placement(
    *,
    placement: Placement,
    probabilities: np.ndarray,
    true_topk: np.ndarray,
) -> list[dict[str, str | float | int]]:
    num_layers, num_tokens, _ = probabilities.shape
    top_k = true_topk.shape[-1]
    rows: list[dict[str, str | float | int]] = []

    rank_gamma = []
    rank_overlap = []
    rank_full_topk = []
    rank_local_counts = []

    for rank, local_experts in enumerate(placement.local_experts_by_rank):
        mask = np.zeros(probabilities.shape[-1], dtype=bool)
        mask[list(local_experts)] = True
        gamma = probabilities[:, :, mask].sum(axis=-1)
        overlap = np.isin(true_topk, list(local_experts)).sum(axis=-1) / top_k
        full_topk = (overlap == 1.0).astype(np.float32)

        rank_gamma.append(gamma)
        rank_overlap.append(overlap)
        rank_full_topk.append(full_topk)
        rank_local_counts.append(len(local_experts))

        gamma_stats = summarize(gamma)
        overlap_stats = summarize(overlap)
        rows.append(
            {
                "placement": placement.name,
                "policy": "fixed_rank",
                "rank": rank,
                "local_experts": len(local_experts),
                "gamma_mean": gamma_stats["mean"],
                "gamma_p50": gamma_stats["p50"],
                "gamma_p90": gamma_stats["p90"],
                "topk_overlap_mean": overlap_stats["mean"],
                "topk_overlap_p50": overlap_stats["p50"],
                "topk_overlap_p90": overlap_stats["p90"],
                "full_topk_local_rate": float(np.mean(full_topk)),
            }
        )

    gamma_by_rank = np.stack(rank_gamma, axis=-1)
    overlap_by_rank = np.stack(rank_overlap, axis=-1)
    full_topk_by_rank = np.stack(rank_full_topk, axis=-1)
    best_gamma_rank = np.argmax(gamma_by_rank, axis=-1)
    layer_idx = np.arange(num_layers)[:, None]
    token_idx = np.arange(num_tokens)[None, :]

    best_gamma = gamma_by_rank[layer_idx, token_idx, best_gamma_rank]
    best_overlap = overlap_by_rank[layer_idx, token_idx, best_gamma_rank]
    best_full_topk = full_topk_by_rank[layer_idx, token_idx, best_gamma_rank]

    gamma_stats = summarize(best_gamma)
    overlap_stats = summarize(best_overlap)
    rows.append(
        {
            "placement": placement.name,
            "policy": "best_gamma_rank",
            "rank": -1,
            "local_experts": float(np.mean(rank_local_counts)),
            "gamma_mean": gamma_stats["mean"],
            "gamma_p50": gamma_stats["p50"],
            "gamma_p90": gamma_stats["p90"],
            "topk_overlap_mean": overlap_stats["mean"],
            "topk_overlap_p50": overlap_stats["p50"],
            "topk_overlap_p90": overlap_stats["p90"],
            "full_topk_local_rate": float(np.mean(best_full_topk)),
        }
    )
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-logits", type=Path)
    parser.add_argument("--num-layers", type=int, default=24)
    parser.add_argument("--num-tokens", type=int, default=256)
    parser.add_argument("--num-experts", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--ep-size", type=int, default=2)
    parser.add_argument("--hot-expert-count", type=int, default=8)
    parser.add_argument("--hot-bias", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.router_logits:
        logits = load_logits(args.router_logits)
    else:
        hot_experts = list(range(args.hot_expert_count))
        logits = synthetic_router_logits(
            num_layers=args.num_layers,
            num_tokens=args.num_tokens,
            num_experts=args.num_experts,
            hot_experts=hot_experts,
            hot_bias=args.hot_bias,
            seed=args.seed,
        )

    if logits.ndim != 3:
        raise ValueError("router logits must have shape [layers, tokens, experts]")

    num_layers, num_tokens, num_experts = logits.shape
    probabilities = softmax(logits)
    true_topk = topk_indices(probabilities, args.top_k)

    hot_expert_scores = probabilities.mean(axis=(0, 1))
    hot_experts = topk_indices(hot_expert_scores[None, :], args.hot_expert_count)[
        0
    ].tolist()

    placements = [
        contiguous_placement(num_experts, args.ep_size),
        round_robin_placement(num_experts, args.ep_size),
        random_placement(num_experts, args.ep_size, args.seed),
    ]
    placements.append(hot_grouped_placement(num_experts, args.ep_size, hot_experts))
    placements.append(hot_replicated_placement(placements[0], hot_experts))

    rows = []
    for placement in placements:
        rows.extend(
            evaluate_placement(
                placement=placement,
                probabilities=probabilities,
                true_topk=true_topk,
            )
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
                    "num_layers": num_layers,
                    "num_tokens": num_tokens,
                    "num_experts": num_experts,
                    "top_k": args.top_k,
                    "ep_size": args.ep_size,
                    "hot_experts": hot_experts,
                    "rows": rows,
                },
                f,
                indent=2,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
