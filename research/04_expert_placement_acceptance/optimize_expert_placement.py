#!/usr/bin/env python3
"""Optimize and evaluate expert placement from router traces."""

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
    replicated: bool = False


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


def capacities(num_experts: int, ep_size: int) -> list[int]:
    base = num_experts // ep_size
    remainder = num_experts % ep_size
    return [base + int(rank < remainder) for rank in range(ep_size)]


def contiguous_placement(num_experts: int, ep_size: int) -> Placement:
    caps = capacities(num_experts, ep_size)
    local_experts = []
    cursor = 0
    for cap in caps:
        local_experts.append(set(range(cursor, cursor + cap)))
        cursor += cap
    return Placement("contiguous", local_experts)


def random_placement(num_experts: int, ep_size: int, seed: int) -> Placement:
    rng = np.random.default_rng(seed)
    experts = np.arange(num_experts)
    rng.shuffle(experts)
    caps = capacities(num_experts, ep_size)
    local_experts = []
    cursor = 0
    for cap in caps:
        local_experts.append(set(experts[cursor : cursor + cap].tolist()))
        cursor += cap
    return Placement("random", local_experts)


def load_balanced_placement(load: np.ndarray, ep_size: int) -> Placement:
    num_experts = load.shape[0]
    caps = capacities(num_experts, ep_size)
    local_experts = [set() for _ in range(ep_size)]
    rank_load = np.zeros(ep_size, dtype=np.float64)
    for expert in np.argsort(-load).tolist():
        candidates = [
            rank
            for rank in range(ep_size)
            if len(local_experts[rank]) < caps[rank]
        ]
        rank = min(candidates, key=lambda r: rank_load[r])
        local_experts[rank].add(expert)
        rank_load[rank] += load[expert]
    return Placement("load_balanced", local_experts)


def coactivation_matrix(true_topk: np.ndarray, num_experts: int) -> np.ndarray:
    matrix = np.zeros((num_experts, num_experts), dtype=np.float64)
    flat_topk = true_topk.reshape(-1, true_topk.shape[-1])
    for experts in flat_topk:
        for i, expert_i in enumerate(experts):
            for expert_j in experts[i + 1 :]:
                matrix[expert_i, expert_j] += 1.0
                matrix[expert_j, expert_i] += 1.0
    return matrix


def coactivation_greedy_placement(
    load: np.ndarray,
    coactivation: np.ndarray,
    ep_size: int,
) -> Placement:
    num_experts = load.shape[0]
    caps = capacities(num_experts, ep_size)
    local_experts = [set() for _ in range(ep_size)]
    sorted_experts = np.argsort(-load).tolist()

    seeds = sorted_experts[:ep_size]
    for rank, expert in enumerate(seeds):
        local_experts[rank].add(expert)

    assigned = set(seeds)
    for expert in sorted_experts[ep_size:]:
        best_rank = None
        best_score = -float("inf")
        for rank in range(ep_size):
            if len(local_experts[rank]) >= caps[rank]:
                continue
            affinity = sum(coactivation[expert, other] for other in local_experts[rank])
            balance = 1.0 - (len(local_experts[rank]) / caps[rank])
            score = affinity + 0.01 * load[expert] + 0.001 * balance
            if score > best_score:
                best_score = score
                best_rank = rank
        assert best_rank is not None
        local_experts[best_rank].add(expert)
        assigned.add(expert)
    assert len(assigned) == num_experts
    return Placement("coactivation_greedy", local_experts)


def replicate_hot_experts(base: Placement, hot_experts: list[int]) -> Placement:
    local_experts = [set(experts).union(hot_experts) for experts in base.local_experts_by_rank]
    return Placement(f"{base.name}_hot_replicated", local_experts, replicated=True)


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
) -> list[dict[str, str | float | int | bool]]:
    num_layers, num_tokens, _ = probabilities.shape
    top_k = true_topk.shape[-1]
    rows: list[dict[str, str | float | int | bool]] = []

    rank_gamma = []
    rank_overlap = []
    rank_full_topk = []
    rank_top1 = []
    rank_local_counts = []

    true_top1 = true_topk[..., 0]
    for rank, local_experts in enumerate(placement.local_experts_by_rank):
        local_list = sorted(local_experts)
        mask = np.zeros(probabilities.shape[-1], dtype=bool)
        mask[local_list] = True
        gamma = probabilities[:, :, mask].sum(axis=-1)
        overlap = np.isin(true_topk, local_list).sum(axis=-1) / top_k
        full_topk = (overlap == 1.0).astype(np.float32)
        top1_local = np.isin(true_top1, local_list).astype(np.float32)

        rank_gamma.append(gamma)
        rank_overlap.append(overlap)
        rank_full_topk.append(full_topk)
        rank_top1.append(top1_local)
        rank_local_counts.append(len(local_experts))

        gamma_stats = summarize(gamma)
        overlap_stats = summarize(overlap)
        rows.append(
            {
                "placement": placement.name,
                "replicated": placement.replicated,
                "policy": "fixed_rank",
                "rank": rank,
                "local_experts": len(local_experts),
                "gamma_mean": gamma_stats["mean"],
                "gamma_p50": gamma_stats["p50"],
                "gamma_p90": gamma_stats["p90"],
                "topk_overlap_mean": overlap_stats["mean"],
                "topk_overlap_p50": overlap_stats["p50"],
                "topk_overlap_p90": overlap_stats["p90"],
                "top1_local_rate": float(np.mean(top1_local)),
                "full_topk_local_rate": float(np.mean(full_topk)),
            }
        )

    gamma_by_rank = np.stack(rank_gamma, axis=-1)
    overlap_by_rank = np.stack(rank_overlap, axis=-1)
    full_topk_by_rank = np.stack(rank_full_topk, axis=-1)
    top1_by_rank = np.stack(rank_top1, axis=-1)
    best_rank = np.argmax(gamma_by_rank, axis=-1)
    layer_idx = np.arange(num_layers)[:, None]
    token_idx = np.arange(num_tokens)[None, :]

    best_gamma = gamma_by_rank[layer_idx, token_idx, best_rank]
    best_overlap = overlap_by_rank[layer_idx, token_idx, best_rank]
    best_full_topk = full_topk_by_rank[layer_idx, token_idx, best_rank]
    best_top1 = top1_by_rank[layer_idx, token_idx, best_rank]

    gamma_stats = summarize(best_gamma)
    overlap_stats = summarize(best_overlap)
    rows.append(
        {
            "placement": placement.name,
            "replicated": placement.replicated,
            "policy": "best_gamma_rank",
            "rank": -1,
            "local_experts": float(np.mean(rank_local_counts)),
            "gamma_mean": gamma_stats["mean"],
            "gamma_p50": gamma_stats["p50"],
            "gamma_p90": gamma_stats["p90"],
            "topk_overlap_mean": overlap_stats["mean"],
            "topk_overlap_p50": overlap_stats["p50"],
            "topk_overlap_p90": overlap_stats["p90"],
            "top1_local_rate": float(np.mean(best_top1)),
            "full_topk_local_rate": float(np.mean(best_full_topk)),
        }
    )
    return rows


def load_router_logits(path: Path) -> np.ndarray:
    data = np.load(path)
    if "router_logits" not in data:
        raise ValueError("NPZ must contain router_logits")
    logits = data["router_logits"]
    if logits.ndim != 3:
        raise ValueError("router_logits must have shape [layers, tokens, experts]")
    return logits


def parse_int_list(raw_value: str) -> list[int]:
    values = [int(value) for value in raw_value.split(",") if value]
    if not values:
        raise argparse.ArgumentTypeError("at least one EP size is required")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-logits", type=Path, required=True)
    parser.add_argument("--ep-sizes", type=parse_int_list, default=[2, 4, 8])
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--replicate-hot-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="placement")
    args = parser.parse_args()

    logits = load_router_logits(args.router_logits)
    probabilities = softmax(logits)
    true_topk = topk_indices(probabilities, args.top_k)
    expert_load = probabilities.mean(axis=(0, 1))
    coactivation = coactivation_matrix(true_topk, logits.shape[-1])
    hot_experts = np.argsort(-expert_load)[: args.replicate_hot_count].tolist()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    placements_json = {}
    for ep_size in args.ep_sizes:
        base_placements = [
            contiguous_placement(logits.shape[-1], ep_size),
            random_placement(logits.shape[-1], ep_size, args.seed),
            load_balanced_placement(expert_load, ep_size),
            coactivation_greedy_placement(expert_load, coactivation, ep_size),
        ]
        placements = []
        for placement in base_placements:
            placements.append(placement)
            placements.append(replicate_hot_experts(placement, hot_experts))

        placements_json[str(ep_size)] = {
            placement.name: [
                sorted(experts) for experts in placement.local_experts_by_rank
            ]
            for placement in placements
        }

        for placement in placements:
            for row in evaluate_placement(
                placement=placement,
                probabilities=probabilities,
                true_topk=true_topk,
            ):
                all_rows.append({"ep_size": ep_size, **row})

    placement_path = args.output_dir / f"{args.prefix}_maps.json"
    with placement_path.open("w") as f:
        json.dump(
            {
                "router_logits": str(args.router_logits),
                "top_k": args.top_k,
                "hot_experts": hot_experts,
                "placements": placements_json,
            },
            f,
            indent=2,
        )

    metrics_path = args.output_dir / f"{args.prefix}_coverage_metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    print(json.dumps({"placement_maps": str(placement_path), "metrics": str(metrics_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
