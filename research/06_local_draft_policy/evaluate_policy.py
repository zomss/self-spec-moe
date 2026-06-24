#!/usr/bin/env python3
"""Evaluate simple local-draft policies from routing and acceptance metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def row_key(row: dict[str, str]) -> tuple[str, int]:
    return row["placement"], int(row["rank"])


def optional_float(row: dict[str, str], key: str, fallback: float) -> float:
    value = row.get(key)
    if value in (None, ""):
        return fallback
    return float(value)


def join_metrics(
    routing_rows: list[dict[str, str]],
    acceptance_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    routing_by_key = {
        row_key(row): row
        for row in routing_rows
        if row["policy"] == "fixed_rank"
    }
    rows = []
    for acc in acceptance_rows:
        key = row_key(acc)
        if key not in routing_by_key:
            continue
        route = routing_by_key[key]
        rows.append(
            {
                "placement": acc["placement"],
                "rank": int(acc["rank"]),
                "draft_top_k": int(acc["draft_top_k"]),
                "local_experts": int(acc["local_experts"]),
                "gamma": float(route["gamma_mean"]),
                "topk_overlap": float(route["topk_overlap_mean"]),
                "top1_local_rate": optional_float(
                    route,
                    "top1_local_rate",
                    float(route["topk_overlap_mean"]),
                ),
                "full_topk_local_rate": float(route["full_topk_local_rate"]),
                "expected_acceptance": float(acc["expected_acceptance_overlap"]),
                "sampled_acceptance": float(acc["sampled_acceptance_rate"]),
                "top1_token_match": float(acc["top1_match_rate"]),
                "kl_pq": float(acc["kl_pq_mean"]),
            }
        )
    return rows


def threshold_values(values: list[float]) -> list[float]:
    candidates = sorted(set(round(value, 3) for value in values))
    return [0.0] + candidates


def evaluate_policies(joined_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    gamma_thresholds = threshold_values([float(row["gamma"]) for row in joined_rows])
    top1_thresholds = threshold_values(
        [float(row["top1_local_rate"]) for row in joined_rows]
    )
    overlap_thresholds = threshold_values(
        [float(row["topk_overlap"]) for row in joined_rows]
    )
    draft_top_ks = sorted(set(int(row["draft_top_k"]) for row in joined_rows))

    policy_rows = []
    total_rows = len(joined_rows)
    for draft_top_k in draft_top_ks:
        rows_for_k = [
            row for row in joined_rows if int(row["draft_top_k"]) == draft_top_k
        ]
        for gamma_threshold in gamma_thresholds:
            for top1_threshold in top1_thresholds:
                for overlap_threshold in overlap_thresholds:
                    selected = [
                        row
                        for row in rows_for_k
                        if float(row["gamma"]) >= gamma_threshold
                        and float(row["top1_local_rate"]) >= top1_threshold
                        and float(row["topk_overlap"]) >= overlap_threshold
                    ]
                    if not selected:
                        continue
                    sampled = [float(row["sampled_acceptance"]) for row in selected]
                    expected = [
                        float(row["expected_acceptance"]) for row in selected
                    ]
                    policy_rows.append(
                        {
                            "draft_top_k": draft_top_k,
                            "gamma_threshold": gamma_threshold,
                            "top1_local_threshold": top1_threshold,
                            "topk_overlap_threshold": overlap_threshold,
                            "selected_count": len(selected),
                            "selected_fraction": len(selected) / total_rows,
                            "sampled_acceptance_mean": float(np.mean(sampled)),
                            "sampled_acceptance_max": float(np.max(sampled)),
                            "expected_acceptance_mean": float(np.mean(expected)),
                            "placements": ";".join(
                                f"{row['placement']}:{row['rank']}" for row in selected
                            ),
                        }
                    )
    policy_rows.sort(
        key=lambda row: (
            float(row["sampled_acceptance_mean"]),
            float(row["selected_fraction"]),
        ),
        reverse=True,
    )
    return policy_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routing-metrics", type=Path, required=True)
    parser.add_argument("--acceptance-metrics", type=Path, required=True)
    parser.add_argument("--output-joined-csv", type=Path, required=True)
    parser.add_argument("--output-policy-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    joined = join_metrics(
        read_csv(args.routing_metrics),
        read_csv(args.acceptance_metrics),
    )
    if not joined:
        raise ValueError("No joined rows produced")
    policies = evaluate_policies(joined)
    write_csv(args.output_joined_csv, joined)
    write_csv(args.output_policy_csv, policies)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as f:
            json.dump(
                {
                    "joined_rows": joined,
                    "policy_rows": policies[:50],
                },
                f,
                indent=2,
            )
    print(json.dumps({"joined": len(joined), "policies": len(policies)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
