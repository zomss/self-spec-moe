# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""G98-0 harness validator: deterministic synthetic proofs, CPU-only.

Runs the registered proof cases against the three harness modules and
writes a machine-readable validation artifact. Exits non-zero if any
proof fails. No GPU, engine, or Phase 97 source is touched.

Usage:
    .venv/bin/python research/98_selector_demo/scripts/validate_w98_harness.py \
        [--output research/98_selector_demo/data/g98_0/w98_harness_validation.json]
"""

from __future__ import annotations

import argparse
import json
import math
import random
from itertools import combinations
from pathlib import Path

import w98_accounting as acc
import w98_cost_model as cost
import w98_knapsack as knap

KMAX = 4


def _chain(rid: str, p: float, n: int, seed: int) -> list[acc.StepRow]:
    rng = random.Random(seed)
    rows, u = [], 0
    for _ in range(n):
        accepted = 0
        for _ in range(KMAX):
            if rng.random() < p:
                accepted += 1
            else:
                break
        rows.append(acc.StepRow(rid, u, True, KMAX, accepted))
        u += accepted + 1
    return rows


def proof_closure() -> bool:
    rows = [
        acc.StepRow("r0", 0, True, KMAX, 3),
        acc.StepRow("r0", 4, False, 0, 0),
        acc.StepRow("r1", 0, True, KMAX, 4, clipped=2),
    ]
    result = acc.close_interval(rows)
    if result.e_emitted + result.c_clipped != result.a_accepted + result.h_steps:
        return False
    try:
        acc.close_interval(rows, emitted_total=result.e_emitted + 1)
    except acc.W98AccountingError:
        return True
    return False


def proof_tau_recovery() -> bool:
    p = 0.7
    rows: list[acc.StepRow] = []
    for rid in range(64):
        rows.extend(_chain(f"r{rid}", p, 400, seed=rid))
    (bucket,) = acc.bin_rows(rows, KMAX, []).values()
    for k in range(1, KMAX + 1):
        expected = 1.0 + sum(p ** (i + 1) for i in range(k))
        if abs(acc.tau_at_depth(bucket, k) / expected - 1.0) > 0.02:
            return False
    return True


def proof_u_binning() -> bool:
    edges = [256, 1024]
    return (
        acc.bucket_index(edges, 255) == 0
        and acc.bucket_index(edges, 256) == 1
        and acc.bucket_index(edges, 1024) == 2
    )


def proof_mixed_k_rejected() -> bool:
    rows = [
        acc.StepRow("r0", 0, True, KMAX, 1),
        acc.StepRow("r0", 2, True, 2, 1),
    ]
    try:
        acc.bin_rows(rows, KMAX, [])
    except acc.W98AccountingError:
        return True
    return False


def _singles(deviations: list[float] | None = None) -> list[cost.LeverPoint]:
    configs = [
        (8.0e9, 1.0e9, 1.0),
        (2.0e9, 1.0e9, 1.0),
        (8.0e9, 0.25e9, 1.0),
        (8.0e9, 2.0e9, 1.0),
        (8.0e9, 4.0e9, 1.0),
        (8.0e9, 1.0e9, 1.0 - 4 / 36),
        (8.0e9, 1.0e9, 1.0 - 8 / 36),
    ]
    deviations = deviations or [1.0] * len(configs)
    return [
        cost.LeverPoint(w, kv, keep, keep * (w * 2.0e-9 + kv * 0.5e-9 + 1.0) * d)
        for (w, kv, keep), d in zip(configs, deviations)
    ]


def proof_factored_exact() -> bool:
    model = cost.FactoredCostModel.fit(_singles())
    w, kv, keep = 2.0e9, 0.25e9, 1.0 - 8 / 36
    truth = keep * (w * 2.0e-9 + kv * 0.5e-9 + 1.0)
    return abs(model.predict(w, kv, keep) / truth - 1.0) < 1e-9


def proof_elimination_soundness() -> bool:
    rng = random.Random(23)
    fired = 0
    for _ in range(500):
        q_true = rng.uniform(0.2, 8.0)
        q_lo = q_true * rng.uniform(0.8, 1.0)
        k = rng.choice([2, 4])
        if cost.eliminate(k, q_lo):
            fired += 1
            if (k + 1) / q_true >= 1.015:
                return False
    return fired > 0


def proof_knapsack_brute_force() -> bool:
    rng = random.Random(3)
    r = [random.Random(2).uniform(0.5, 1.0) for _ in range(8)]
    costs = [rng.choice([0.05, 0.1, 0.2]) for _ in range(8)]
    target = 0.35
    chosen = knap.knapsack_select(r, costs, target, scale=100)
    value = sum(math.log(r[i]) for i in chosen)
    best = float("-inf")
    for size in range(1, 9):
        for subset in combinations(range(8), size):
            if sum(costs[i] for i in subset) < target:
                continue
            best = max(best, sum(math.log(r[i]) for i in subset))
    return abs(value - best) < 1e-12


def proof_greedy_conflict() -> bool:
    r = [0.99, 0.98, 0.90, 0.85, 0.80, 0.75]

    def measured(layer_set: frozenset[int]) -> float:
        value = math.prod(r[i] for i in layer_set)
        if {0, 1} <= set(layer_set):
            value *= 0.5
        return value

    greedy_pick, _ = knap.greedy_forward(measured, 6, 2)
    bound_pick = knap.top_k_by_retention(r, 2)
    return measured(greedy_pick) > measured(bound_pick)


PROOFS = {
    "closure_identity": proof_closure,
    "tau_recovery_all_depths": proof_tau_recovery,
    "u_binning_right_open": proof_u_binning,
    "mixed_k_rejected": proof_mixed_k_rejected,
    "factored_model_exact": proof_factored_exact,
    "elimination_soundness_sweep": proof_elimination_soundness,
    "knapsack_matches_brute_force": proof_knapsack_brute_force,
    "greedy_escalation_beats_bound": proof_greedy_conflict,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_output = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "g98_0"
        / "w98_harness_validation.json"
    )
    parser.add_argument("--output", type=Path, default=default_output)
    args = parser.parse_args()

    results = {name: bool(fn()) for name, fn in PROOFS.items()}
    artifact = {
        "artifact_id": "w98-g98-0-harness-validation",
        "phase": 98,
        "gate": "G98-0",
        "gpu_used": False,
        "proofs": results,
        "all_passed": all(results.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    for name, passed in results.items():
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
    print(f"artifact: {args.output}")
    return 0 if artifact["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
