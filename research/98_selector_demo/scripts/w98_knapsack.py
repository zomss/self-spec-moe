# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Skip-identity selection: knapsack with measured profits (Phase 98 G98-0).

Value side = measured per-layer retention (single-layer-skip acceptance
ratio); cost side = Round-1 physics. The product bound composes retentions
admit-only; greedy forward selection is the escalation path when the bound
fails badly. CPU-only; no engine imports.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable, Mapping, Sequence


class W98KnapsackError(ValueError):
    """Invalid input to identity selection."""


def validate_retention(r: Sequence[float]) -> None:
    """Retentions are single-layer acceptance ratios in (0, 1]."""
    if not r:
        raise W98KnapsackError("empty retention vector")
    for i, v in enumerate(r):
        if not 0.0 < v <= 1.0:
            raise W98KnapsackError(f"retention[{i}]={v} outside (0, 1]")


def top_k_by_retention(r: Sequence[float], k: int) -> frozenset[int]:
    """Uniform-cost knapsack: the k least-sensitive layers."""
    validate_retention(r)
    if not 0 < k <= len(r):
        raise W98KnapsackError(f"count {k} outside [1, {len(r)}]")
    order = sorted(range(len(r)), key=lambda i: (-r[i], i))
    return frozenset(order[:k])


def worst_k_by_retention(r: Sequence[float], k: int) -> frozenset[int]:
    """Control: the k most-sensitive layers."""
    validate_retention(r)
    if not 0 < k <= len(r):
        raise W98KnapsackError(f"count {k} outside [1, {len(r)}]")
    order = sorted(range(len(r)), key=lambda i: (r[i], i))
    return frozenset(order[:k])


def random_k(n_layers: int, k: int, seed: int) -> frozenset[int]:
    """Control: a seeded uniform random layer set of size k."""
    if not 0 < k <= n_layers:
        raise W98KnapsackError(f"count {k} outside [1, {n_layers}]")
    return frozenset(random.Random(seed).sample(range(n_layers), k))


def product_bound(f0: float, r: Sequence[float], layer_set: Iterable[int]) -> float:
    """Admit-only lower bound: f(set) >= f0 * prod r_i (C2)."""
    validate_retention(r)
    if f0 <= 0:
        raise W98KnapsackError("f0 must be positive")
    out = f0
    for i in layer_set:
        out *= r[i]
    return out


def knapsack_select(
    r: Sequence[float],
    costs: Sequence[float],
    savings_target: float,
    scale: int = 10_000,
) -> frozenset[int]:
    """Maximizes retained log-acceptance subject to a savings target.

    Exact DP over integer-scaled costs: choose S maximizing
    sum(log r_i) subject to sum(costs_i for i in S) >= savings_target.
    With uniform costs this reduces to top-k; the DP exists for the
    heterogeneous case (MoE/MLA layer mixes).
    """
    validate_retention(r)
    if len(costs) != len(r):
        raise W98KnapsackError("costs and retentions differ in length")
    if any(c <= 0 for c in costs):
        raise W98KnapsackError("costs must be positive")
    int_costs = [round(c * scale) for c in costs]
    target = math.ceil(savings_target * scale)
    total = sum(int_costs)
    if target > total:
        raise W98KnapsackError("savings target exceeds total available")
    values = [math.log(v) for v in r]
    neg_inf = float("-inf")
    parent: dict[tuple[int, int], tuple[int, bool]] = {}
    prev: list[float] = [neg_inf] * (total + 1)
    prev[0] = 0.0
    for i, (ci, vi) in enumerate(zip(int_costs, values)):
        cur = list(prev)
        for s in range(total + 1):
            parent[(i, s)] = (s, False)
        for s in range(total - ci, -1, -1):
            if prev[s] == neg_inf:
                continue
            cand = prev[s] + vi
            if cand > cur[s + ci]:
                cur[s + ci] = cand
                parent[(i, s + ci)] = (s, True)
        prev = cur
    final = prev
    best_s = max(
        (s for s in range(target, total + 1) if final[s] > neg_inf),
        key=lambda s: final[s],
        default=None,
    )
    if best_s is None:
        raise W98KnapsackError("no feasible set reaches the savings target")
    picked: set[int] = set()
    s = best_s
    for i in range(len(r) - 1, -1, -1):
        s, taken = parent[(i, s)]
        if taken:
            picked.add(i)
    return frozenset(picked)


def greedy_forward(
    measure_fn: Callable[[frozenset[int]], float],
    n_layers: int,
    k: int,
) -> tuple[frozenset[int], list[float]]:
    """Escalation path: adds the measured-best layer one at a time.

    Args:
        measure_fn: Measured composed acceptance for a candidate set.
        n_layers: Total layer count.
        k: Target set size.

    Returns:
        The selected set and the measured value after each addition.
    """
    if not 0 < k <= n_layers:
        raise W98KnapsackError(f"count {k} outside [1, {n_layers}]")
    chosen: frozenset[int] = frozenset()
    trace: list[float] = []
    for _ in range(k):
        best_layer, best_value = None, float("-inf")
        for i in range(n_layers):
            if i in chosen:
                continue
            value = measure_fn(chosen | {i})
            if value > best_value:
                best_layer, best_value = i, value
        chosen = chosen | {best_layer}
        trace.append(best_value)
    return chosen, trace


def consolidation_matrix(
    sets_by_regime: Mapping[str, frozenset[int]],
    r_by_regime: Mapping[str, Sequence[float]],
) -> dict[str, dict[str, float]]:
    """Bound-predicted retention of each regime's set under every regime."""
    if set(sets_by_regime) != set(r_by_regime):
        raise W98KnapsackError("regime keys differ between sets and profiles")
    return {
        owner: {
            g: product_bound(1.0, r_by_regime[g], layer_set)
            for g in sorted(r_by_regime)
        }
        for owner, layer_set in sets_by_regime.items()
    }


def robust_single_set(
    sets_by_regime: Mapping[str, frozenset[int]],
    r_by_regime: Mapping[str, Sequence[float]],
    eps: float,
) -> tuple[str | None, float]:
    """Finds one set within eps of every regime's own choice, if any.

    Returns the owning regime of the best single candidate and its
    maximum relative retention shortfall versus each regime's own set;
    the candidate name is None when no set is within eps everywhere.
    """
    matrix = consolidation_matrix(sets_by_regime, r_by_regime)
    own = {g: matrix[g][g] for g in matrix}
    best_owner, best_shortfall = None, float("inf")
    for owner in sorted(matrix):
        shortfall = max(1.0 - matrix[owner][g] / own[g] for g in sorted(own))
        if shortfall < best_shortfall:
            best_owner, best_shortfall = owner, shortfall
    if best_shortfall <= eps:
        return best_owner, best_shortfall
    return None, best_shortfall
