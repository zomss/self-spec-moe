# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synthetic proofs for the G98-0 skip-identity selection harness."""

import math
import random
from itertools import combinations

import pytest
from w98_knapsack import (
    W98KnapsackError,
    consolidation_matrix,
    greedy_forward,
    knapsack_select,
    product_bound,
    random_k,
    robust_single_set,
    top_k_by_retention,
    worst_k_by_retention,
)


def make_retentions(n, seed):
    rng = random.Random(seed)
    return [rng.uniform(0.5, 1.0) for _ in range(n)]


def brute_force_best(r, costs, target, k=None):
    """Exhaustive argmax of the product bound for small instances."""
    best_set, best_value = None, float("-inf")
    n = len(r)
    sizes = [k] if k is not None else range(1, n + 1)
    for size in sizes:
        for subset in combinations(range(n), size):
            if costs and sum(costs[i] for i in subset) < target:
                continue
            value = sum(math.log(r[i]) for i in subset)
            if value > best_value:
                best_set, best_value = frozenset(subset), value
    return best_set, best_value


class TestSelection:
    def test_top_k_matches_brute_force_under_uniform_costs(self):
        r = make_retentions(8, seed=1)
        for k in (1, 3, 5):
            chosen = top_k_by_retention(r, k)
            _, best_value = brute_force_best(r, costs=None, target=0, k=k)
            value = sum(math.log(r[i]) for i in chosen)
            assert value == pytest.approx(best_value)

    def test_knapsack_matches_brute_force_heterogeneous(self):
        r = make_retentions(8, seed=2)
        rng = random.Random(3)
        costs = [rng.choice([0.05, 0.1, 0.2]) for _ in range(8)]
        target = 0.35
        chosen = knapsack_select(r, costs, target, scale=100)
        assert sum(costs[i] for i in chosen) >= target - 1e-9
        _, best_value = brute_force_best(r, costs, target)
        value = sum(math.log(r[i]) for i in chosen)
        assert value == pytest.approx(best_value)

    def test_knapsack_uniform_reduces_to_top_k(self):
        r = make_retentions(10, seed=4)
        costs = [0.1] * 10
        chosen = knapsack_select(r, costs, savings_target=0.3, scale=100)
        assert chosen == top_k_by_retention(r, 3)

    def test_infeasible_target_rejected(self):
        with pytest.raises(W98KnapsackError, match="target"):
            knapsack_select([0.9, 0.9], [0.1, 0.1], savings_target=0.5)

    def test_controls_are_valid_and_distinct(self):
        r = [0.99, 0.95, 0.9, 0.7, 0.6, 0.5]
        assert top_k_by_retention(r, 2) == frozenset({0, 1})
        assert worst_k_by_retention(r, 2) == frozenset({4, 5})
        rand = random_k(6, 2, seed=7)
        assert len(rand) == 2
        assert random_k(6, 2, seed=7) == rand

    def test_invalid_retention_rejected(self):
        with pytest.raises(W98KnapsackError, match="outside"):
            top_k_by_retention([0.5, 1.2], 1)


class TestProductBound:
    def test_bound_is_admit_only_under_interaction(self):
        """Synthetic truth with favorable interaction stays above the
        bound, so an admission by the bound is never optimistic."""
        r = make_retentions(8, seed=8)

        def truth(layer_set):
            return math.prod(r[i] for i in layer_set) ** 0.8

        for subset in combinations(range(8), 3):
            bound = product_bound(1.0, r, subset)
            assert truth(subset) >= bound

    def test_bound_composes_with_baseline(self):
        r = [0.9, 0.8]
        assert product_bound(0.5, r, [0, 1]) == pytest.approx(0.36)


class TestGreedyEscalation:
    def test_greedy_avoids_a_planted_conflict(self):
        """Layers 0 and 1 are individually best but conflict when
        composed; greedy sidesteps the pair, the product-bound pick
        cannot see it."""
        r = [0.99, 0.98, 0.90, 0.85, 0.80, 0.75]

        def measured(layer_set):
            value = math.prod(r[i] for i in layer_set)
            if {0, 1} <= set(layer_set):
                value *= 0.5
            return value

        bound_pick = top_k_by_retention(r, 2)
        assert bound_pick == frozenset({0, 1})
        greedy_pick, trace = greedy_forward(measured, 6, 2)
        assert not {0, 1} <= greedy_pick
        assert measured(greedy_pick) > measured(bound_pick)
        assert len(trace) == 2

    def test_greedy_matches_top_k_without_interaction(self):
        r = make_retentions(6, seed=9)

        def measured(layer_set):
            return math.prod(r[i] for i in layer_set)

        greedy_pick, _ = greedy_forward(measured, 6, 3)
        assert greedy_pick == top_k_by_retention(r, 3)


class TestConsolidation:
    def test_matrix_shape_and_diagonal(self):
        sets = {"R5": frozenset({0, 1}), "R1": frozenset({2, 3})}
        profiles = {
            "R5": [0.9, 0.9, 0.6, 0.6],
            "R1": [0.6, 0.6, 0.9, 0.9],
        }
        matrix = consolidation_matrix(sets, profiles)
        assert matrix["R5"]["R5"] == pytest.approx(0.81)
        assert matrix["R5"]["R1"] == pytest.approx(0.36)

    def test_conflicting_regimes_have_no_robust_set(self):
        sets = {"R5": frozenset({0, 1}), "R1": frozenset({2, 3})}
        profiles = {
            "R5": [0.9, 0.9, 0.6, 0.6],
            "R1": [0.6, 0.6, 0.9, 0.9],
        }
        owner, shortfall = robust_single_set(sets, profiles, eps=0.05)
        assert owner is None
        assert shortfall > 0.05

    def test_agreeing_regimes_share_a_robust_set(self):
        shared = frozenset({0, 1})
        sets = {"R5": shared, "R1": shared}
        profiles = {
            "R5": [0.95, 0.94, 0.6, 0.6],
            "R1": [0.93, 0.96, 0.6, 0.6],
        }
        owner, shortfall = robust_single_set(sets, profiles, eps=0.05)
        assert owner is not None
        assert shortfall == pytest.approx(0.0)

    def test_mismatched_regime_keys_rejected(self):
        with pytest.raises(W98KnapsackError, match="regime keys"):
            consolidation_matrix({"R5": frozenset({0})}, {"R1": [0.9]})
